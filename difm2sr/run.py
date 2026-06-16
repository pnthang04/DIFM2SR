import argparse
from logging import getLogger, FileHandler
import torch
import inspect
import numpy as np

# RecBole versions used by this project still reference NumPy 1.x scalar aliases
# during Config initialization. NumPy 2 removed some of them, so restore the
# canonical dtype names before RecBole applies its compatibility settings.
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color


from data.dataset import DIFM2SRDataset
from collections import OrderedDict

from difm2sr import DIFM2SR

from recbole.trainer import Trainer


def cfg_get(config, key, default=None):
    return config[key] if key in config else default


def torch_load_compat(path, map_location=None):
    kwargs = {}
    if map_location is not None:
        kwargs['map_location'] = map_location
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def load_model_checkpoint(model, checkpoint_path, logger):
    checkpoint = torch_load_compat(checkpoint_path, map_location=model.device)
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        f"Loaded baseline checkpoint from {checkpoint_path}. "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def get_logger_filename(logger):
    file_handler = next((handler for handler in logger.handlers if isinstance(handler, FileHandler)), None)
    if file_handler:
        filename = file_handler.baseFilename
        print(f"The log file name is {filename}")
    else:
        raise Exception("No file handler found in logger")
    return filename


def run_smoke_train_steps(config, model, train_data, steps):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    model.train()
    losses = []
    for step, interaction in enumerate(train_data, start=1):
        if step > steps:
            break
        interaction = interaction.to(config['device'])
        optimizer.zero_grad()
        loss = model.calculate_loss(interaction)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        print(f"smoke step {step}/{steps} loss={loss_value:.6f}")
    return losses


def pretrain_baseline_epochs(config, model, train_data, epochs, logger):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        steps = 0
        for interaction in train_data:
            interaction = interaction.to(config['device'])
            optimizer.zero_grad()
            loss = model.calculate_loss(interaction)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
        msg = f"[diffusion] baseline pretrain epoch {epoch + 1}/{epochs} loss={total_loss / max(steps, 1):.6f}"
        print(msg)
        logger.info(msg)


def prepare_diffusion_denoiser(config, model, train_data, logger):
    if not bool(cfg_get(config, 'use_diffusion_denoiser', False)):
        return

    checkpoint_path = str(cfg_get(config, 'baseline_checkpoint_path', '') or '')
    if checkpoint_path:
        load_model_checkpoint(model, checkpoint_path, logger)
    else:
        pretrain_epochs = int(cfg_get(config, 'baseline_pretrain_epochs', 5))
        if pretrain_epochs <= 0:
            logger.warning(
                "Diffusion denoiser enabled without baseline checkpoint and baseline_pretrain_epochs <= 0. "
                "The ID condition will come from the current model weights."
            )
        else:
            pretrain_baseline_epochs(config, model, train_data, pretrain_epochs, logger)

    model.warmup_diffusion_denoiser(
        epochs=int(cfg_get(config, 'diffusion_warmup_epochs', 5)),
        batch_size=int(cfg_get(config, 'diffusion_batch_size', 1024)),
        lr=float(cfg_get(config, 'diffusion_lr', 0.001)),
        beta_graph=float(cfg_get(config, 'beta_graph', 0.01)),
        logger=logger,
    )
    clean_text, clean_image = model.generate_clean_modal_features(
        batch_size=int(cfg_get(config, 'diffusion_batch_size', 1024))
    )
    blend_alpha = float(cfg_get(config, 'denoiser_blend_alpha', 0.2))
    blend_alpha = min(max(blend_alpha, 0.0), 1.0)
    text_alpha = float(cfg_get(config, 'denoiser_text_blend_alpha', -1.0))
    image_alpha = float(cfg_get(config, 'denoiser_image_blend_alpha', -1.0))
    text_alpha = blend_alpha if text_alpha < 0 else min(max(text_alpha, 0.0), 1.0)
    image_alpha = blend_alpha if image_alpha < 0 else min(max(image_alpha, 0.0), 1.0)
    # Blend clean/raw modal features per modality so diffusion can nudge weak modalities more gently.
    final_text = text_alpha * clean_text + (1.0 - text_alpha) * model.raw_text_embs
    final_image = image_alpha * clean_image + (1.0 - image_alpha) * model.raw_img_embs
    final_text[0].zero_()
    final_image[0].zero_()
    model.build_modal_structures(final_text, final_image)
    for p in model.diffusion_denoiser.parameters():
        p.requires_grad = False
    msg = (
        f"[diffusion] rebuilt CGC/CIP with blended features "
        f"alpha={blend_alpha} text_alpha={text_alpha} image_alpha={image_alpha}: "
        f"co_vm_adj={tuple(model.co_vm_adj.shape)} co_tm_adj={tuple(model.co_tm_adj.shape)} "
        f"sensev={tuple(model.sensev.shape)} senset={tuple(model.senset.shape)}"
    )
    print(msg)
    logger.info(msg)


def run(dataset, setting='DIFM2SR.yaml,run.yaml',log_prefix="", **kwargs):
    smoke_steps = int(kwargs.pop('smoke_steps', 0) or 0)
    setting = setting.split(',')
    config = Config(model=DIFM2SR, dataset=dataset, config_file_list=setting, config_dict=kwargs)

    config['log_prefix'] = log_prefix

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = DIFM2SRDataset(config)

    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    # --- Robustness for custom datasets/configs ---
    # Some prepared `.inter` files already contain longer sequences than the config's
    # MAX_ITEM_LIST_LENGTH. If the model's position embedding is smaller than the
    # actual padded sequence length, CUDA will crash with an index-out-of-bounds.
    try:
        sample_batch = next(iter(train_data))
        if 'item_id_list' in sample_batch and hasattr(sample_batch['item_id_list'], 'shape'):
            actual_seq_len = int(sample_batch['item_id_list'].shape[1])
            cfg_max_len = int(config['MAX_ITEM_LIST_LENGTH'])
            if actual_seq_len > cfg_max_len:
                logger.warning(
                    f"Detected item_id_list padded length {actual_seq_len} > MAX_ITEM_LIST_LENGTH {cfg_max_len}. "
                    f"Auto-updating MAX_ITEM_LIST_LENGTH to {actual_seq_len} to avoid position-embedding OOB."
                )
                config['MAX_ITEM_LIST_LENGTH'] = actual_seq_len
    except Exception as e:
        logger.warning(f"Unable to infer actual sequence length from dataloader: {e}")

    # With CE loss, keep/ensure a dict-style train_neg_sample_args because the
    # RecBole Trainer calls `.get()` on it.
    try:
        if str(config['loss_type']).upper() == 'CE':
            tna = cfg_get(config, 'train_neg_sample_args', None)
            if tna is None:
                config['train_neg_sample_args'] = {
                    'distribution': 'none',
                    'sample_num': 'none',
                    'alpha': 'none',
                    'dynamic': False,
                    'candidate_num': 0,
                }
    except Exception:
        pass
    
    # Extract co-occurrence data from the dataset
    # Get item sequences from the inter_feat
    import numpy as np
    
    # Get the item_id_list sequences from the dataset
    if hasattr(train_data.dataset, 'field2id_token'):
        # Build co-occurrence matrix from sequences
        item_seq_field = 'item_id_list'
        if item_seq_field in train_data.dataset.inter_feat:
            seq_tensor = train_data.dataset.inter_feat[item_seq_field]
            # seq_tensor shape: (num_interactions, max_seq_length)
            co_data = seq_tensor.numpy() if hasattr(seq_tensor, 'numpy') else np.array(seq_tensor)
            # Get lengths from the dataset if available
            if hasattr(train_data.dataset, 'item_list_length'):
                co_lens = train_data.dataset.item_list_length
            else:
                # Compute lengths by counting non-zero/non-padding elements
                co_lens = np.sum(co_data != 0, axis=1) if co_data.ndim > 1 else np.array([len(co_data)])
        else:
            # Fallback: create dummy co-occurrence data
            co_data = np.zeros((len(train_data.dataset), 50), dtype=np.int64)
            co_lens = np.ones(len(train_data.dataset), dtype=np.int64)
    else:
        co_data = np.zeros((len(train_data.dataset), 50), dtype=np.int64)
        co_lens = np.ones(len(train_data.dataset), dtype=np.int64)
    
    model = DIFM2SR(config, train_data.dataset, co_data, co_lens).to(config['device'])
    prepare_diffusion_denoiser(config, model, train_data, logger)
    logger.info(model)
    trainer = Trainer(config, model)

    if smoke_steps > 0:
        losses = run_smoke_train_steps(config, model, train_data, smoke_steps)
        logger.info(f"Smoke train completed for {len(losses)} steps. losses={losses}")
        return config['model'], config['dataset'], {'smoke_losses': losses}

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # PyTorch 2.6+ changed torch.load default to weights_only=True, which can
    # break loading RecBole checkpoints (pickle instructions not supported by
    # the weights-only unpickler). Since we are loading our own local checkpoint
    # here, force weights_only=False during evaluation.
    test_result = None
    _orig_torch_load = None
    try:
        if 'weights_only' in inspect.signature(torch.load).parameters:
            _orig_torch_load = torch.load

            def _torch_load_compat(*args, **kwargs):
                kwargs.setdefault('weights_only', False)
                return _orig_torch_load(*args, **kwargs)

            torch.load = _torch_load_compat

        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
    finally:
        if _orig_torch_load is not None:
            torch.load = _orig_torch_load

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    logger_Filename = get_logger_filename(logger)
    logger.info(f"Write log to {logger_Filename}")

    # Save evaluation results to JSON
    import json
    import os
    from datetime import datetime
    
    results = {
        'dataset': config['dataset'],
        'best_valid_result': dict(best_valid_result),
        'test_result': dict(test_result) if test_result else None,
        'best_valid_score': float(best_valid_score) if best_valid_score else None,
    }
    
    # Create results directory if not exists
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save to JSON with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dataset_name_clean = config['dataset'].replace(' ', '_').replace('\n', '').replace('\x1b', '')
    result_filename = os.path.join(results_dir, f'DIFM2SR-{dataset_name_clean}-{log_prefix}{timestamp}.json')
    with open(result_filename, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {result_filename}")

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='baby', help='dataset name')
    parser.add_argument('-f', type=bool, default=True)
    parser.add_argument('-setting', type=str, default='DIFM2SR.yaml,run.yaml')
    parser.add_argument('-note', type=str, default='')
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--use-diffusion-denoiser', action='store_true')
    parser.add_argument('--baseline-checkpoint-path', type=str, default='')
    parser.add_argument('--baseline-pretrain-epochs', type=int, default=5)
    parser.add_argument('--diffusion-warmup-epochs', type=int, default=5)
    parser.add_argument('--diffusion-lr', type=float, default=0.001)
    parser.add_argument('--diffusion-batch-size', type=int, default=1024)
    parser.add_argument('--diffusion-steps', type=int, default=4)
    parser.add_argument('--beta-graph', type=float, default=0.001)
    parser.add_argument('--denoiser-blend-alpha', type=float, default=0.1)
    parser.add_argument('--denoiser-text-blend-alpha', type=float, default=-1.0)
    parser.add_argument('--denoiser-image-blend-alpha', type=float, default=-1.0)
    parser.add_argument('--condition-type', type=str, default='id_graph')
    parser.add_argument('--w-balw', type=float, default=0.0)
    parser.add_argument('--balw-type', type=str, default='sample', choices=['sample', 'batch', 'threshold'])
    parser.add_argument('--balw-max-weight', type=float, default=0.75)
    parser.add_argument('--balw-log-batches', type=int, default=5)
    parser.add_argument('--w-simw', type=float, default=0.0)
    parser.add_argument('--simw-mu', type=float, default=2.0)
    parser.add_argument('--simw-target', type=str, default='router', choices=['router', 'final'])
    parser.add_argument('--simw-log-batches', type=int, default=5)
    args, unparsed = parser.parse_known_args()
    print(args)

    run(
        args.d,
        setting=args.setting,
        fix_enc=args.f,
        log_prefix=args.note,
        smoke_steps=args.smoke_steps,
        use_diffusion_denoiser=args.use_diffusion_denoiser,
        baseline_checkpoint_path=args.baseline_checkpoint_path,
        baseline_pretrain_epochs=args.baseline_pretrain_epochs,
        diffusion_warmup_epochs=args.diffusion_warmup_epochs,
        diffusion_lr=args.diffusion_lr,
        diffusion_batch_size=args.diffusion_batch_size,
        diffusion_steps=args.diffusion_steps,
        beta_graph=args.beta_graph,
        denoiser_blend_alpha=args.denoiser_blend_alpha,
        denoiser_text_blend_alpha=args.denoiser_text_blend_alpha,
        denoiser_image_blend_alpha=args.denoiser_image_blend_alpha,
        condition_type=args.condition_type,
        w_balw=args.w_balw,
        balw_type=args.balw_type,
        balw_max_weight=args.balw_max_weight,
        balw_log_batches=args.balw_log_batches,
        w_simw=args.w_simw,
        simw_mu=args.simw_mu,
        simw_target=args.simw_target,
        simw_log_batches=args.simw_log_batches,
    )
