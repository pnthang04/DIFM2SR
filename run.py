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


from data.dataset import SGPDataset
from collections import OrderedDict

from sgp import SGP

from recbole.trainer import Trainer

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


def run(dataset, setting='SGP4SR.yaml,run.yaml',log_prefix="", **kwargs):
    smoke_steps = int(kwargs.pop('smoke_steps', 0) or 0)
    setting = setting.split(',')
    config = Config(model=SGP, dataset=dataset, config_file_list=setting, config_dict=kwargs)

    config['log_prefix'] = log_prefix

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = SGPDataset(config)

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
            tna = config.get('train_neg_sample_args', None)
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
    
    model = SGP(config, train_data.dataset, co_data, co_lens).to(config['device'])
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
    result_filename = os.path.join(results_dir, f'SGP-{dataset_name_clean}-{log_prefix}{timestamp}.json')
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
    parser.add_argument('-setting', type=str, default='SGP4SR.yaml,run.yaml')
    parser.add_argument('-note', type=str, default='')
    parser.add_argument('--smoke-steps', type=int, default=0)
    args, unparsed = parser.parse_known_args()
    print(args)

    run(args.d, setting=args.setting, fix_enc=args.f, log_prefix=args.note, smoke_steps=args.smoke_steps)
