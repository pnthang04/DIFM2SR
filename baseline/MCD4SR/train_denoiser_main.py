'''NDCG@20'''
import torch
import time
from accelerate import Accelerator
from baseline.MCD4SR.data.processed import load_seq_datasets
from baseline.MCD4SR.data.utils import batch_to
from baseline.MCD4SR.evaluate.metrics import MetricsAccumulator
from baseline.MCD4SR.modules.model_denoiser_main import EncoderDecoderRetrievalModel
from torch.utils.data import DataLoader
from baseline.MCD4SR.utils import parser
from baseline.MCD4SR.utils.collate_fn import collate_seqbatch
from baseline.MCD4SR.utils.optimizer import build_optimizer, set_module_lr
from baseline.MCD4SR.utils.util import *

def test(
    split_batches=True,
    mixed_precision_type="fp16",
    amp=False,
    args=None
):
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )
    device = accelerator.device

    # ----------------- 加载数据 -----------------
    _, _, test_dataset = load_seq_datasets(
        args=args,
        inter_file=args.inter_path,
        num_items=args.num_items,
        max_seq_len=args.max_seq_len
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=collate_seqbatch,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_dataloader = accelerator.prepare(test_dataloader)

    model = EncoderDecoderRetrievalModel(
        text_dim = args.text_dim,
        visual_dim = args.visual_dim,
        attn_dim = args.attn_embed_dim,
        dropout=args.dropout,
        attn_heads=args.attn_heads,
        encoder_layers = args.attn_layers,
        max_pos = args.max_seq_len,
        top_k=args.top_k,
        args=args
    )

    # 没有训练，所以不需要 optimizer
    model, _ = accelerator.prepare(model, None)

    # ----------------- 初始化指标 -----------------
    metrics_accumulator = MetricsAccumulator()
    best_test_metrics = {
        "recall": {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0},
        "ndcg": {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0},
        "precision": {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0},
        "mrr": {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0},
    }

    # ----------------- 加载最佳模型 -----------------
    best_model_path = os.path.join(args.ckpts, "best_model.pt")
    if not os.path.exists(best_model_path):
        print("❌ No best model found for standalone testing")
        return

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # ----------------- 测试 -----------------
    test_metrics = evaluate_model(metrics_accumulator, model, test_dataloader, device, args, epoch=-1)

    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("📊 STANDALONE TEST RESULTS")
        print("="*60)
        print_test_metrics(-1, test_metrics)

        # 更新 best_test_metrics
        current_ndcg20 = test_metrics.get('ndcg@20', 0.0)
        if current_ndcg20 > best_test_metrics['ndcg'][20]:
            for metric in best_test_metrics:
                for k in [5, 10, 20, 50]:
                    metric_key = f"{metric}@{k}"
                    best_test_metrics[metric][k] = test_metrics.get(metric_key, best_test_metrics[metric][k])
            print_best_metrics(best_test_metrics, "Test", best_test_epoch=-1)

def train(
    split_batches=True,
    amp=False,
    mixed_precision_type="fp16",
    args=None
):  

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    # ----------------- 加载训练序列数据 -----------------
    train_dataset, eval_dataset, test_dataset = load_seq_datasets(
        args=args,
        inter_file=args.inter_path,
        num_items=args.num_items,
        max_seq_len=args.max_seq_len
    )

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        collate_fn=collate_seqbatch,
        num_workers=args.num_workers,
        pin_memory=True             
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.test_batch_size, 
        shuffle=False, 
        collate_fn=collate_seqbatch,
        num_workers=args.num_workers,
        pin_memory=True      
    )

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.test_batch_size, 
        shuffle=False, 
        collate_fn=collate_seqbatch,
        num_workers=args.num_workers,
        pin_memory=True      
    )

    # 使用accelerator准备数据加载器（分布式训练）
    train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader, test_dataloader
    )

    model = EncoderDecoderRetrievalModel(
        text_dim = args.text_dim,
        visual_dim = args.visual_dim,
        attn_dim = args.attn_embed_dim,
        dropout=args.dropout,
        attn_heads=args.attn_heads,
        encoder_layers = args.attn_layers,
        max_pos = args.max_seq_len,
        top_k=args.top_k,
        args=args
    )

    optimizer = build_optimizer(
        model,
        weight_decay=args.weight_decay,
        lr_id_encoder=args.lr_encoder,
        lr_modal_encoder=args.lr_encoder,
        lr_denoiser=args.lr_encoder,
        lr_item=1e-3,
        lr_modal_proj=1e-3,
        lr_default=1e-3
    )

    # 使用accelerator准备模型、优化器、调度器
    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    # 初始化评估指标累积器
    metrics_accumulator = MetricsAccumulator()
    num_params = sum(p.numel() for p in model.parameters())
    # 打印模型参数数量
    print(f"Device: {device}, Num Parameters: {num_params}")

    # 初始化全局最优字典
    best_eval_metrics = {
        "recall": {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0},
        "ndcg": {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0},
        "precision": {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0},
        "mrr": {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0},
    }
    best_test_metrics = {
        "recall": {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0},
        "ndcg": {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0},
        "precision": {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0},
        "mrr": {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0},
    }

    best_eval_epoch = 0
    best_test_epoch = 0

    # ---------------------------
    # 训练阶段
    # ---------------------------
    '''
    修改全局最优、早停机制
    '''
    start_time = time.time()

    if accelerator.is_main_process:
        print("\nInitial freeze done:")

    # 修正：早停机制相关变量
    no_improvement_count = 0  # 连续验证轮次未提升计数
    max_no_improvement = args.max_no_improvement   # 最大连续验证轮次未提升
    early_stop_triggered = False
    last_eval_epoch = -1      # 记录上一次验证的epoch

    print_lr_and_frozen_status(model, optimizer)
    for epoch in range(args.num_epochs):
        if early_stop_triggered:
            break  # 早停触发，退出训练循环
        # -------------------- 训练 --------------------
        for step, batch in enumerate(train_dataloader, start=1):
            model.train()
            optimizer.zero_grad()
            data = batch_to(batch, device)
            
            with accelerator.autocast():
                model_output = model.calculate_loss(batch=data, device=device, args=args, epoch=epoch)
                loss = model_output.loss["total_loss"]
            
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # -------------------- 损失打印部分 --------------------
            if accelerator.is_main_process:
                total_steps = len(train_dataloader)
                print_steps = [int(total_steps * i / 5) for i in range(1, 6)]
                
                if (step + 1) in print_steps or (step + 1) == total_steps:
                    elapsed = time.time() - start_time
                    postfix_str = format_postfix({
                        "total_loss": loss.item(),
                        "item_classify": args.w_icla * model_output.loss.get('icla', 0.0),
                        "similar_w": args.w_simw * model_output.loss.get('simw', 0.0),
                        "balance_w": args.w_balw * model_output.loss.get('balw', 0.0),
                        "modal_consistency": args.w_moct * model_output.loss.get('moct', 0.0),
                    })
                    print(f"\rTrain [{epoch+1}/{args.num_epochs}] "
                        f"Step [{step+1}/{total_steps}] "
                        f"Elapsed: {format_time(elapsed)} | {postfix_str}")
        # -------------------- 验证集评估 --------------------
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            eval_metrics = evaluate_model(metrics_accumulator, model, eval_dataloader, device, args, epoch)
            
            if accelerator.is_main_process:
                print_eval_metrics(epoch + 1, eval_metrics)

            # 保存验证集最佳模型（关键指标改为 ndcg@5 和 ndcg@10）
            current_ndcg20 = eval_metrics.get('ndcg@20', 0.0)
            
            # 检查是否提升（ndcg@5 和 ndcg@10 都提升才保存）
            improvement_found = False
            if current_ndcg20 > best_eval_metrics['ndcg'][20]:
                improvement_found = True
                no_improvement_count = 0  # 重置未提升计数
                best_eval_epoch = epoch + 1
                save_model(best_eval_epoch, model, optimizer, args.experiment_path)
                print(f"\n🎉 New Best Eval Model Saved at epoch {best_eval_epoch}")
                
                # 一次性更新所有指标
                for metric in best_eval_metrics:
                    for k in [5, 10, 20, 50]:
                        metric_key = f"{metric}@{k}"
                        best_eval_metrics[metric][k] = eval_metrics.get(metric_key, best_eval_metrics[metric][k])
                
                print_best_metrics(best_eval_metrics, "Eval", best_eval_epoch)

                # ✅ 当验证集有提升时立刻测试
                model.eval()
                test_metrics = evaluate_model(metrics_accumulator, model, test_dataloader, device, args, epoch)
                if accelerator.is_main_process:
                    print_test_metrics(epoch + 1, test_metrics)

                current_ndcg20 = test_metrics.get('ndcg@20', 0.0)
                if current_ndcg20 > best_test_metrics['ndcg'][20]:
                    best_test_epoch = epoch + 1
                    print(f"\n🎉 New Best Test Metrics Found at epoch {best_test_epoch}")
                    
                    # 一次性更新所有测试指标
                    for metric in best_test_metrics:
                        for k in [5, 10, 20, 50]:
                            metric_key = f"{metric}@{k}"
                            best_test_metrics[metric][k] = test_metrics.get(metric_key, best_test_metrics[metric][k])
                    
                    print_best_metrics(best_test_metrics, "Test", best_test_epoch)
            
            else:
                # 没有提升，计数增加
                no_improvement_count += 1
                if accelerator.is_main_process:
                    print(f"⚠️ No improvement for {no_improvement_count} consecutive validation evaluations")
                    print(f"Current: NDCG@20={current_ndcg20:.4f}")
                    print(f"Best:    NDCG@20={best_eval_metrics['ndcg'][20]:.4f}")

            # 检查早停条件
            if no_improvement_count >= max_no_improvement:
                early_stop_triggered = True
                if accelerator.is_main_process:
                    print(f"\n🚨 Early stopping triggered after {epoch + 1} epochs!")
                    print(f"No improvement in NDCG@5 & NDCG@20 for {max_no_improvement} consecutive validation evaluations")
                break  # 跳出训练循环

        # -------------------- 测试集评估 --------------------
        if not early_stop_triggered and (epoch + 1) % args.test_every == 0:
            model.eval()
            test_metrics = evaluate_model(metrics_accumulator, model, test_dataloader, device, args, epoch)
            
            if accelerator.is_main_process:
                print_test_metrics(epoch + 1, test_metrics)

            # 保存测试集最佳模型（关键指标同样改为 ndcg@5 & ndcg@10）
            current_ndcg20 = test_metrics.get('ndcg@20', 0.0)
            
            if current_ndcg20 > best_test_metrics['ndcg'][20]:
                best_test_epoch = epoch + 1
                print(f"\n🎉 New Best Test Metrics Found at epoch {best_test_epoch}")
                
                # 一次性更新所有测试指标
                for metric in best_test_metrics:
                    for k in [5, 10, 20, 50]:
                        metric_key = f"{metric}@{k}"
                        best_test_metrics[metric][k] = test_metrics.get(metric_key, best_test_metrics[metric][k])
                
                print_best_metrics(best_test_metrics, "Test", best_test_epoch)

    # --------------------------- 
    # 最终结果总结（不需要重新测试）
    # ---------------------------
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        if early_stop_triggered:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            print(f"No improvement for {no_improvement_count} consecutive validation cycles")
        else:
            print(f"Training finished normally at epoch {args.num_epochs}")
        
        # 直接打印最佳结果，不需要重新测试
        print(f"Best validation epoch: {best_eval_epoch}")
        print(f"Best test epoch: {best_test_epoch}")
        
        print("\n" + "="*60)
        print("BEST RESULTS SUMMARY")
        print("="*60)
        
        # 打印验证集最佳结果
        print("VALIDATION SET (Best):")
        print_best_metrics(best_eval_metrics, "Eval", best_eval_epoch)
        
        # 打印测试集最佳结果
        print("\nTEST SET (Best):")
        print_best_metrics(best_test_metrics, "Test", best_test_epoch)
        
        # 训练统计
        print("\n" + "="*60)
        print("TRAINING STATISTICS")
        print("="*60)
        print(f"Total epochs trained: {epoch + 1}")
        print(f"Early stopping: {'Yes' if early_stop_triggered else 'No'}")
        if early_stop_triggered:
            print(f"Stopped after {no_improvement_count} validation cycles without improvement")
        print(f"Final validation NDCG@20: {best_eval_metrics['ndcg'][20]:.4f}")
        print(f"Final test NDCG@20: {best_test_metrics['ndcg'][20]:.4f}")


if __name__ == "__main__":
    # config
    args = parser.get_args()
    parser.setup(args)
    if getattr(args, "test", False):
        test(args=args)
    else:
        train(args=args)
