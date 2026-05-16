import torch
from torch.nn import functional as F
from data.utils import batch_to
import os
from modules.transformer.attention_padding_mask import MultiHeadAttention
from torch import nn
import numpy as np
import math
import random

def print_lr_and_frozen_status(model, optimizer):
    # 建立参数到学习率的映射
    param_to_lr = {}
    for i, param_group in enumerate(optimizer.param_groups):
        for p in param_group['params']:
            param_to_lr[id(p)] = param_group['lr']

    # 遍历模型参数，打印冻结状态和对应学习率
    for name, param in model.named_parameters():
        lr = param_to_lr.get(id(param), None)
        print(f"{name}: requires_grad = {param.requires_grad}, lr = {lr}")

def format_postfix(loss_dict=None):
    """格式化 tqdm 后缀字符串"""
    parts = []
    if loss_dict:
        parts += [f"{k}={v:.4f}" for k, v in loss_dict.items()]
    return " | ".join(parts)

def evaluate_model(metrics_accumulator, model, dataloader, device, args, epoch):
    """通用评估函数"""
    metrics_accumulator.reset()
    for batch in dataloader:
        data = batch_to(batch, device)
        with torch.no_grad():
            model_output = model.calculate_loss(batch=data, device=device, args=args, epoch=epoch)
            metrics_accumulator.accumulate(
                actual=data.target.unsqueeze(1),
                top_k=model_output.topk_idx
            )
    return metrics_accumulator.get_results(ks=[5, 10, 20, 50])

def print_eval_metrics(epoch, metrics):
    """打印验证集指标"""
    print("\n" + "=" * 10 + f" Eval Metrics at Epoch {epoch} " + "=" * 10)
    print_metrics_table(metrics)

def print_test_metrics(epoch, metrics):
    """打印测试集指标"""
    print("\n" + "=" * 10 + f" Test Metrics at Epoch {epoch} " + "=" * 10)
    print_metrics_table(metrics)


def print_metrics_table(metrics):
    """通用指标表格打印"""
    # 打印 @5/@10/@20/@50 指标
    print(f"{'Metric':<15} {'@5':<8} {'@10':<8} {'@20':<8} {'@50':<8}")
    print("-" * 45)
    for metric in ["recall", "ndcg", "precision", "mrr"]:
        line = f"{metric:<15}"
        for k in [5, 10, 20, 50]:
            line += f"{metrics.get(f'{metric}@{k}', 0.0):.4f} "
        print(line)
    
    # 打印曲线，每5个一行
    curve = metrics.get("curve", {})
    print("\nMetrics curve (1-50), 5 per line:")
    for metric in ["recall", "ndcg", "precision", "mrr"]:
        vals = curve.get(metric, [])
        if vals:
            print(f"{metric}:")
            for i in range(0, len(vals), 5):
                chunk = vals[i:i+5]
                print("  ", " ".join(f"{v:.4f}" for v in chunk))
    
def print_best_metrics(best_metrics, dataset_type, best_epoch):
    print("\n" + "=" * 10 + f" Best {dataset_type} Metrics at Epoch {best_epoch} " + "=" * 10)
    print(f"{'Metric':<15} {'@5':<8} {'@10':<8} {'@20':<8} {'@50':<8}")
    print("-" * 45)
    for metric in ["recall", "ndcg", "precision", "mrr"]:
        line = f"{metric:<15}"
        for k in [5, 10, 20, 50]:
            line += f"{best_metrics[metric][k]:.4f} "  # 直接访问字典值
        print(line)

def save_model(best_eval_epoch, model, optimizer, save_path):
    """模型保存函数"""
    state = {
        "epoch": best_eval_epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state, os.path.join(save_path, "best_model.pt"))

def freeze_layer(
    model,
    freeze_denoiser=False,
    freeze_modal_encoder=False,      
    freeze_proj=False,        
    freeze_item_embedding=False,
    freeze_item_id_encoder=False
):
    for name, param in model.named_parameters():
        if "item_id_encoder" in name:
            param.requires_grad = not freeze_item_id_encoder
        elif "denoiser" in name:
            param.requires_grad = not freeze_denoiser
        elif "text_encoder" in name or "visual_encoder" in name:
            param.requires_grad = not freeze_modal_encoder
        elif "text_proj_in" in name or "visual_proj_in" in name:
            param.requires_grad = not freeze_proj
        elif "item_embedding.weight" in name:
            param.requires_grad = not freeze_item_embedding


def format_time(seconds: float) -> str:
    """
    将秒数转换为 HH:MM:SS 格式字符串
    """
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
