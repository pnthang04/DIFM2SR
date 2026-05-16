from torch.optim import AdamW

def build_optimizer(model, weight_decay=0.01, 
                   lr_id_encoder=1e-4,        # 编码器学习率
                   lr_modal_encoder=1e-4,        # 编码器学习率
                   lr_denoiser=1e-4,      # 去噪器学习率
                   lr_item=1e-3,          # item相关学习率
                   lr_modal_proj=1e-5,     # 模态投影层学习率
                   lr_default=1e-3):      # 默认参数学习率
    param_groups = []
    assigned_params = set()
    
    # ---------------- 模态编码器分组（text_encoder + visual_encoder） ----------------
    modal_encoder_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and ("text_encoder" in name or "visual_encoder" in name):
            modal_encoder_params.append(param)
            assigned_params.add(param)
    if modal_encoder_params:
        param_groups.append({
            "params": modal_encoder_params,
            "lr": lr_modal_encoder,
            "weight_decay": weight_decay,
            "module_key": "modal_encoders",
        })
    
    # ---------------- 模态投影层分组（text_proj_in + visual_proj_in） ----------------
    modal_proj_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and ("text_proj_in" in name or "visual_proj_in" in name):
            modal_proj_params.append(param)
            assigned_params.add(param)
    if modal_proj_params:
        param_groups.append({
            "params": modal_proj_params,
            "lr": lr_modal_proj,
            "weight_decay": weight_decay,
            "module_key": "modal_projectors",
        })
    
    # ---------------- item_id_encoder 专属分组 ----------------
    item_encoder_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and "item_id_encoder" in name:
            item_encoder_params.append(param)
            assigned_params.add(param)
    if item_encoder_params:
        param_groups.append({
            "params": item_encoder_params,
            "lr": lr_id_encoder,
            "weight_decay": weight_decay,
            "module_key": "item_id_encoder",
    })
    
    # ---------------- 去噪器分组 ----------------
    denoiser_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and "denoiser" in name:
            denoiser_params.append(param)
            assigned_params.add(param)
    if denoiser_params:
        param_groups.append({
            "params": denoiser_params,
            "lr": lr_denoiser,
            "weight_decay": weight_decay,
            "module_key": "denoisers",
        })
    
    # ---------------- item_embedding 专属分组 ----------------
    item_emb_param = None
    for name, param in model.named_parameters():
        if name == "item_embedding.weight":
            item_emb_param = param
            assigned_params.add(param)
            break
    if item_emb_param is not None:
        param_groups.append({
            "params": [item_emb_param],
            "lr": lr_item,
            "weight_decay": weight_decay,
            "module_key": "item_embedding.weight",
        })
    
    # ---------------- 默认分组（其余所有参数） ----------------
    default_params = [p for p in model.parameters() if p.requires_grad and p not in assigned_params]
    if default_params:
        param_groups.append({
            "params": default_params,
            "lr": lr_default,
            "weight_decay": weight_decay,
            "module_key": "default",
        })
    
    optimizer = AdamW(param_groups)
    return optimizer



def set_module_lr(optimizer, module_key: str, new_lr: float, verbose=True):
    """
    动态修改某个模块的学习率（立即生效）
    """
    updated = False
    for group in optimizer.param_groups:
        if group.get("module_key") == module_key:
            old_lr = group["lr"]
            group["lr"] = new_lr
            updated = True
            if verbose:
                print(f"[LR-UPDATE] {module_key}: {old_lr:.6f} → {new_lr:.6f}")
            break

    if not updated and verbose:
        print(f"[WARN] No param_group found for module '{module_key}'.")
