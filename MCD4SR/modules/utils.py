def gather_indexes(output, gather_index):
    """
    output: [B, L, D]
    gather_index: [B] 表示每个样本需要取的位置 (比如最后一个非padding位置)
    """
    gather_index = gather_index.view(-1, 1, 1).expand(-1, 1, output.shape[-1])
    output_tensor = output.gather(dim=1, index=gather_index)
    return output_tensor.squeeze(1)  # [B, D]