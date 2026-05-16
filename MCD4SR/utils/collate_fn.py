import torch
from data.processed import SeqBatch

def collate_seqbatch(batch):
    """
    batch: list of SeqBatch
    返回一个 SeqBatch，其中每个字段都堆叠成 batch_size 的 tensor
    """
    user_ids = torch.stack([b.user_ids for b in batch])
    hist = torch.stack([b.hist for b in batch])
    target = torch.stack([b.target for b in batch])
    seq_mask = torch.stack([b.seq_mask for b in batch])
    target_ids = torch.stack([b.target_ids for b in batch])

    return SeqBatch(
        user_ids=user_ids,
        hist=hist,
        target=target,
        seq_mask=seq_mask,
        target_ids = target_ids
    )
