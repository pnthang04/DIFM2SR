from data.processed import SeqBatch

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

def batch_to(batch, device):
    return SeqBatch(**{
        k: v.to(device) if v is not None else None
        for k, v in batch._asdict().items()
    })


def next_batch(dataloader_iter, device):
    batch = next(dataloader_iter)
    return batch_to(batch, device)

