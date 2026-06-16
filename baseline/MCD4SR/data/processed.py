import random
from collections import defaultdict
from typing import NamedTuple
import torch
from torch.utils.data import Dataset

class SeqBatch(NamedTuple):
    user_ids: torch.Tensor       # [B]
    hist: torch.Tensor           # [B, max_seq_len]
    target: torch.Tensor         # [B]
    seq_mask: torch.Tensor       # [B, max_seq_len]
    target_ids: torch.Tensor = None       # [B, max_seq_len]


class SeqData(Dataset):
    def __init__(self, args, samples, num_items=None, is_train=True):
        self.args = args
        self.samples = samples
        self.is_train = is_train
        self.num_items = num_items
        self.max_seq_len = args.max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user, hist, target, _ = self.samples[idx]

        # === 序列 padding ===
        hist_len = len(hist)
        pad_id = self.num_items  # 使用 num_items 作为 padding id
        hist_padded = hist + [pad_id] * (self.max_seq_len - hist_len)
        mask = [1] * hist_len + [0] * (self.max_seq_len - hist_len)

        # === 构造 target_ids ===
        # 例：hist=[1,2,3,4], target=5 → target_ids=[2,3,4,5]
        if hist_len > 0:
            target_seq = hist[1:] + [target]
        else:
            target_seq = [target]
        target_padded = target_seq + [pad_id] * (self.max_seq_len - len(target_seq))
        target_padded = target_padded[:self.max_seq_len]

        return SeqBatch(
            user_ids=torch.tensor(user, dtype=torch.long),
            hist=torch.tensor(hist_padded, dtype=torch.long),
            target=torch.tensor(target, dtype=torch.long),
            seq_mask=torch.tensor(mask, dtype=torch.bool),
            target_ids=torch.tensor(target_padded, dtype=torch.long)
        )


def load_seq_datasets(args, inter_file, num_items,
                      max_seq_len=20, min_hist_len=1, seed=42):
    random.seed(seed)

    # === 读取用户交互 ===
    user2items = defaultdict(list)
    with open(inter_file, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            if not line.strip():
                continue
            user, item, rating, ts = line.strip().split()
            user, item, ts = int(user), int(item), float(ts)
            user2items[user].append((ts, item))

    # === 按时间排序 ===
    for user in user2items:
        user2items[user].sort(key=lambda x: x[0])
        user2items[user] = [item for ts, item in user2items[user]]

    # === 划分 train/valid/test ===
    train_samples, valid_samples, test_samples = [], [], []

    for user, items in user2items.items():
        if len(items) < 3:
            continue

        # --- 测试集 ---
        target_test = items[-1]
        hist_test = items[:-1][-max_seq_len:]
        test_samples.append((user, hist_test, target_test, max_seq_len))

        # --- 验证集 ---
        target_valid = items[-2]
        hist_valid = items[:-2][-max_seq_len:]
        valid_samples.append((user, hist_valid, target_valid, max_seq_len))

        # --- 训练集 ---
        for end_idx in range(1, len(items) - 2):
            target = items[end_idx]
            hist = items[:end_idx][-max_seq_len:]
            train_samples.append((user, hist, target, max_seq_len))

    print(f"数据统计: 训练集 {len(train_samples)} 样本, 验证集 {len(valid_samples)} 样本, 测试集 {len(test_samples)} 样本")

    return (
        SeqData(args, train_samples, num_items, True),
        SeqData(args, valid_samples, num_items, False),
        SeqData(args, test_samples, num_items, False)
    )
