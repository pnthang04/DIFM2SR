import numpy as np
import torch

def recall_(pos_index, pos_len):
    """Recall@k"""
    B, K = pos_index.shape
    hit_cnt = np.cumsum(pos_index, axis=1)
    valid_mask = (pos_len > 0)
    if valid_mask.sum() == 0:
        return np.zeros(K, dtype=float)
    hit_valid = hit_cnt[valid_mask]
    denom = pos_len[valid_mask].reshape(-1, 1)
    return (hit_valid / denom).mean(axis=0)

def precision_(pos_index, pos_len):
    """Precision@k"""
    B, K = pos_index.shape
    prec = np.cumsum(pos_index, axis=1) / np.arange(1, K + 1)
    valid_mask = (pos_len > 0)
    if valid_mask.sum() == 0:
        return np.zeros(K, dtype=float)
    return prec[valid_mask].mean(axis=0)

def ndcg_(pos_index, pos_len, eps=1e-8):
    """NDCG@k"""
    B, K = pos_index.shape
    ranks = np.arange(1, K + 1)
    gains = pos_index / np.log2(ranks + 1)
    dcg = np.cumsum(gains, axis=1)

    valid_mask = (pos_len > 0)
    if valid_mask.sum() == 0:
        return np.zeros(K, dtype=float)

    idcg_list = []
    for u in np.where(valid_mask)[0]:
        l = min(int(pos_len[u]), K)
        if l == 0:
            idcg_row = np.zeros(K, dtype=float)
        else:
            ideal_gains = 1.0 / np.log2(np.arange(1, l + 1) + 1)
            cumsum = np.cumsum(ideal_gains)
            if l < K:
                idcg_row = np.concatenate([cumsum, np.full(K - l, cumsum[-1])])
            else:
                idcg_row = cumsum
        idcg_list.append(idcg_row)
    idcg = np.stack(idcg_list, axis=0)

    dcg_valid = dcg[valid_mask]
    ndcg = dcg_valid / (idcg + eps)
    return ndcg.mean(axis=0)

def mrr_(pos_index, pos_len):
    """MRR@k"""
    B, K = pos_index.shape
    valid_mask = (pos_len > 0)
    if valid_mask.sum() == 0:
        return np.zeros(K, dtype=float)

    mrr_values = np.zeros((B, K), dtype=float)
    for i in range(B):
        if not valid_mask[i]:
            continue
        hit_positions = np.where(pos_index[i] == 1)[0]
        if len(hit_positions) > 0:
            first_hit = hit_positions[0]
            mrr_values[i, first_hit:] = 1.0 / (first_hit + 1)
    return mrr_values[valid_mask].mean(axis=0)


class MetricsAccumulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_pos_index = []
        self.all_pos_len = []

    def accumulate(self, actual, top_k):
        """
        actual: [B, L] ground-truth item ids (padding = -1)
        top_k: [B, K] recommended item ids
        """
        if torch.is_tensor(actual):
            actual = actual.detach().cpu().numpy()
        else:
            actual = np.array(actual)

        if torch.is_tensor(top_k):
            top_k = top_k.detach().cpu().numpy()
        else:
            top_k = np.array(top_k)

        B, K = top_k.shape

        pos_index = np.zeros((B, K), dtype=int)
        pos_len = np.zeros(B, dtype=int)

        for i in range(B):
            all_actual = set(actual[i].tolist())
            all_actual = {a for a in all_actual if a != -1}
            pos_len[i] = len(all_actual)
            if pos_len[i] > 0:
                pos_index[i] = np.isin(top_k[i], list(all_actual)).astype(int)
            else:
                pos_index[i] = 0

        self.all_pos_index.append(pos_index)
        self.all_pos_len.append(pos_len)

    def get_results(self, ks=[5, 10, 20, 50]):
        pos_index = np.concatenate(self.all_pos_index, axis=0)
        pos_len = np.concatenate(self.all_pos_len, axis=0)

        K = pos_index.shape[1]
        max_k = min(50, K)

        # 一次性计算指标曲线
        recall_curve = recall_(pos_index, pos_len)
        precision_curve = precision_(pos_index, pos_len)
        ndcg_curve = ndcg_(pos_index, pos_len)
        mrr_curve = mrr_(pos_index, pos_len)

        results = {}

        for k in ks:
            if k <= K:
                results[f"recall@{k}"] = float(recall_curve[k - 1])
                results[f"precision@{k}"] = float(precision_curve[k - 1])
                results[f"ndcg@{k}"] = float(ndcg_curve[k - 1])
                results[f"mrr@{k}"] = float(mrr_curve[k - 1])

        results["curve"] = {
            "recall": recall_curve[:max_k].tolist(),
            "precision": precision_curve[:max_k].tolist(),
            "ndcg": ndcg_curve[:max_k].tolist(),
            "mrr": mrr_curve[:max_k].tolist(),
        }

        return results