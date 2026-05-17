import os
from typing import Dict
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from skrec.recommender.base import AbstractRecommender
from skrec.io import SequentialPairwiseIterator
from skrec.run_config import RunConfig
from skrec.utils.py import ModelConfig, EarlyStopping
from skrec.utils.torch import l2_loss, sp_mat_to_sp_tensor, get_initializer
from skrec.utils.common import normalize_adj_matrix
from skrec.io import RSDataset
from skrec.utils.common import make_sure_dirs
import itertools


class MiFuSRConfig(ModelConfig):
    def __init__(self,
                 lr=1e-3,
                 reg=1e-3,
                 n_dim=64,
                 n_layers=3,
                 norm_type="rw",
                 mi_max=0.8,
                 mi_min=0.2,
                 long_term=False,
                 n_seqs=5,
                 n_next=3,
                 batch_size=1024,
                 epochs=2000,
                 early_stop=200,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg
        self.n_dim: int = n_dim
        self.n_layers: int = n_layers
        self.norm_type: str = norm_type
        self.long_term: bool = long_term
        self.mi_max: float = mi_max
        self.mi_min: float = mi_min
        self.n_seqs: int = n_seqs
        self.n_next: int = n_next
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop


class Trans(nn.Module):
    def __init__(self):
        super(Trans, self).__init__()

    def forward(self, user_embeds, head_embeds, tail_embeds):
        r_trans_embeds = user_embeds + head_embeds
        r_trans_embeds = torch.unsqueeze(r_trans_embeds, dim=1)
        hat_y = - torch.norm(r_trans_embeds - tail_embeds, p=None, dim=-1)
        return hat_y


def log_loss(yij):
    """ bpr loss
    """
    return -F.logsigmoid(yij)


class GNNLayer(nn.Module):
    def __init__(self, n_layers, adj_mat, use_residual=False, mean_features=False):
        super(GNNLayer, self).__init__()
        self.adj_mat = adj_mat.coalesce()
        self.n_layers = n_layers
        self.use_residual: bool = use_residual
        self.mean_features: bool = mean_features
        indices = self.adj_mat.indices()
        self._rows = indices[0]
        self._cols = indices[1]
        self._data = self.adj_mat.values().view([-1, 1])

    def forward(self, x):
        ego_embeds = x
        all_embeddings = [ego_embeds]
        for _ in range(self.n_layers):
            side_embeds = torch.zeros_like(ego_embeds)
            side_embeds.index_add_(0, self._rows, ego_embeds[self._cols] * self._data)
            if self.use_residual:
                ego_embeds = side_embeds + ego_embeds
            else:
                ego_embeds = side_embeds
            all_embeddings += [ego_embeds]

        if self.mean_features:
            out_embeds = torch.stack(all_embeddings, dim=1).mean(dim=1, keepdim=False)
        else:
            out_embeds = ego_embeds

        return out_embeds, all_embeddings


class MINE(nn.Module):
    def __init__(self, input_dim, hidden_size=10):
        super(MINE, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.BatchNorm1d(1))

    def forward(self, x, y):
        batch_size = x.size(0)
        tiled_x = torch.cat([x, x], dim=0)
        idx = torch.randperm(batch_size)

        shuffled_y = y[idx]
        concat_y = torch.cat([y, shuffled_y], dim=0)
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers(inputs)

        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]

        loss = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        return loss


class _Model(nn.Module):
    def __init__(self, dataset: RSDataset, seq_graph, config: MiFuSRConfig, img_feat, txt_feat):
        super(_Model, self).__init__()
        n_users = dataset.num_users
        n_items = dataset.num_items
        self.config = config
        self.n_dim = config.n_dim
        self.long_term = config.long_term
        n_layers = config.n_layers

        self._trans = Trans()

        # init embeddings
        self.img_feat = img_feat
        self.txt_feat = txt_feat
        self.img_linear = nn.Linear(dataset.img_dim, self.n_dim)
        self.txt_linear = nn.Linear(dataset.txt_dim, self.n_dim)

        self.user_embeds = nn.Embedding(n_users, self.n_dim)
        self.item_biases = nn.Embedding(n_items, 1)

        self.img_gnn = GNNLayer(n_layers, seq_graph, use_residual=True, mean_features=False)
        self.txt_gnn = GNNLayer(n_layers, seq_graph, use_residual=True, mean_features=False)

        # for test
        self.final_img_embeds = None
        self.final_txt_embeds = None
        self.reset_parameters()

    def reset_parameters(self):
        init = get_initializer("normal")
        init(self.user_embeds.weight)
        self.img_linear.reset_parameters()
        self.txt_linear.reset_parameters()
        nn.init.zeros_(self.item_biases.weight)

    def item_gcn_forward(self):
        img_embeds = self.img_linear(self.img_feat)
        txt_embeds = self.txt_linear(self.txt_feat)

        img_embeds, all_img_embeds = self.img_gnn(img_embeds)
        txt_embeds, all_txt_embeds = self.txt_gnn(txt_embeds)

        return img_embeds, txt_embeds, all_img_embeds, all_txt_embeds

    def _mean_history(self, item_embeddings, head_items):
        # fuse to get short-term embeddings
        pad_id = item_embeddings.shape[0]
        item_embeddings = F.pad(item_embeddings, (0, 0, 0, 1), value=0)
        item_seq_embeds = F.embedding(head_items, item_embeddings)  # (b,l,d)
        mask = torch.not_equal(head_items, pad_id).float()  # (b,l)
        his_embeds = item_seq_embeds.sum(dim=1) / mask.sum(dim=1, keepdim=True)  # (b,d)/(b,1)

        return his_embeds

    def _forward_head_embed(self, item_embeddings, head_items):
        # embed item sequence
        if head_items.dim() > 1:
            last_embeds = F.embedding(head_items[:, -1], item_embeddings)  # b*d
        else:
            last_embeds = F.embedding(head_items, item_embeddings)

        if self.long_term is True and head_items.dim() > 1:
            his_embeds = self._mean_history(item_embeddings, head_items)
            head_embeds = last_embeds + his_embeds
        else:
            head_embeds = last_embeds

        return head_embeds

    def forward(self, users, head_items, tail_items):
        # GCN
        img_embeds, txt_embeds, all_img_embeds, all_txt_embeds = self.item_gcn_forward()

        user_embeds = self.user_embeds(users)
        item_bias = self.item_biases(tail_items).squeeze()

        img_head_embeds = self._forward_head_embed(img_embeds, head_items)
        txt_head_embeds = self._forward_head_embed(txt_embeds, head_items)
        img_tail_embeds = F.embedding(tail_items, img_embeds)
        txt_tail_embeds = F.embedding(tail_items, txt_embeds)

        reg_params = [user_embeds, img_head_embeds, img_tail_embeds,
                      txt_head_embeds, txt_tail_embeds, item_bias]

        # trans
        img_ratings = self._trans(user_embeds, img_head_embeds, img_tail_embeds)
        txt_ratings = self._trans(user_embeds, txt_head_embeds, txt_tail_embeds)

        train_ratings = img_ratings + txt_ratings + item_bias

        return train_ratings, reg_params, all_img_embeds, all_txt_embeds, user_embeds

    def predict(self, users, head_items):
        if self.final_img_embeds is None or self.final_txt_embeds is None:
            raise ValueError("Please first switch to 'eval' mode.")

        user_embeds = self.user_embeds(users)

        img_head_embeds = self._forward_head_embed(self.final_img_embeds, head_items)
        txt_head_embeds = self._forward_head_embed(self.final_txt_embeds, head_items)

        img_ratings = self._trans(user_embeds, img_head_embeds, self.final_img_embeds)
        txt_ratings = self._trans(user_embeds, txt_head_embeds, self.final_txt_embeds)
        eval_ratings = img_ratings + txt_ratings + torch.squeeze(self.item_biases.weight)
        return eval_ratings

    def eval(self):
        super(_Model, self).eval()
        self.final_img_embeds, self.final_txt_embeds, _, _ = self.item_gcn_forward()


class MiFuSR(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = MiFuSRConfig(**model_config)
        super().__init__(run_config, self.config)

        self.users_num, self.items_num = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._init_constant()
        img_feat = torch.from_numpy(self.dataset.img_features).type(torch.FloatTensor).to(self.device)
        txt_feat = torch.from_numpy(self.dataset.txt_features).type(torch.FloatTensor).to(self.device)
        self.model: _Model = _Model(self.dataset, self.seq_adj_mat, self.config, img_feat, txt_feat).to(self.device)
        self.mine_max: MINE = MINE(input_dim=self.config.n_dim*2, hidden_size=self.config.n_dim).to(self.device)
        self.mine_img_min: MINE = MINE(input_dim=self.config.n_dim*2, hidden_size=self.config.n_dim).to(self.device)
        self.mine_txt_min: MINE = MINE(input_dim=self.config.n_dim*2, hidden_size=self.config.n_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        mine_param = itertools.chain(self.mine_max.parameters(),
                                     self.mine_img_min.parameters(),
                                     self.mine_txt_min.parameters())
        self.mine_optm = torch.optim.Adam(mine_param, lr=self.config.lr)

    def _init_constant(self):
        dir_name = self.dataset.cache_dir
        dir_name = os.path.join(dir_name, self.__class__.__name__.lower())
        make_sure_dirs(dir_name)

        self.user_pos_train = self.dataset.train_data.to_user_dict_by_time()

        self.test_item_seqs = self.dataset.train_data.to_truncated_seq_dict(self.config.n_seqs,
                                                                            pad_value=self.items_num,
                                                                            padding='pre', truncating='pre')
        if self.config.norm_type == 'rw':
            norm_method = "left"
        else:
            norm_method = "symmetric"
        seq_g_name = os.path.join(dir_name, f"seq_graph_{self.config.norm_type}.npz")

        if os.path.exists(seq_g_name):
            seq_adj_mat = sp.load_npz(seq_g_name)
        else:
            seq_adj_mat = self._build_item_graph()
            seq_adj_mat = normalize_adj_matrix(seq_adj_mat, norm_method)
            sp.save_npz(seq_g_name, seq_adj_mat)

        self.seq_adj_mat = sp_mat_to_sp_tensor(seq_adj_mat).to(self.device)

    def _build_item_graph(self):
        th_rs_dict = defaultdict(list)
        for user, pos_items in self.user_pos_train.items():
            for h, t in zip(pos_items[:-1], pos_items[1:]):
                th_rs_dict[(t, h)].append(user)

        th_len_list = [[t, h, len(rs)] for (t, h), rs in th_rs_dict.items()]
        tail_list, head_list, edge_num = list(zip(*th_len_list))

        adj_mat = sp.csr_matrix((edge_num, (tail_list, head_list)), dtype=np.float32,
                                shape=(self.items_num, self.items_num))  # in matrix

        return adj_mat

    def fit(self):
        data_iter = SequentialPairwiseIterator(self.dataset.train_data,
                                               num_previous=self.config.n_seqs, num_next=self.config.n_next,
                                               pad=self.items_num, batch_size=self.config.batch_size,
                                               shuffle=True, drop_last=False)

        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            self.model.train()
            if self.config.mi_min + self.config.mi_max > 0:
                for bat_users, bat_item_seq, bat_pos_next, bat_neg_next in data_iter:
                    bat_users = torch.from_numpy(bat_users).long().to(self.device)
                    bat_item_seq = torch.from_numpy(bat_item_seq).long().to(self.device)
                    bat_pos_next = torch.from_numpy(bat_pos_next).long().to(self.device)
                    bat_neg_next = torch.from_numpy(bat_neg_next).long().to(self.device)

                    bat_pos_next = bat_pos_next.reshape(bat_pos_next.shape[0], -1)
                    bat_neg_next = bat_neg_next.reshape(bat_neg_next.shape[0], -1)

                    bat_tail_items = torch.cat([bat_pos_next, bat_neg_next], dim=1)
                    train_ratings, params, img_embeds, txt_embeds, user_embeds = self.model(bat_users, bat_item_seq, bat_tail_items)
                    max_loss, min_loss = self.calculate_mi(img_embeds, txt_embeds)
                    mi_loss = -(max_loss + min_loss)
                    self.mine_optm.zero_grad()
                    mi_loss.backward()
                    self.mine_optm.step()

            all_max_mi = []
            all_min_mi = []
            for bat_users, bat_item_seq, bat_pos_next, bat_neg_next in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_item_seq = torch.from_numpy(bat_item_seq).long().to(self.device)
                bat_pos_next = torch.from_numpy(bat_pos_next).long().to(self.device)
                bat_neg_next = torch.from_numpy(bat_neg_next).long().to(self.device)

                bat_pos_next = bat_pos_next.reshape(bat_pos_next.shape[0], -1)
                bat_neg_next = bat_neg_next.reshape(bat_neg_next.shape[0], -1)

                bat_tail_items = torch.cat([bat_pos_next, bat_neg_next], dim=1)
                train_ratings, params, img_embeds, txt_embeds, user_embeds = self.model(bat_users, bat_item_seq, bat_tail_items)

                max_loss, min_loss = self.calculate_mi(img_embeds, txt_embeds)
                all_max_mi.append(float(max_loss))
                all_min_mi.append(float(min_loss))

                yui, yuj = train_ratings.split(self.config.n_next, dim=1)
                bpr_loss = log_loss(yui - yuj).sum()
                reg_loss = l2_loss(*params)

                mi_loss = - self.config.mi_max * max_loss + self.config.mi_min * min_loss
                final_loss = bpr_loss + self.config.reg * reg_loss + mi_loss
                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()

            cur_result = self.evaluate()
            self.logger.info(f"epoch {epoch}:".ljust(12) + f"\t{cur_result.values_str}")
            if early_stopping(cur_result):
                self.logger.info("early stop")
                break

        self.logger.info("best:".ljust(12) + f"\t{early_stopping.best_result.values_str}")
        return early_stopping.best_result

    def calculate_mi(self, img_embeds, txt_embeds) -> (torch.Tensor, torch.Tensor):
        max_loss = 0.0
        if self.config.mi_max > 0:
            for img_embed, txt_embed in zip(img_embeds, txt_embeds):
                max_loss += self.mine_max(img_embed, txt_embed)

        min_loss = 0.0
        if self.config.mi_min > 0:
            min_loss = self.mine_img_min(img_embeds[0], img_embeds[-1])
            min_loss += self.mine_txt_min(txt_embeds[0], txt_embeds[-1])

        return max_loss, min_loss

    def evaluate(self, test_users=None):
        self.model.eval()
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        last_items = [self.test_item_seqs[u] for u in users]
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        last_items = torch.from_numpy(np.asarray(last_items)).long().to(self.device)
        bat_ratings = self.model.predict(users, last_items)
        return bat_ratings.cpu().detach().numpy()
