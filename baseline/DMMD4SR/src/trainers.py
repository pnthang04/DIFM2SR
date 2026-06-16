
import json
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from baseline.DMMD4SR.src.utils import recall_at_k, ndcg_k, cal_mrr, get_metric



class Trainer:
    def __init__(self, model, train_dataloader,cluster_dataloader, eval_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model = model

        self.batch_size = self.args.batch_size
        self.sim=self.args.sim



        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader


        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader


        self.optim = Adam(self.model.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))



    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        HIT_20, NDCG_20, MRR = get_metric(pred_list, 20)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.4f}".format(HIT_1),
            "NDCG@1": "{:.4f}".format(NDCG_1),
            "HIT@5": "{:.4f}".format(HIT_5),
            "NDCG@5": "{:.4f}".format(NDCG_5),
            "HIT@10": "{:.4f}".format(HIT_10),
            "NDCG@10": "{:.4f}".format(NDCG_10),
            "MRR": "{:.4f}".format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        topks = [5, 10, 20, 50]
        recall = {k: recall_at_k(answers, pred_list, k) for k in topks}
        ndcg = {k: ndcg_k(answers, pred_list, k) for k in topks}
        post_fix = {
            "recall@5": round(recall[5], 4),
            "recall@10": round(recall[10], 4),
            "recall@20": round(recall[20], 4),
            "recall@50": round(recall[50], 4),
            "ndcg@5": round(ndcg[5], 4),
            "ndcg@10": round(ndcg[10], 4),
            "ndcg@20": round(ndcg[20], 4),
            "ndcg@50": round(ndcg[50], 4),
        }
        result_info = json.dumps(post_fix, indent=2)
        print(result_info)
        with open(self.args.log_file, "a") as f:
            f.write(result_info + "\n")
        scores = [
            recall[5], recall[10], recall[20], recall[50],
            ndcg[5], ndcg[10], ndcg[20], ndcg[50],
        ]
        return scores, result_info



    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    # False Negative Mask
    def mask_correlated_samples_(self, label):
        label=label.view(1,-1)
        label=label.expand((2,label.shape[-1])).reshape(1,-1)
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, label.t())
        return mask==0

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot',intent_id=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.t()) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if self.args.f_neg:
            mask = self.mask_correlated_samples_(intent_id)
            negative_samples = sim
            negative_samples[mask==0]=float("-inf")
        else:
            mask = self.mask_correlated_samples(batch_size)
            negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def cicl_loss(self,coarse_intents,target_item):
        coarse_intent_1,coarse_intent_2=coarse_intents[0],coarse_intents[1]
        sem_nce_logits, sem_nce_labels = self.info_nce(coarse_intent_1[:, -1, :], coarse_intent_2[:, -1, :],
                                                       self.args.temperature, coarse_intent_1.shape[0], self.sim,
                                                       target_item[:, -1])
        cicl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cicl_loss




class ICLRecTrainer(Trainer):
    def __init__(self, model, train_dataloader,cluster_dataloader,eval_dataloader, test_dataloader, args):
        super(ICLRecTrainer, self).__init__(
            model, train_dataloader, cluster_dataloader,eval_dataloader, test_dataloader, args
        )

    def iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True):
        str_code = "train" if train else "test"
        if train:
 

            # ------ model training -----#
            print("Performing Rec model Training:")
            self.model.train()
            rec_avg_loss = 0.0
            joint_avg_loss = 0.0
            icl_losses=0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch) in rec_cf_data_iter:
                """             
                rec_batch shape: key_name x batch_size x feature_dim
                """
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, subsequence_1, target_pos_1, _, _ = rec_batch

                # ---------- prediction task -----------------------#
                intent_output, t_emb_loss, v_emb_loss, diff_loss = self.model(subsequence_1, is_train=True)
                logits = self.predict_full(intent_output[:, -1, :])  #  [Bx|I|]
                rec_loss = nn.CrossEntropyLoss()(logits, target_pos_1[:, -1])

                # ---------- intent representation learning task ---------------#

                # ---------- multi-task learning --------------------#
                joint_loss =self.args.rec_weight * rec_loss + self.args.icl_loss * t_emb_loss + self.args.icl_loss * v_emb_loss + self.args.diff_loss * diff_loss  # 推荐的损失 + 意图损失（为 0 ） + 扩散模型损失

                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()
                icl_losses = 0
                joint_avg_loss += joint_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_cf_data_iter)),
                "icl_avg_loss": "{:.4f}".format(icl_losses/ len(rec_cf_data_iter)),
                "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(rec_cf_data_iter)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

        else:
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, answers = batch
                    recommend_output, _, _, _ = self.model(input_ids) # [BxLxH]
                    recommend_output = recommend_output[:, -1, :] # [BxH]
                    # recommendation results
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    top_k = min(50, rating_pred.shape[1])
                    ind = np.argpartition(rating_pred, -top_k)[:, -top_k:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output, _, _, _ = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
