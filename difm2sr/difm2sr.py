import numpy as np
import copy
import torch
import math
import os
from torch import nn
import torch.nn.functional as F
import torch.fft
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
from sklearn.cluster import KMeans, MiniBatchKMeans
from diffusion_denoiser import GraphConditionedDiffusionDenoiser


def cfg_get(config, key, default=None):
    return config[key] if key in config else default


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs 


class IntentAwareWeightMapper(torch.nn.Module):
    def __init__(self, hidden_size, temperature):
        super(IntentAwareWeightMapper, self).__init__()
        self.l1 = torch.nn.Linear(3 * hidden_size, hidden_size // 8)
        self.l2 = torch.nn.Linear(hidden_size // 8, 3)
        self.temperature = temperature

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l1(x)
        x = F.normalize(x, p=2, dim=1)
        x = F.gelu(x)
        x = self.l2(x) / self.temperature
        return torch.softmax(x, dim=1)

    def reset_to_prior(self, prior):
        prior = torch.as_tensor(prior, dtype=self.l2.bias.dtype, device=self.l2.bias.device)
        prior = prior / prior.sum()
        with torch.no_grad():
            self.l2.weight.zero_()
            self.l2.bias.copy_(torch.log(prior.clamp_min(1e-8)) * self.temperature)


class CrossModalAttentiveMoE(torch.nn.Module):
    """Cross-modal MoE that updates each modality through residual expert deltas."""

    def __init__(self, hidden_size, num_experts=4, dropout=0.1, residual_scale=0.1):
        super(CrossModalAttentiveMoE, self).__init__()
        self.num_experts = int(num_experts)
        self.residual_scale = float(residual_scale)
        input_size = hidden_size * 3
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_size, input_size),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(input_size, input_size),
            )
            for _ in range(self.num_experts)
        ])
        self.gate = torch.nn.Linear(input_size, 3 * self.num_experts)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm_id = torch.nn.LayerNorm(hidden_size)
        self.norm_text = torch.nn.LayerNorm(hidden_size)
        self.norm_visual = torch.nn.LayerNorm(hidden_size)

    def forward(self, id_repr, text_repr, visual_repr):
        x = torch.cat([id_repr, text_repr, visual_repr], dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        expert_id, expert_text, expert_visual = torch.chunk(expert_outputs, 3, dim=-1)
        gates = torch.softmax(self.gate(x).view(x.size(0), 3, self.num_experts), dim=-1)

        delta_id = torch.sum(gates[:, 0].unsqueeze(-1) * expert_id, dim=1)
        delta_text = torch.sum(gates[:, 1].unsqueeze(-1) * expert_text, dim=1)
        delta_visual = torch.sum(gates[:, 2].unsqueeze(-1) * expert_visual, dim=1)

        id_repr = self.norm_id(id_repr + self.residual_scale * self.dropout(delta_id))
        text_repr = self.norm_text(text_repr + self.residual_scale * self.dropout(delta_text))
        visual_repr = self.norm_visual(visual_repr + self.residual_scale * self.dropout(delta_visual))
        return id_repr, text_repr, visual_repr
    
class DIFM2SR(SequentialRecommender):
    r"""
    """

    def __init__(self, config, dataset, co_data, co_lens):
        super(DIFM2SR, self).__init__(config, dataset)
        
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.co_seq = F.normalize(self.get_co(co_data,co_lens), dim=1).to(self.device)
        self.pos_emb = torch.nn.Embedding(self.max_seq_length, self.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=config['hidden_dropout_prob'])
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
        self.means_k = config['means_k']
        self.knn_k = config['knn_k']
        self.bal = config['bal']
        self.miu_c = config['miu_c']
        self.miu_m = config['miu_m']
        self.mb = config['mb']
        self.fusion_temperature = config['fusion_temperature']
        self.fusion_scale = config['fusion_scale']
        self.fusion_prior = config['fusion_prior']
        self.fusion_dynamic_ratio = config['fusion_dynamic_ratio']
        self.w_balw = float(cfg_get(config, 'w_balw', 0.0))
        self.balw_type = str(cfg_get(config, 'balw_type', 'sample')).lower()
        self.balw_max_weight = float(cfg_get(config, 'balw_max_weight', 0.75))
        self.balw_log_batches = int(cfg_get(config, 'balw_log_batches', 5))
        self._balw_logged_batches = 0
        if self.balw_type not in {'sample', 'batch', 'threshold'}:
            raise ValueError("balw_type must be one of: sample, batch, threshold")
        self.w_simw = float(cfg_get(config, 'w_simw', 0.0))
        self.simw_mu = float(cfg_get(config, 'simw_mu', 2.0))
        self.simw_target = str(cfg_get(config, 'simw_target', 'router')).lower()
        self.simw_log_batches = int(cfg_get(config, 'simw_log_batches', 5))
        self._simw_logged_batches = 0
        if self.simw_target not in {'router', 'final'}:
            raise ValueError("simw_target must be one of: router, final")
        self.kmeans = MiniBatchKMeans(n_clusters=self.means_k, init_size=1024, batch_size=1024, random_state=100)
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.item_embedding = torch.nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.use_diffusion_denoiser = bool(cfg_get(config, 'use_diffusion_denoiser', False))
        self.condition_type = cfg_get(config, 'condition_type', 'id_graph')
        self.diffusion_denoiser = None
        self._diffusion_shapes_logged = False
        self.intent_weight_mapper = IntentAwareWeightMapper(self.hidden_size, self.fusion_temperature)
        self.use_cross_modal_moe = bool(cfg_get(config, 'use_cross_modal_moe', True))
        self.cross_modal_moe = CrossModalAttentiveMoE(
            hidden_size=self.hidden_size,
            num_experts=int(cfg_get(config, 'cross_moe_experts', 4)),
            dropout=float(cfg_get(config, 'cross_moe_dropout', config['hidden_dropout_prob'])),
            residual_scale=float(cfg_get(config, 'cross_moe_residual_scale', 0.1)),
        )
        self.w_future = float(cfg_get(config, 'w_future', 0.0))
        self.w_future_contrastive = float(cfg_get(config, 'w_future_contrastive', 0.0))
        self.future_horizon = int(cfg_get(config, 'future_horizon', 3))
        self.future_temperature = float(cfg_get(config, 'future_temperature', 0.07))
        self.future_min_confidence = float(cfg_get(config, 'future_min_confidence', 0.1))
        self.future_log_batches = int(cfg_get(config, 'future_log_batches', 5))
        self._future_logged_batches = 0
        fusion_prior = torch.as_tensor(self.fusion_prior, dtype=torch.float32)
        self.register_buffer("fusion_prior_weight", fusion_prior / fusion_prior.sum())
        
        self.ssl = 'us_x'
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.sem_aug_nce_fct = nn.CrossEntropyLoss()
        self.LayerNorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(config['n_layers']):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  torch.nn.MultiheadAttention(self.hidden_size, config['n_heads'], config['hidden_dropout_prob'])                                                               
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(self.hidden_size, config['hidden_dropout_prob'])
            self.forward_layers.append(new_fwd_layer) 

        self.knn_k_co = self.knn_k // self.bal
        self.knn_k_ma = self.knn_k - self.knn_k_co
        self.image_embedding = (dataset.image_embedding).to(self.device)
        self.text_embedding = (dataset.text_embedding).to(self.device)
        self.raw_img_embs = self.image_embedding.weight.detach().clone().to(self.device)
        self.raw_text_embs = self.text_embedding.weight.detach().clone().to(self.device)
        self.build_modal_structures(self.raw_text_embs, self.raw_img_embs)
        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)
        self.intent_weight_mapper.reset_to_prior(self.fusion_prior)
        if self.use_diffusion_denoiser:
            if self.condition_type != 'id_graph':
                raise ValueError("V1B only supports condition_type='id_graph'.")
            self.diffusion_denoiser = GraphConditionedDiffusionDenoiser(
                feature_dim=self.raw_text_embs.shape[1],
                condition_dim=self.hidden_size * 2,
                hidden_dim=max(self.hidden_size, self.raw_text_embs.shape[1]),
                diffusion_steps=int(cfg_get(config, 'diffusion_steps', 8)),
            ).to(self.device)

    def build_modal_structures(self, text_features, image_features):
        """Build CGC graph and CIP centers from the provided modal features."""
        self.text_embs = text_features.detach().clone().to(self.device).float()
        self.img_embs = image_features.detach().clone().to(self.device).float()
        self.co_img_embs = self.co_seq @ self.img_embs
        self.co_text_embs = self.co_seq @ self.text_embs
        _, self.co_vm_adj = self.get_knn_adj_mat(self.img_embs)
        _, self.co_tm_adj = self.get_knn_adj_mat(self.text_embs)
        self.sensev, self.vsample = self.get_center(self.img_embs)
        self.senset, self.tsample = self.get_center(self.text_embs)

    def build_diffusion_condition(self, item_id_embeddings):
        if self.condition_type != 'id_graph':
            raise ValueError("V1B only supports condition_type='id_graph'.")
        item_id_embeddings = item_id_embeddings.detach().to(self.device).float()
        graph_id_embeddings = self.co_seq @ item_id_embeddings
        condition = torch.cat([item_id_embeddings, graph_id_embeddings], dim=-1)
        condition[0].zero_()
        return condition

    def warmup_diffusion_denoiser(self, epochs, batch_size, lr, beta_graph, logger=None):
        if self.diffusion_denoiser is None:
            raise RuntimeError("Diffusion denoiser is not enabled.")

        original_requires_grad = {p: p.requires_grad for p in self.parameters()}
        self.diffusion_denoiser.train()
        for p in self.parameters():
            p.requires_grad = False
        for p in self.diffusion_denoiser.parameters():
            p.requires_grad = True

        learned_id = self.item_embedding.weight.detach()
        condition = self.build_diffusion_condition(learned_id)
        text_graph = (self.co_seq @ self.raw_text_embs).detach()
        image_graph = (self.co_seq @ self.raw_img_embs).detach()
        item_indices = torch.arange(1, self.n_items, device=self.device)
        optimizer = torch.optim.Adam(self.diffusion_denoiser.parameters(), lr=lr)

        if not self._diffusion_shapes_logged:
            msg = (
                "[diffusion] "
                f"learned_id={tuple(learned_id.shape)} "
                f"raw_text={tuple(self.raw_text_embs.shape)} raw_image={tuple(self.raw_img_embs.shape)} "
                f"condition={tuple(condition.shape)} beta_graph={beta_graph}"
            )
            print(msg)
            if logger is not None:
                logger.info(msg)
            self._diffusion_shapes_logged = True

        for epoch in range(int(epochs)):
            perm = item_indices[torch.randperm(item_indices.numel(), device=self.device)]
            total_loss = 0.0
            total_diff = 0.0
            total_graph = 0.0
            steps = 0
            for start in range(0, perm.numel(), int(batch_size)):
                idx = perm[start:start + int(batch_size)]
                optimizer.zero_grad()
                loss, diff_loss, graph_loss = self.diffusion_denoiser.warmup_loss(
                    self.raw_text_embs[idx],
                    self.raw_img_embs[idx],
                    condition[idx],
                    text_graph[idx],
                    image_graph[idx],
                    beta_graph,
                )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                total_diff += float(diff_loss.cpu())
                total_graph += float(graph_loss.cpu())
                steps += 1
            msg = (
                f"[diffusion] warmup epoch {epoch + 1}/{epochs} "
                f"loss={total_loss / max(steps, 1):.6f} "
                f"diff={total_diff / max(steps, 1):.6f} "
                f"graph={total_graph / max(steps, 1):.6f}"
            )
            print(msg)
            if logger is not None:
                logger.info(msg)

        for p, requires_grad in original_requires_grad.items():
            p.requires_grad = requires_grad
        self.diffusion_denoiser.eval()

    @torch.no_grad()
    def generate_clean_modal_features(self, batch_size):
        if self.diffusion_denoiser is None:
            raise RuntimeError("Diffusion denoiser is not enabled.")
        learned_id = self.item_embedding.weight.detach()
        condition = self.build_diffusion_condition(learned_id)
        clean_text, clean_image = [], []
        for start in range(0, self.n_items, int(batch_size)):
            end = min(start + int(batch_size), self.n_items)
            text_batch, image_batch = self.diffusion_denoiser.denoise_batch(
                self.raw_text_embs[start:end],
                self.raw_img_embs[start:end],
                condition[start:end],
            )
            clean_text.append(text_batch.detach())
            clean_image.append(image_batch.detach())
        text_clean = torch.cat(clean_text, dim=0)
        image_clean = torch.cat(clean_image, dim=0)
        text_clean[0].zero_()
        image_clean[0].zero_()
        msg = (
            "[diffusion] "
            f"clean_text={tuple(text_clean.shape)} clean_image={tuple(image_clean.shape)}"
        )
        print(msg)
        return text_clean, image_clean

    def mod(self, build_item_graph=True):
        h = self.item_embedding.weight.to(self.device)
        vcoh = torch.mm(self.co_vm_adj, h)
        tcoh = torch.mm(self.co_tm_adj, h)
        if self.use_cross_modal_moe:
            h, tcoh, vcoh = self.cross_modal_moe(h, tcoh, vcoh)
            padding_mask = torch.ones((h.size(0), 1), dtype=h.dtype, device=h.device)
            padding_mask[0] = 0
            h = h * padding_mask
            tcoh = tcoh * padding_mask
            vcoh = vcoh * padding_mask
        return h,vcoh,tcoh
    
    def forward(self, item_seq, item_seq_len):
        log_seqs = item_seq.cpu().numpy()
        ID,hmv_emb,hmt_emb = self.mod()  
        
        hv_after = torch.mm(self.sensev,self.co_vm_adj.to_dense())
        ht_after = torch.mm(self.senset,self.co_tm_adj.to_dense())
        hv_after = hv_after @ ID
        ht_after = ht_after @ ID     
        
        seqsv = hmv_emb[torch.LongTensor(log_seqs).to(self.device)]
        seqst = hmt_emb[torch.LongTensor(log_seqs).to(self.device)]
        seqsi = ID[torch.LongTensor(log_seqs).to(self.device)]
        cc = torch.tile(torch.arange(hv_after.shape[0]), (log_seqs.shape[0], 1))
        vsg=hv_after[cc]
        tsg=ht_after[cc]
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.device)

        # idco
        co_sensei = self.co_seq @ ID
        co_sense = co_sensei[torch.LongTensor(log_seqs).to(self.device)] 
        
        # id
        seqsi = co_sense 
        seqsi *= self.item_embedding.embedding_dim ** 0.9
        seqsi += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqsi = self.emb_dropout(seqsi)
        tl = seqsi.shape[1]
        attention_maski = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))
        seqsi *= ~timeline_mask.unsqueeze(-1) 

        # image
        seqsv *= self.item_embedding.embedding_dim ** 0.5
        seqsv += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqsv = self.emb_dropout(seqsv)
        tl = seqsv.shape[1]
        attention_maskv = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))
        
        seqsv *= ~timeline_mask.unsqueeze(-1)

        # text       
        seqst *= self.item_embedding.embedding_dim ** 0.5
        seqst += self.pos_emb(torch.LongTensor(positions).to(self.device))        
        seqst = self.emb_dropout(seqst)
        t2 = seqst.shape[1]
        attention_maskt = ~torch.tril(torch.ones((t2, t2), dtype=torch.bool, device=self.device))
        seqst *= ~timeline_mask.unsqueeze(-1)

        # star
        vsg = vsg * self.item_embedding.embedding_dim ** 0.5
        tsg = tsg * self.item_embedding.embedding_dim ** 0.5
        vsg = self.emb_dropout(vsg)
        tsg = self.emb_dropout(tsg)     
        suov = self.compute_max_similarity_index(seqsv,vsg)
        suot = self.compute_max_similarity_index(seqst,tsg)
        attsg = ~torch.tril(torch.ones((vsg.shape[1], t2), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):    
            vsg = torch.transpose(vsg, 0, 1)
            Qvsg = self.attention_layernorms[i](vsg)          
            seqsv = torch.transpose(seqsv, 0, 1)           
            Qv = self.attention_layernorms[i](seqsv)
            vvv, _= self.attention_layers[i](Qvsg, seqsv, seqsv, attn_mask=attsg)
            mha_outputsv, _= self.attention_layers[i](Qv, seqsv, seqsv, attn_mask=attention_maskv)     
            seqsv = Qv + mha_outputsv
            seqsv = torch.transpose(seqsv, 0, 1)
            seqsv = self.forward_layernorms[i](seqsv)
            seqsv = self.forward_layers[i](seqsv)
            seqsv *=  ~timeline_mask.unsqueeze(-1)                
            vsg = Qvsg + vvv
            vsg = torch.transpose(vsg, 0, 1)
            vsg = self.forward_layers[i](vsg)
              
            tsg = torch.transpose(tsg, 0, 1)
            Qtsg = self.attention_layernorms[i](tsg)
            seqst = torch.transpose(seqst, 0, 1)
            Qt = self.attention_layernorms[i](seqst)
            ttt, _= self.attention_layers[i](Qtsg, seqst, seqst, attn_mask=attsg)
            mha_outputst, _= self.attention_layers[i](Qt, seqst, seqst, attn_mask=attention_maskt) 
            seqst = Qt + mha_outputst
            seqst = torch.transpose(seqst, 0, 1)
            seqst = self.forward_layernorms[i](seqst)
            seqst = self.forward_layers[i](seqst)
            seqst *=  ~timeline_mask.unsqueeze(-1)
            tsg = Qtsg + ttt
            tsg = torch.transpose(tsg, 0, 1)
            tsg = self.forward_layers[i](tsg)

            seqsi = torch.transpose(seqsi, 0, 1)
            Qi = self.attention_layernorms[i](seqsi)
            mha_outputsi, _= self.attention_layers[i](Qi, seqsi, seqsi, attn_mask=attention_maski)
            seqsi = Qi + mha_outputsi
            seqsi = torch.transpose(seqsi, 0, 1)
            seqsi = self.forward_layernorms[i](seqsi)
            seqsi = self.forward_layers[i](seqsi)
            seqsi *=  ~timeline_mask.unsqueeze(-1)  

        cenv = torch.matmul(suov, vsg)
        cent = torch.matmul(suot, tsg)
        log_featsv = self.last_layernorm(seqsv + cenv)
        log_featst = self.last_layernorm(seqst + cent)
        outputv = self.gather_indexes(log_featsv, item_seq_len - 1) 
        outputt = self.gather_indexes(log_featst, item_seq_len - 1) 

        u_id = self.gather_indexes(seqsi, item_seq_len - 1)
        u_text = self.gather_indexes(seqst, item_seq_len - 1)
        u_visual = self.gather_indexes(seqsv, item_seq_len - 1)
        fusion_input = torch.cat([u_id, u_text, u_visual], dim=-1)
        router_weights = self.intent_weight_mapper(fusion_input)
        prior_weights = self.fusion_prior_weight.to(dtype=router_weights.dtype).view(1, -1)
        weights = (1 - self.fusion_dynamic_ratio) * prior_weights + self.fusion_dynamic_ratio * router_weights
        w_id = weights[:, 0].view(-1, 1, 1)
        w_text = weights[:, 1].view(-1, 1, 1)
        w_visual = weights[:, 2].view(-1, 1, 1)
        seqs = self.fusion_scale * (w_id * seqsi + w_text * seqst + w_visual * seqsv)

        log_feats = self.last_layernorm(seqs) 
        output = self.gather_indexes(log_feats, item_seq_len - 1)   
        return output, outputv, outputt, weights, router_weights, log_feats, ID, hmv_emb, hmt_emb  # [B H]

    def BALW(self, modality_weights):
        """Entropy balance regularization for id/text/visual fusion weights."""
        modality_weights = torch.clamp(modality_weights, min=1e-9)
        if self.balw_type == 'batch':
            mean_weights = torch.clamp(modality_weights.mean(dim=0), min=1e-9)
            n_modalities = mean_weights.numel()
            return n_modalities * torch.sum(mean_weights * torch.log(mean_weights))
        if self.balw_type == 'threshold':
            max_weights = modality_weights.max(dim=1).values
            return torch.mean(torch.relu(max_weights - self.balw_max_weight) ** 2)
        n_modalities = modality_weights.size(1)
        return torch.mean(n_modalities * torch.sum(modality_weights * torch.log(modality_weights), dim=1))

    def maybe_log_fusion_weights(self, modality_weights):
        if self.w_balw <= 0 or self._balw_logged_batches >= self.balw_log_batches:
            return
        with torch.no_grad():
            mean_weights = modality_weights.detach().mean(dim=0).cpu().tolist()
            max_weight = float(modality_weights.detach().max(dim=1).values.mean().cpu())
        msg = (
            "[balw] "
            f"type={self.balw_type} "
            f"mean_id={mean_weights[0]:.4f} mean_text={mean_weights[1]:.4f} "
            f"mean_visual={mean_weights[2]:.4f} mean_max={max_weight:.4f}"
        )
        print(msg)
        self._balw_logged_batches += 1

    def eva_imp(self, score, target, mu):
        target_logits = score.gather(1, target.unsqueeze(1)).squeeze(1)
        rank = (score > target_logits.unsqueeze(1)).sum(dim=1) + 1
        uncertainty = (rank - 1).float() / score.size(1)
        return torch.exp(mu * uncertainty.clamp(min=1e-8))

    def SIMW(self, scores, target, weights, mu):
        """Ranking-feedback ideal weight supervision for id/text/visual router."""
        id_dist = self.eva_imp(scores[0], target, mu)
        text_dist = self.eva_imp(scores[1], target, mu)
        visual_dist = self.eva_imp(scores[2], target, mu)
        dists = torch.stack([id_dist, text_dist, visual_dist], dim=1)
        ideal_weights = torch.softmax(-dists, dim=-1).detach()
        loss = F.kl_div(
            torch.log(torch.clamp(weights, min=1e-9)),
            ideal_weights,
            reduction='batchmean',
        )
        self.maybe_log_simw(weights, ideal_weights)
        return loss

    def maybe_log_simw(self, weights, ideal_weights):
        if self.w_simw <= 0 or self._simw_logged_batches >= self.simw_log_batches:
            return
        with torch.no_grad():
            router_mean = weights.detach().mean(dim=0).cpu().tolist()
            ideal_mean = ideal_weights.detach().mean(dim=0).cpu().tolist()
        msg = (
            "[simw] "
            f"target={self.simw_target} "
            f"ideal_id={ideal_mean[0]:.4f} ideal_text={ideal_mean[1]:.4f} "
            f"ideal_visual={ideal_mean[2]:.4f} "
            f"router_id={router_mean[0]:.4f} router_text={router_mean[1]:.4f} "
            f"router_visual={router_mean[2]:.4f}"
        )
        print(msg)
        self._simw_logged_batches += 1

    def future_aware_auxiliary_loss(self, log_feats, item_seq, item_seq_len, item_emb):
        """Auxiliary future supervision from earlier states to later items in the same sequence."""
        if self.w_future <= 0 and self.w_future_contrastive <= 0:
            return log_feats.new_tensor(0.0)
        if self.future_horizon <= 0:
            return log_feats.new_tensor(0.0)

        last_idx = (item_seq_len - 1).clamp(min=0)
        target_items = item_seq.gather(1, last_idx.view(-1, 1)).squeeze(1)
        target_states = self.gather_indexes(log_feats, last_idx)
        losses = []
        contrastive_losses = []

        for horizon in range(1, self.future_horizon + 1):
            valid = (item_seq_len > horizon) & (target_items > 0)
            if not torch.any(valid):
                continue

            source_idx = (item_seq_len - 1 - horizon).clamp(min=0)
            source_states = self.gather_indexes(log_feats, source_idx)
            valid_source = source_states[valid]
            valid_target_items = target_items[valid]

            if self.w_future > 0:
                logits = torch.matmul(valid_source, item_emb.transpose(0, 1))
                ce = F.cross_entropy(logits, valid_target_items, reduction='none')
                confidence = torch.exp(-ce.detach()).clamp(min=self.future_min_confidence, max=1.0)
                losses.append(torch.mean(confidence * ce))

            if self.w_future_contrastive > 0 and int(valid.sum().item()) > 1:
                src = F.normalize(valid_source, dim=-1)
                tgt = F.normalize(target_states[valid].detach(), dim=-1)
                logits = torch.matmul(src, tgt.transpose(0, 1)) / self.future_temperature
                labels = torch.arange(logits.size(0), device=logits.device)
                contrastive_losses.append(F.cross_entropy(logits, labels))

        if not losses and not contrastive_losses:
            return log_feats.new_tensor(0.0)

        aux_loss = log_feats.new_tensor(0.0)
        if losses:
            aux_loss = aux_loss + self.w_future * torch.stack(losses).mean()
        if contrastive_losses:
            aux_loss = aux_loss + self.w_future_contrastive * torch.stack(contrastive_losses).mean()
        self.maybe_log_future_loss(aux_loss)
        return aux_loss

    def maybe_log_future_loss(self, loss):
        if self._future_logged_batches >= self.future_log_batches:
            return
        print(
            "[future] "
            f"horizon={self.future_horizon} "
            f"w_future={self.w_future} "
            f"w_contrastive={self.w_future_contrastive} "
            f"loss={float(loss.detach().cpu()):.6f}"
        )
        self._future_logged_batches += 1
    
    def compute_max_similarity_index(self, j, i):
        similarity = torch.matmul(j, i.transpose(1, 2))
        tensor_reshaped = similarity.view(-1, self.means_k)  
        result_reshaped = F.gumbel_softmax(tensor_reshaped, tau=1, hard=False)
        result = result_reshaped.view(j.shape[0], j.shape[1], self.means_k)
        return result

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        (
            seq_output, outputv, outputt, modality_weights, router_weights,
            log_feats, id_emb, hmv_emb, hmt_emb
        ) = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = id_emb[pos_items]
            neg_items_emb = id_emb[neg_items]
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = id_emb
            id_logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            visual_logits = torch.matmul(outputv, hmv_emb.transpose(0, 1))
            text_logits = torch.matmul(outputt, hmt_emb.transpose(0, 1))
            loss = self.loss_fct(id_logits, pos_items)
            loss += self.mb * self.loss_fct(visual_logits, pos_items)
            loss += (1 - self.mb) * self.loss_fct(text_logits, pos_items)
            if self.w_simw > 0:
                simw_weights = router_weights if self.simw_target == 'router' else modality_weights
                loss += self.w_simw * self.SIMW(
                    [id_logits, text_logits, visual_logits],
                    pos_items,
                    simw_weights,
                    self.simw_mu,
                )

        if self.w_balw > 0:
            self.maybe_log_fusion_weights(modality_weights)
            loss += self.w_balw * self.BALW(modality_weights)

        loss += self.future_aware_auxiliary_loss(log_feats, item_seq, item_seq_len, id_emb)

        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        (
            seq_output, outputv, outputt, _, _, _, id_emb, vcoh, tcoh
        ) = self.forward(item_seq, item_seq_len)
        test_items_emb = id_emb
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        scores += self.mb * torch.matmul(outputv, vcoh.transpose(0, 1))  # [B]
        scores += (1 - self.mb) * torch.matmul(outputt, tcoh.transpose(0, 1))  # [B] 
        return scores 
    
    def get_co(self, seqs, lens):
        seqs = np.asarray(seqs, dtype=np.int64)
        lens = np.asarray(lens, dtype=np.int64)
        co_mat = np.zeros((self.n_items, self.n_items), dtype=np.float32)

        for row, seq_len in zip(seqs, lens):
            items = row[:seq_len]
            items = items[items != 0]
            seq_len = len(items)
            for distance in range(1, seq_len):
                src = items[:-distance]
                dst = items[distance:]
                weight = np.float32(1.0 / distance)
                np.add.at(co_mat, (src, dst), weight)
                np.add.at(co_mat, (dst, src), weight)

        return torch.from_numpy(co_mat)

    def extract_common_and_complement(self, a, b, n):
        m, _ = a.shape
        c = torch.full((m, n), -1, dtype=torch.int64)

        for i in range(m):
            row_a = a[i].tolist()
            row_b = b[i].tolist()
            common_elements = list(set(row_a) & set(row_b))
            remaining_elements = [x for x in row_a if x not in common_elements]
            c[i][:len(common_elements)] = torch.tensor(common_elements, dtype=torch.int64)
            c[i][len(common_elements):] = torch.tensor(remaining_elements[:n - len(common_elements)], dtype=torch.int64)
        
        return c.to(self.device)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind_ma = torch.topk(sim, int(self.knn_k * self.miu_m), dim=-1)
        _, knn_ind_co = torch.topk(self.co_seq, int(self.knn_k * self.miu_c), dim=-1)
        knn_ind = self.extract_common_and_complement(knn_ind_ma, knn_ind_co, self.knn_k)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)
        
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size) 
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_center(self,embs):
        means = embs.detach().cpu().numpy()
        self.kmeans.fit(means)
        centers = torch.tensor(self.kmeans.cluster_centers_).to(self.device)
        sample = torch.tensor(self.kmeans.labels_)
        o = torch.zeros(1, self.n_items).to(self.device)
        for i in range(max(sample)+1):
            op=copy.deepcopy(sample).unsqueeze(0).to(self.device)
            for j in range(self.n_items-1):
                if op[0,j]==i:
                    op[0,j]=1
                else:
                    op[0,j]=0
            o=torch.cat((o, op), 0)
        sense=o[1:]
        sense = sense/(1e-7+torch.sum(sense,dim=-1).unsqueeze(1))
        return sense,sample
