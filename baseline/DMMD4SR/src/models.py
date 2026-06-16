import math
import os
import pickle
from tqdm import tqdm
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from baseline.DMMD4SR.src.modules import Encoder, LayerNorm
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from baseline.DMMD4SR.src.diffusion import SDNet, DiffusionProcess


class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()

        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.n_clusters = args.n_clusters
        self.lambda_history = args.lambda_history
        self.lambda_intent = args.lambda_intent

        self.encoder = Encoder(args)

        # 多模态
        self.text_embeddings = nn.Embedding(args.item_size, args.pretrain_emb_dim, padding_idx=0)
        self.img_embeddings = nn.Embedding(args.item_size, args.pretrain_emb_dim, padding_idx=0)

        # 全连接层映射
        self.fc_img = nn.Linear(args.pretrain_emb_dim, args.hidden_size)
        self.fc_text = nn.Linear(args.pretrain_emb_dim, args.hidden_size)

        # 扩散模型
        self.time_emb_dim = 16
        self.steps = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        in_dims = [args.hidden_size, args.hidden_size]
        out_dims = [args.hidden_size, args.hidden_size]

        # 第一层：域差异性扩散模型
        self.sdnet_td = SDNet(in_dims=in_dims, out_dims=out_dims, emb_size=self.time_emb_dim)  # 文本域差异性
        self.sdnet_vd = SDNet(in_dims=in_dims, out_dims=out_dims, emb_size=self.time_emb_dim)  # 图像域差异性
        
        # 第二层：背景噪声扩散模型（条件扩散）
        self.sdnet_tc = SDNet(in_dims=in_dims, out_dims=out_dims, emb_size=self.time_emb_dim)  # 文本条件去噪
        self.sdnet_vc = SDNet(in_dims=in_dims, out_dims=out_dims, emb_size=self.time_emb_dim)  # 图像条件去噪
        self.sdnet_f = SDNet(in_dims=in_dims, out_dims=out_dims, emb_size=self.time_emb_dim)   # 最终序列去噪
        
        self.mu_text = nn.Linear(args.hidden_size, args.hidden_size)
        self.sigma_text = nn.Linear(args.hidden_size, args.hidden_size)
        self.mu_img = nn.Linear(args.hidden_size, args.hidden_size)
        self.sigma_img = nn.Linear(args.hidden_size, args.hidden_size)

        self.num_experts = 4
        self.text_experts = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(self.num_experts)])
        self.img_experts = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(self.num_experts)])


        self.fusion_layer = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU()
        )

        self.gate = nn.Linear(args.hidden_size, self.num_experts)

        # ID表征处理网络
        self.id_projection = nn.Linear(args.hidden_size, args.hidden_size)
        self.mlp= nn.Sequential(nn.Linear(args.hidden_size*args.max_seq_length, self.n_clusters))

        # 条件融合网络 - 学习如何结合ID条件和去噪表征
        self.condition_fusion = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size)
        )
        
        # 门控机制 - 控制条件信息的注入量
        self.condition_gate = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.Sigmoid()
        )

        noise_schedule = "linear"
        noise_scale = [1, 0.1]
        noise_min = 0.0001
        noise_max = 0.02
        self.denoise_weight = 0.08
        self.diffusion_process = DiffusionProcess(
            noise_schedule=noise_schedule,
            noise_scale=noise_scale,
            noise_min=noise_min,
            noise_max=noise_max,
            steps=self.steps,
            device=self.device
        )

        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.apply(self.init_weights)

        # multimodal embedding
        self.replace_embedding()

    def replace_embedding(self):
        """ multimodal init """

        text_features_list = self._load_feature_tensor(self.args.text_embedding_path)
        image_features_list = self._load_feature_tensor(self.args.image_embedding_path)
        self._copy_feature_weight(self.text_embeddings, text_features_list)
        self._copy_feature_weight(self.img_embeddings, image_features_list)

    def _load_feature_tensor(self, path):
        if path.endswith(".npy") or not path.endswith(".pt"):
            return torch.from_numpy(np.load(path)).float()
        return torch.load(path, map_location="cpu").float()

    def _copy_feature_weight(self, embedding, features):
        if features.shape[1] != embedding.embedding_dim:
            raise ValueError(
                f"Feature dim {features.shape[1]} does not match pretrain_emb_dim "
                f"{embedding.embedding_dim}. Set --pretrain_emb_dim {features.shape[1]}."
            )

        token_ids = getattr(self.args, "item_token_ids", None)
        if token_ids is None:
            copy_len = min(features.shape[0], embedding.weight.data.shape[0] - 2)
            embedding.weight.data[1:1 + copy_len, :] = features[:copy_len]
            return

        for internal_id, token in enumerate(token_ids):
            if internal_id == 0 or token is None:
                continue
            try:
                feature_idx = int(token)
            except ValueError:
                continue
            if 0 <= feature_idx < features.shape[0] and internal_id < embedding.weight.data.shape[0]:
                embedding.weight.data[internal_id, :] = features[feature_idx]
    
    def multimodal_fusion(self, text_embeddings, img_embeddings, item_embeddings=None):
        """
            Uncertainty-Guided Modality Denoising Fusion Layer
        """
        
        
        # Enhanced experts to capture uncertainty for each modality
        t_mu = self.mu_text(text_embeddings)
        t_sigma = torch.exp(self.sigma_text(text_embeddings))
        
        i_mu = self.mu_img(img_embeddings)
        i_sigma = torch.exp(self.sigma_img(img_embeddings))
        
        # Reparameterization trick for sampling
        t_z = t_mu + t_sigma * torch.randn_like(t_mu)
        i_z = i_mu + i_sigma * torch.randn_like(i_mu)
        
        
        t_gate = self.topk_routing_text(t_z)
        i_gate = self.topk_routing_img(i_z)
        

        # Process features through corresponding experts
        expert_outputs = [expert(t_z) for expert in self.text_experts]
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-2)  # Shape: [B, S, num_experts, H]
        # t_gate shape is [B, S, num_experts]
        gate_weights = t_gate.unsqueeze(-1)  # Shape: [B, S, num_experts, 1]
        t_out = torch.sum(stacked_expert_outputs * gate_weights, dim=-2) # Weighted sum -> Shape: [B, S, H]

        # Process features through corresponding image experts
        expert_outputs = [expert(i_z) for expert in self.img_experts]
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-2) # Shape: [B, S, num_experts, H]
        # i_gate shape is [B, S, num_experts]
        gate_weights = i_gate.unsqueeze(-1) # Shape: [B, S, num_experts, 1]
        i_out = torch.sum(stacked_expert_outputs * gate_weights, dim=-2) # Weighted sum -> Shape: [B, S, H]
        
        # Combine modalities with uncertainty-aware weighting
        fusion = item_embeddings + self.fusion_layer(torch.cat([t_out, i_out], dim=-1))

        
        return fusion, t_mu, t_sigma, i_mu, i_sigma

    def topk_routing_text(self, gate_logits, k=2):
        # Apply softmax
        gate_logits = self.gate(gate_logits)

        routing_weights = F.softmax(gate_logits, dim=-1)
        
        # topk
        top_k_weights, indices = torch.topk(routing_weights, k=k, dim=-1)
        routing_weights_new = torch.zeros_like(routing_weights)
        routing_weights_new.scatter_(-1, indices, top_k_weights)

        # norm
        routing_weights_new = routing_weights_new / routing_weights_new.sum(dim=-1, keepdim=True)
        return routing_weights_new

    def topk_routing_img(self, gate_logits, k=2):
        # Apply softmax
        gate_logits = self.gate(gate_logits)

        routing_weights = F.softmax(gate_logits, dim=-1)

        # topk
        top_k_weights, indices = torch.topk(routing_weights, k=k, dim=-1)
        routing_weights_new = torch.zeros_like(routing_weights)
        routing_weights_new.scatter_(-1, indices, top_k_weights)

        # norm
        routing_weights_new = routing_weights_new / routing_weights_new.sum(dim=-1, keepdim=True)
        return routing_weights_new
    
    
    def compute_kl_loss(self, mu, sigma):
        # Add epsilon for numerical stability inside the log
        epsilon = 1e-8
        # The formula is -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Stable calculation: -0.5 * sum(1 + 2*log(sigma) - mu^2 - sigma^2)
        # Or adding epsilon to variance:
        kl_loss = -0.5 * torch.mean(torch.sum(1 + torch.log(sigma.pow(2) + epsilon) - mu.pow(2) - sigma.pow(2), dim=-1))
        return kl_loss



    def get_modality_embeddings(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        # id embedding
        item_embeddings = self.item_embeddings(
            sequence)  # shape: batch_size, sequence_length, args.hidden_size
        position_embeddings = self.position_embeddings(position_ids)

        t_emb_loss = 0
        v_emb_loss = 0
        text_embeddings_denoised = None
        img_embeddings_denoised = None
        
        if self.args.is_use_mm:
            # get text and image embedding
            text_embeddings = self.text_embeddings(sequence)
            text_embeddings = self.fc_text(text_embeddings)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
            
            img_embeddings = self.img_embeddings(sequence)
            img_embeddings = self.fc_img(img_embeddings)
            img_embeddings = F.normalize(img_embeddings, p=2, dim=-1)
            

        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb, text_embeddings, img_embeddings, item_embeddings
    
    def intent_cluster(self,input,n_clusters):
        X=input.view(input.shape[0],-1)

        centers = X[torch.randperm(X.size(0))[:n_clusters]].to(input.device)
        labels=self.mlp(X)
        labels = F.gumbel_softmax(labels, tau=0.1, hard=True)
        # labels = torch.argmax(labels, dim=1)

        for i in range(self.n_clusters):
            if torch.sum(labels[:, i]) == 0:
                centers[i] = X[torch.randint(0, X.size(0), (1,))]
            else:
                centers[i] = torch.mean(X[labels[:, i].bool()], dim=0)
        return centers.view(n_clusters,input.shape[1],input.shape[-1]), torch.argmax(labels, dim=1)
    
    # model same as SASRec
    def forward(self, input_ids, is_train=False):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # multimodal embedding
        sequence_emb, text_embeddings, img_embeddings, item_id_emb = self.get_modality_embeddings(input_ids)

        # init loss
        t_emb_loss = 0
        v_emb_loss = 0
        diff_loss = 0
        
        if self.args.is_use_mm:
            if is_train:
                """ Multimodal Domain Shift Denoising Layer """
                # text
                t_domain_terms = self.diffusion_process.caculate_losses(self.sdnet_td, text_embeddings)
                t_domain_loss = t_domain_terms['loss'].mean()
                
                # image
                v_domain_terms = self.diffusion_process.caculate_losses(self.sdnet_vd, img_embeddings)
                v_domain_loss = v_domain_terms['loss'].mean()
                
                text_embeddings_denoised = self.diffusion_process.p_sample(self.sdnet_td, text_embeddings, steps=5, sampling_noise=False)
                img_embeddings_denoised = self.diffusion_process.p_sample(self.sdnet_vd, img_embeddings, steps=5, sampling_noise=False)

                
                
                """ Multimodal Interest-agnostic Denoising Layer """
                # ID_embdding as candition
                self.centroids, self.labels = self.intent_cluster(item_id_emb, self.n_clusters)
                id_condition = item_id_emb*self.lambda_history+self.centroids[self.labels]*self.lambda_intent
                
                # text
                text_concat_features = torch.cat([text_embeddings_denoised, id_condition], dim=-1)
                text_gate = self.condition_gate(text_concat_features)
                text_fusion_features = self.condition_fusion(text_concat_features)
                text_conditioned_emb = text_embeddings_denoised * (1 - text_gate) + text_fusion_features * text_gate
                
                # image
                img_concat_features = torch.cat([img_embeddings_denoised, id_condition], dim=-1)
                img_gate = self.condition_gate(img_concat_features)
                img_fusion_features = self.condition_fusion(img_concat_features)
                img_conditioned_emb = img_embeddings_denoised * (1 - img_gate) + img_fusion_features * img_gate
                
                # loss
                t_condition_loss = self.diffusion_process.caculate_losses(self.sdnet_tc, text_conditioned_emb)['loss'].mean()
                v_condition_loss = self.diffusion_process.caculate_losses(self.sdnet_vc, img_conditioned_emb)['loss'].mean()
                
                # after denoised
                text_embeddings_denoised = self.diffusion_process.p_sample(self.sdnet_tc, text_conditioned_emb, steps=5, sampling_noise=False)
                img_embeddings_denoised = self.diffusion_process.p_sample(self.sdnet_vc, img_conditioned_emb, steps=5, sampling_noise=False)
                
                #  final emb
                enhanced_seq_emb, t_mu, t_sigma, i_mu, i_sigma = self.multimodal_fusion(text_embeddings_denoised, img_embeddings_denoised, sequence_emb)
                # enhanced_seq_emb = sequence_emb + text_embeddings_denoised + img_embeddings_denoised
                
                # denoised loss
                final_diff_loss = self.diffusion_process.caculate_losses(self.sdnet_f, enhanced_seq_emb)['loss'].mean()
                kl_loss = self.compute_kl_loss(t_mu, t_sigma) + self.compute_kl_loss(i_mu, i_sigma)

                # total loss
                t_emb_loss = t_domain_loss
                v_emb_loss = v_domain_loss
                diff_loss = 0.5 * (t_condition_loss + v_condition_loss) + 0.5 * final_diff_loss + 2 * kl_loss

            else:  # inference
                """ Multimodal Domain Shift Denoising Layer """
                text_embeddings_denoised = self.diffusion_process.p_sample(self.sdnet_td, text_embeddings, steps=5, sampling_noise=False)
                img_embeddings_denoised = self.diffusion_process.p_sample(self.sdnet_vd, img_embeddings, steps=5, sampling_noise=False)
                

                """ Multimodal Interest-agnostic Denoising Layer """
                self.centroids, self.labels = self.intent_cluster(item_id_emb, self.n_clusters)
                id_condition = item_id_emb*self.lambda_history+self.centroids[self.labels]*self.lambda_intent
                
                # text
                text_concat_features = torch.cat([text_embeddings_denoised, id_condition], dim=-1)
                text_gate = self.condition_gate(text_concat_features)
                text_fusion_features = self.condition_fusion(text_concat_features)
                text_conditioned_emb = text_embeddings_denoised * (1 - text_gate) + text_fusion_features * text_gate
                text_denoised = self.diffusion_process.p_sample(self.sdnet_tc, text_conditioned_emb, steps=5, sampling_noise=False)
                
                # image
                img_concat_features = torch.cat([img_embeddings_denoised, id_condition], dim=-1)
                img_gate = self.condition_gate(img_concat_features)
                img_fusion_features = self.condition_fusion(img_concat_features)
                img_conditioned_emb = img_embeddings_denoised * (1 - img_gate) + img_fusion_features * img_gate
                img_denoised = self.diffusion_process.p_sample(self.sdnet_vc, img_conditioned_emb, steps=5, sampling_noise=False)

                enhanced_seq_emb, _, _, _, _ = self.multimodal_fusion(text_embeddings_denoised, img_embeddings_denoised, sequence_emb)
                
                # after denoised
                final_seq_emb = self.diffusion_process.p_sample(self.sdnet_f, enhanced_seq_emb, steps=5, sampling_noise=False)
             
                sequence_emb = (1 - self.denoise_weight) * sequence_emb + self.denoise_weight * final_seq_emb

        # transformer encoder
        item_encoded_layers = self.encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output, t_emb_loss, v_emb_loss, diff_loss

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

