import torch
from torch import Tensor
from typing import Optional, Tuple
from modules.transformer.model_padding_mask import TransformerDecoder
from torch import nn
from torch.nn import functional as F
from data.processed import SeqBatch
from typing import NamedTuple
from modules.utils import *
from utils.util import *

torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

class ModelOutput(NamedTuple):
    loss: Optional[dict] = None
    topk_idx: Optional[Tensor] = None

class UserPrefSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.qkv_out = nn.ModuleDict({
            "item": nn.ModuleDict({
                "q": nn.Linear(hidden_dim, hidden_dim),
                "k": nn.Linear(hidden_dim, hidden_dim),
                "v": nn.Linear(hidden_dim, hidden_dim),
                "out": nn.Linear(hidden_dim, hidden_dim),
            }),
            "text": nn.ModuleDict({
                "q": nn.Linear(hidden_dim, hidden_dim),
                "k": nn.Linear(hidden_dim, hidden_dim),
                "v": nn.Linear(hidden_dim, hidden_dim),
                "out": nn.Linear(hidden_dim, hidden_dim),
            }),
            "visual": nn.ModuleDict({
                "q": nn.Linear(hidden_dim, hidden_dim),
                "k": nn.Linear(hidden_dim, hidden_dim),
                "v": nn.Linear(hidden_dim, hidden_dim),
                "out": nn.Linear(hidden_dim, hidden_dim),
            })
        })

    def _self_attend(self, seq, mask, qkv):
        """
        seq: [B, L, D]
        mask: [B, L]
        """
        B, L, D = seq.shape
        Q, K, V = qkv["q"](seq), qkv["k"](seq), qkv["v"](seq)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)  # [B, L, L]
        attn = attn.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # [B, L, D]
        pref = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return qkv["out"](pref)  # [B, D]

    def forward(self, item_seq, text_seq, visual_seq, mask):
        item_pref = self._self_attend(item_seq, mask, self.qkv_out["item"])
        text_pref = self._self_attend(text_seq, mask, self.qkv_out["text"])
        visual_pref = self._self_attend(visual_seq, mask, self.qkv_out["visual"])
        return item_pref, text_pref, visual_pref

class PrefMapper(nn.Module):
    def __init__(self, dim, channel_num, t):
        super().__init__()
        self.l1 = nn.Linear(dim, int(dim/8))
        self.l2 = nn.Linear(int(dim/8), channel_num)
        self.t = t
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l2(F.relu(F.normalize(self.l1(x), p=2, dim=1)))/self.t
        output = torch.softmax(x, dim=1)
        return output

class MultiModalDenoiser(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_text2visual = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_visual2text = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # 归一化层
        self.norm_text = nn.LayerNorm(hidden_dim)
        self.norm_visual = nn.LayerNorm(hidden_dim)

        self.res_scale = nn.Parameter(torch.tensor(0.5))

        self.id_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, text_emb, visual_emb, id_emb, mask=None):
        """
        text_emb:   [B, L, D]
        visual_emb: [B, L, D]
        id_emb:     [B, L, D]
        mask:       [B, L] or None
        """

        text_refined, _ = self.cross_text2visual(
            text_emb, visual_emb, visual_emb,
            key_padding_mask=(~mask) if mask is not None else None
        )
        visual_refined, _ = self.cross_visual2text(
            visual_emb, text_emb, text_emb,
            key_padding_mask=(~mask) if mask is not None else None
        )

        gate = self.id_gate(id_emb)  
        gate_t = gate[..., 0:1]      # text 权重
        gate_v = gate[..., 1:2]      # visual 权重

        text_out = self.norm_text(text_emb + self.res_scale * (gate_t * text_refined))
        visual_out = self.norm_visual(visual_emb + self.res_scale * (gate_v * visual_refined))

        return text_out, visual_out


class ModalityProjector(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, use_res=True, use_norm=True):
        super().__init__()
        self.use_res = use_res
        self.use_norm = use_norm

        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )

        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()

        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()

        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

        self._init_weights()
    def _init_weights(self):
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)
            nn.init.constant_(self.residual_proj.bias, 0.0)

    def forward(self, x):
        out = self.proj(x)
        if self.use_res:
            out = self.alpha * out + self.residual_proj(x)
        out = self.norm(out)
        return out

class EncoderDecoderRetrievalModel(nn.Module):
    def __init__(
        self,
        text_dim: int = 512,
        visual_dim: int = 512,
        attn_dim: int = 64,
        dropout: float = 0.1,
        attn_heads: int = 8,
        encoder_layers: int = 2,
        max_pos: int = 2048,
        top_k: int = 20,
        args=None
    ):
        super().__init__()

        self.attn_dim = attn_dim
        self.text_dim = text_dim
        self.attn_heads = attn_heads
        self.encoder_layers = encoder_layers
        self.top_k = top_k
        self.layer_norm_eps = 1e-12
        self.hidden_size = args.hidden_size
        self.num_items = args.num_items
        self.temperature = args.temperature
        self.alpha = nn.Parameter(torch.tensor(0.6)) 

        text_feat = torch.load(args.text_feat_path)
        image_feat = torch.load(args.image_feat_path)
        pad_text = torch.zeros((1, text_feat.size(1)))
        pad_image = torch.zeros((1, image_feat.size(1)))
        text_feat = torch.cat([text_feat, pad_text], dim=0)
        image_feat = torch.cat([image_feat, pad_image], dim=0)
        self.item_text_embedding = nn.Embedding.from_pretrained(text_feat, freeze=True)
        self.item_visual_embedding = nn.Embedding.from_pretrained(image_feat, freeze=True)

        # -------------------- item_embedding / pos --------------------
        self.item_embedding = nn.Embedding(self.num_items + 1, attn_dim, padding_idx=self.num_items)
        self.position_embedding = nn.Embedding(max_pos, attn_dim)

        # -------------------- Projector --------------------
        self.text_proj_in = ModalityProjector(text_dim, attn_dim)
        self.visual_proj_in = ModalityProjector(visual_dim, attn_dim)

        # -------------------- layer norm & dropout --------------------
        self.item_ln = nn.LayerNorm(self.attn_dim, eps=self.layer_norm_eps)
        self.text_ln = nn.LayerNorm(self.attn_dim, eps=self.layer_norm_eps)
        self.visual_ln = nn.LayerNorm(self.attn_dim, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dropout = nn.Dropout(p=args.emb_dropout)
        
        # -------------------- Conditional Guide --------------------
        self.denoiser = MultiModalDenoiser(hidden_dim=self.attn_dim, dropout=dropout)
        self.user_pref = UserPrefSelfAttention(hidden_dim=self.attn_dim)
        self.router = PrefMapper(3 * self.attn_dim, 3, self.temperature)

        # -------------------- Transformer --------------------
        self.item_id_encoder = TransformerDecoder(
            d_in=attn_dim, d_out=attn_dim, dropout=dropout,
            num_heads=self.attn_heads, n_layers=self.encoder_layers//2,
            do_cross_attn=True
        )
        self.text_encoder = TransformerDecoder(
            d_in=attn_dim, d_out=attn_dim, dropout=dropout,
            num_heads=self.attn_heads, n_layers=self.encoder_layers//2,
            do_cross_attn=True
        )
        self.visual_encoder = TransformerDecoder(
            d_in=attn_dim, d_out=attn_dim, dropout=dropout,
            num_heads=self.attn_heads, n_layers=self.encoder_layers//2,
            do_cross_attn=True
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch: SeqBatch, device=None, args=None, epoch=None):
        B, L = batch.hist.shape
        valid_mask = batch.seq_mask.to(device)
        seq_length = valid_mask.sum(dim=1) - 1  # [B]
        hist_ids = batch.hist.to(device)

        item_seq_emb = self.item_embedding(hist_ids) # [B, 50, 256]
        text_seq_emb = self.text_proj_in(self.item_text_embedding(hist_ids)) # [B, 50, 256]
        visual_seq_emb = self.visual_proj_in(self.item_visual_embedding(hist_ids)) # [B, 50, 256]

        text_seq_emb_de, visual_seq_emb_de = self.denoiser(text_seq_emb, visual_seq_emb, item_seq_emb, mask=valid_mask)

        pos_ids = torch.arange(L, device=device)  # [0, 1, 2, ..., L-1]
        pos_emb = self.position_embedding(pos_ids)  # [L, attn_dim]
        pos_emb = pos_emb.unsqueeze(0).repeat(B, 1, 1)  # [B, L, attn_dim]
        item_seq_emb = item_seq_emb + pos_emb # [B, 50, 256]
        text_seq_emb_de = text_seq_emb_de + pos_emb # [B, 50, 256]
        visual_seq_emb_de = visual_seq_emb_de + pos_emb # [B, 50, 256]

        item_seq_emb = self.item_ln(self.embed_dropout(item_seq_emb))
        text_seq_emb_de = self.text_ln(self.embed_dropout(text_seq_emb_de))
        visual_seq_emb_de = self.visual_ln(self.embed_dropout(visual_seq_emb_de))

        item_seq_emb_en = self.item_id_encoder(
            item_seq_emb, padding_mask=valid_mask, is_causal=True,
            context=None, context_mask=None
        )
        text_seq_emb_en = self.text_encoder(
            text_seq_emb_de, padding_mask=valid_mask, is_causal=True,
            context=None, context_mask=None
        )
        visual_seq_emb_en = self.visual_encoder(
            visual_seq_emb_de, padding_mask=valid_mask, is_causal=True,
            context=None, context_mask=None
        )

        if args.num_items >= 23000:
            text_seq_emb_de2, visual_seq_emb_de2 = self.denoiser(
                text_seq_emb_en, visual_seq_emb_en, item_seq_emb_en, mask=valid_mask
            )
        else:
            text_seq_emb_de2 = text_seq_emb_en
            visual_seq_emb_de2 = visual_seq_emb_en

        item_next_emb = gather_indexes(item_seq_emb_en, seq_length)
        text_next_emb = gather_indexes(text_seq_emb_de2, seq_length)
        visual_next_emb = gather_indexes(visual_seq_emb_de2, seq_length)

        item_user_pref, text_user_pref, visual_user_pref = self.user_pref(
            item_seq_emb_en, text_seq_emb_de2, visual_seq_emb_de2, valid_mask
        )

        m_w = self.router(torch.concat([item_user_pref, text_user_pref, visual_user_pref], dim=-1))

        full_item_emb = self.item_embedding.weight
        full_text_emb = self.text_proj_in(self.item_text_embedding.weight)
        full_visual_emb = self.visual_proj_in(self.item_visual_embedding.weight)
        item_scores = torch.matmul(item_next_emb, full_item_emb.t())
        text_scores = torch.matmul(text_next_emb, full_text_emb.t())
        visual_scores = torch.matmul(visual_next_emb, full_visual_emb.t())
        m_w_item, m_w_text, m_w_visual = torch.split(m_w, 1, dim=-1)  # [B, 1], [B, 1], [B, 1]
        scores = m_w_item * item_scores + m_w_text * text_scores + m_w_visual * visual_scores

        scores_list = [item_scores, text_scores, visual_scores, scores]
        topk_scores, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)
        return [scores_list, topk_idx, m_w]
    
    def calculate_loss(self, batch: SeqBatch, device=None, args=None, epoch=None):
        losses = {}
        target = batch.target.to(device)
        seq_out = self.forward(batch, device, args, epoch=epoch)
        if self.training:
            losses['icla'] = self.ICLA(seq_out[0], target)
            losses['simw'] = self.SIMW(seq_out[0], target, seq_out[2], args.mu)
            losses['balw'] = self.BALW(seq_out[2])
            losses['moct'] = self.MOCT(seq_out[0][0], seq_out[0][1], seq_out[0][2], args.moct_t)
            losses['total_loss'] = args.w_icla * losses.get('icla', 0.0) \
                            + args.w_simw * losses.get('simw', 0.0) \
                            + args.w_moct * losses.get('moct', 0.0) \
                            + args.w_balw * losses.get('balw', 0.0)
        return ModelOutput(loss=losses, topk_idx=seq_out[1])

    def BALW(self, w_m):
        w_m = torch.clamp(w_m, min=1e-9)
        N = w_m.size(1)
        entropy = N * torch.sum(w_m * torch.log(w_m), dim=1)
        return torch.mean(entropy)

    def ICLA(self, scores, target):
        loss_fused = self.criterion(scores[3], target)
        return loss_fused

    def eva_imp(self, score, target, mu):
        target_logits = score.gather(1, target.unsqueeze(1)).squeeze(1)
        rank = (score > target_logits.unsqueeze(1)).sum(dim=1) + 1  # [B]
        uncertainty = (rank - 1) / (score.size(1)) 
        return torch.exp(mu * (uncertainty.clamp(min=1e-8)))

    def SIMW(self, scores, target, w, mu):
        item_dist = self.eva_imp(scores[0], target, mu)
        text_dist = self.eva_imp(scores[1], target, mu)
        visual_dist = self.eva_imp(scores[2], target, mu)
        
        dists = torch.stack([item_dist, text_dist, visual_dist], dim=1)
        inv_dists = torch.exp(-dists)
        ideal_weights = inv_dists / inv_dists.sum(dim=1, keepdim=True)
        
        loss_sim = F.kl_div(
            w.log_softmax(dim=-1), 
            ideal_weights.softmax(dim=-1).detach(),
            reduction='batchmean'
        )
        return loss_sim
    
    def MOCT(self, item_scores, text_scores, visual_scores, T=5.0):
        item_log_p = F.log_softmax(item_scores / T, dim=-1)
        text_log_p = F.log_softmax(text_scores / T, dim=-1)
        visual_log_p = F.log_softmax(visual_scores / T, dim=-1)

        item_p = item_log_p.exp()
        text_p = text_log_p.exp()
        visual_p = visual_log_p.exp()

        def sym_kl(log_p1, p2, log_p2, p1):
            return 0.5 * (
                F.kl_div(log_p1, p2.detach(), reduction='batchmean') +
                F.kl_div(log_p2, p1.detach(), reduction='batchmean')
            )

        loss_item_text = sym_kl(item_log_p, text_p, text_log_p, item_p)
        loss_item_visual = sym_kl(item_log_p, visual_p, visual_log_p, item_p)
        loss_text_visual = sym_kl(text_log_p, visual_p, visual_log_p, text_p)

        moct_loss = (loss_item_text + loss_item_visual + loss_text_visual) / 3
        moct_loss = moct_loss * (T ** 2)

        return moct_loss


    
