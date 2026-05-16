import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from torch.nested import Tensor as NestedTensor
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, List, Union

torch.backends.cuda.enable_flash_sdp(True)

AttentionInput = Union[Tensor, NestedTensor]

class Attend(nn.Module):
    def __init__(self, d_out, num_heads, head_dim, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_out = d_out
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attn_mask: Optional[torch.Tensor], is_causal: bool = False):
        # queries, keys, values: [B, num_heads, L, head_dim]
        dropout_p = 0. if not self.training else self.dropout.p
        context_vec = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )
        return context_vec


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        num_heads,
        dropout=0.0,
        qkv_bias=False,
        cross_attn=False
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.cross_attn = cross_attn
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        if self.cross_attn:
            self.q = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.kv = nn.Linear(d_in, 2 * d_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)

        self.proj = nn.Linear(d_out, d_out, bias=False)
        self.attend = Attend(d_out, num_heads, self.head_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        kv_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        x: [B, L_q, d_in]
        x_kv: [B, L_kv, d_in] (cross-attn)
        padding_mask: [B, L_q] self-attention queries mask
        kv_padding_mask: [B, L_kv] keys/values mask (cross-attn)
        """
        B, L_q, _ = x.shape
        L_kv = x_kv.shape[1] if x_kv is not None else L_q

        # linear projections
        if self.cross_attn:
            assert x_kv is not None
            queries = self.q(x)
            keys, values = self.kv(x_kv).chunk(2, dim=-1)
        else:
            queries, keys, values = self.qkv(x).chunk(3, dim=-1)

        # reshape -> [B, num_heads, L, head_dim]
        queries = queries.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # construct attn_mask -> [B, num_heads, L_q, L_kv]
        attn_mask = None
        if padding_mask is not None or kv_padding_mask is not None:
            if padding_mask is None:
                padding_mask = torch.ones((B, L_q), dtype=torch.bool, device=x.device)
            if kv_padding_mask is None:
                kv_padding_mask = torch.ones((B, L_kv), dtype=torch.bool, device=x.device)

            # [B, 1, L_q, 1] & [B, 1, 1, L_kv] -> broadcast to [B, num_heads, L_q, L_kv]
            attn_mask = padding_mask[:, None, :, None] & kv_padding_mask[:, None, None, :]
            attn_mask = attn_mask.masked_fill(~attn_mask, float('-inf'))

        # scaled dot-product attention
        context_vec = self.attend(queries, keys, values, attn_mask=attn_mask, is_causal=is_causal)

        # reshape back -> [B, L_q, d_out]
        context_vec = context_vec.transpose(1, 2).contiguous().view(B, L_q, self.d_out)
        return self.proj(context_vec)
