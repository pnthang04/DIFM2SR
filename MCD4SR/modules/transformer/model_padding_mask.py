from modules.encoder import MLP
from modules.normalize import RMSNorm
from modules.transformer.attention_padding_mask import AttentionInput
from modules.transformer.attention_padding_mask import MultiHeadAttention
from typing import Optional, List
from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool,
        mlp_hidden_dims: list = [1024],
        do_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.do_cross_attn = do_cross_attn

        self.attention = MultiHeadAttention(
            d_in=d_in, 
            d_out=d_out, 
            num_heads=num_heads, 
            cross_attn=False, 
            dropout=dropout, 
            qkv_bias=qkv_bias
        )

        self.ff = nn.Sequential(
            RMSNorm(d_out),
            MLP(
                input_dim=d_out,
                hidden_dims=mlp_hidden_dims,
                out_dim=d_out,
                dropout=dropout,
                normalize=False
            ),
            nn.Dropout(dropout)
        )

        self.attn_norm = RMSNorm(d_out)
        self.ffn_norm = RMSNorm(d_out)
        self.do = nn.Dropout(dropout)

        if self.do_cross_attn:
            self.cross_attention = MultiHeadAttention(
                d_in=d_out, 
                d_out=d_out, 
                num_heads=num_heads, 
                cross_attn=True, 
                dropout=dropout, 
                qkv_bias=qkv_bias
            )
            self.cross_attn_norm = RMSNorm(d_out)

    def forward(
        self,
        x: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # self-attention (query和key/value都是x)
        attn_out = x + self.attention(
            self.do(self.attn_norm(x)),
            x_kv=None,
            padding_mask=padding_mask,
            kv_padding_mask=padding_mask,  # 屏蔽 decoder padding
            is_causal=is_causal
        )

        # cross-attention
        if self.do_cross_attn and x_kv is not None:
            attn_out = attn_out + self.cross_attention(
                x=self.do(self.cross_attn_norm(x)),
                x_kv=x_kv,
                padding_mask=padding_mask,      # decoder query mask
                kv_padding_mask=context_mask,   # encoder key/value mask
                is_causal=False
            )

        proj_out = attn_out + self.ff(attn_out)
        return proj_out

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        n_layers: int,
        do_cross_attn: bool = False
    ) -> None:
        super().__init__()
        self.do_cross_attn = do_cross_attn
        self.layers = nn.ModuleList([
                TransformerBlock(
                    d_in=d_in,
                    d_out=d_out,
                    dropout=dropout,
                    num_heads=num_heads,
                    qkv_bias=False,
                    do_cross_attn=self.do_cross_attn,
                ) for _ in range(n_layers)
            ])

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        context_list: Optional[List[torch.Tensor]] = None,
        context_mask_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            context_list: List of context tensors for alternating KV [ctx1, ctx2, ...]
            context_mask_list: Optional masks for each context
        """
        for i, layer in enumerate(self.layers):
            if context_list is not None and len(context_list) > 0:
                current_ctx = context_list[i % len(context_list)]
                current_mask = (
                    context_mask_list[i % len(context_mask_list)] 
                    if context_mask_list is not None 
                    else None
                )
            else:
                current_ctx = context
                current_mask = context_mask
            
            x = layer(
                x=x,
                x_kv=current_ctx,
                padding_mask=padding_mask,
                is_causal=is_causal,
                context_mask=current_mask
            )
        
        return x