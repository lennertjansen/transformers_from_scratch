"""This module implements the DecoderLayer class."""

import torch.nn as nn
from pytorch.multi_head_attention import MultiHeadAttention
from pytorch.feed_forward_network import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.attention1 = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.attention2 = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn1 = self.attention1(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        # Encoder-decoder attention
        attn2 = self.attention2(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn2))

        # Feed-forward
        ff = self.ff(x)
        x = self.norm3(x + self.dropout(ff))

        return x
