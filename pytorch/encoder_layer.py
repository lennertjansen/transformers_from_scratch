"""This module implements the EncoderLayer class."""

import torch
import torch.nn as nn
from pytorch.multi_head_attention import MultiHeadAttention
from pytorch.feed_forward_network import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x, mask)

        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))

        return x
