"""This module contains the multi-head attention module for the transformer model."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.utils.data import DataLoader, Dataset


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: The dimension of the embeddings.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Define linear layers for query, key, and value
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        # Output linear layer
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            query: Tensor of shape (batch_size, seq_len, d_model)
            key: Tensor of shape (batch_size, seq_len, d_model)
            value: Tensor of shape (batch_size, seq_len, d_model)
            mask: Optional Tensor of shape (batch_size, 1, 1, seq_len) or similar
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
            Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = (
            self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        )  # (batch_size, num_heads, seq_len, d_k)
        K = (
            self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        )  # (batch_size, num_heads, seq_len, d_k)
        V = (
            self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        )  # (batch_size, num_heads, seq_len, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )  # (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.softmax(scores)  # (batch_size, num_heads, seq_len, seq_len)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V)  # (batch_size, num_heads, seq_len, d_k)

        # Concatenate heads
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )  # (batch_size, seq_len, d_model)

        # Final linear layer
        output = self.linear_out(output)  # (batch_size, seq_len, d_model)

        return output
