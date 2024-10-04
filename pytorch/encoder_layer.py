"""This module implements the EncoderLayer class."""

from typing import Optional
import torch
import torch.nn as nn
from pytorch.multi_head_attention import MultiHeadAttention
from pytorch.feed_forward_network import FeedForward


class EncoderLayer(nn.Module):
    """Encoder layer for the transformer model."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        """Initialize encoder layer.

        Args:
            d_model (int): The dimension of the embeddings.
            num_heads (int): The number of attention heads.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the encoder layer.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Tensor of shape (batch_size, seq_len).

        Returns:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, d_model).
        """
        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x, mask)

        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))

        return x
