"""This module contains the point-wise feed-forward network module for the transformer model."""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Feed-forward network for the transformer model."""

    def __init__(self, d_model, d_ff=2048, dropout=0.1) -> None:
        """Initialize feed-forward network.

        Args:
            d_model: The dimension of the embeddings.
            d_ff: The hidden layer size.
            dropout: Dropout probability.
        """
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the feed.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, d_model).

        Returns:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
