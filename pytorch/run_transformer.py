"""Instantiates the Transformer model and performs a forward pass with dummy input data."""

import torch

from pytorch.transformer import Transformer


def main() -> None:

    # Hyperparameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout = 0.1
    max_len = 100

    # Instantiate the model
    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout,
        max_len,
    )

    # Dummy input data
    batch_size = 2
    src_seq_length = 10
    tgt_seq_length = 10

    # Random integer inputs representing word indices
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_length))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_length))

    # Forward pass
    output = model(src, tgt)

    print("Output shape:", output.shape)
    # Output shape should be (batch_size, tgt_seq_length, tgt_vocab_size)


if __name__ == "__main__":
    main()
