import math
import torch
import torch.nn as nn

from pytorch.positional_encoding import PositionalEncoding
from pytorch.encoder_layer import EncoderLayer
from pytorch.decoder_layer import DecoderLayer


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Encoder and Decoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dropout) for _ in range(num_decoder_layers)]
        )

        # Final linear layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def make_src_mask(self, src):
        # src shape: (batch_size, src_len)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # (batch_size, 1, 1, src_len)
        return src_mask

    def make_tgt_mask(self, tgt):
        # tgt shape: (batch_size, tgt_len)
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        return tgt_mask

    def forward(self, src, tgt):
        # Masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # Embeddings
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        # Encoder
        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        # Decoder
        dec_output = tgt
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)

        # Output
        output = self.fc_out(dec_output)
        return output
