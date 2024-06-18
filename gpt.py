import torch.nn as nn
from positional_encoding import PositionalEncoding
from transformer_decoder_layer import TransformerDecoderLayer

class GPT(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        forward_expansion,
        num_layers,
        vocab_size,
        max_len,
        dropout,
    ):
        super(GPT, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_len)
        # stack of Transformer decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        
        # fully connected output layer that projects the decoder output to the vocabulary size
        self.fc_out = nn.Linear(embed_size, vocab_size)
        # dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        x = self.word_embedding(x)
        x = self.position_embedding(x)
        for layer in self.layers:
            x = layer(x, x, x, mask)
        out = self.fc_out(x)
        
        return out
