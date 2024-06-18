import torch.nn as nn
from self_attention import SelfAttention
from feed_forward import FeedForward

class TransformerDecoderLayer(nn.Module):\
    # initializes the components of the transformer decoder layer
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    # applies self-attention to the i/o tensor 
    def forward(self, x, value, key, mask):
        attention = self.attention(x, value, key, mask)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
