import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super(PositionalEncoding, self).__init__()
        # initialize positional encoding matrix
        self.encoding = torch.zeros(max_len, embed_size)
        self.encoding.requires_grad = False # ensure the positional encoding matrix is not trainable

        # create a tensor of shape (max_len, 1) containing position indices from 0 to max_len-1
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(torch.log(torch.tensor(10000.0)) / embed_size))

        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)
        # register the positional encoding matrix as a buffer, so it is not considered a model parameter
        self.register_buffer('pos_encoding', self.encoding)

    def forward(self, x):
        # get the batch size and sequence length from the input tensor
        N, seq_len = x.size(0), x.size(1)
        return x + self.pos_encoding[:seq_len, :].unsqueeze(0).repeat(N, 1, 1)
