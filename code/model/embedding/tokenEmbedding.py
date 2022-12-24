import torch.nn as nn
import math

class TokenEmbedding(nn.Module):

    def __init__(self, d_embed, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Linear(vocab_size, d_embed, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.d_embed = d_embed

    def forward(self, x):
        out = self.sigmoid(self.embedding(x)) * math.sqrt(self.d_embed)
        return out