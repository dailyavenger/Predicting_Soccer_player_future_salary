import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_embed, d_model, h, dr_rate = 0):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = nn.Linear(d_embed, d_model)
        self.k_fc = nn.Linear(d_embed, d_model)
        self.v_fc = nn.Linear(d_embed, d_model)
        self.out_fc = nn.Linear(d_model, d_embed)
        self.dropout = nn.Dropout(p = dr_rate)

    def calculate_attention(self, query, key, value, mask):
        # Dimension for Q, K, V : (n_batch, h, seq_len, d_k)
        # Dimesion for mask : (n_batch, seq_len, seq_len)
        n_k = key.shape[-1]

        # QK^T : (n_batch, h, seq_len, d_k) * (n_batch, h, d_k, seq_len) = (n_batch, h, seq_len, seq_len)
        attention_score= torch.matmul(query, key.transpose(-2, -1)) 
        attention_score = attention_score / math.sqrt(n_k)

        # Check wheter masking or not
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        # (n_batch, h, seq_len, seq_len) 즉 가로 단위로 softmax처리를 한 것
        attention_prob = F.softmax(attention_score, dim = -1)
        attention_prob = self.dropout(attention_prob)
     
        # (n_batch, h, seq_len, seq_len) * (n_batch, h, seq_len, d_k) = (n_batch, h, seq_len, d_k)
        out = torch.matmul(attention_prob, value)

        return out

    def forward(self, *args, query, key, value, mask = None):
        n_batch = query.size(0)

        # 한번에 FC layer를 처리
        def transform(x, fc):
            out = fc(x) # (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model // self.h) # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out

        # d_model = h * d_k로 찢음
        query = transform(query, self.q_fc) # (n_batch, h, seq_len, d_k)
        key = transform(key, self.k_fc) # (n_batch, h, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, h, seq_len, d_k)

        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model
        out = self.out_fc(out)
        return out