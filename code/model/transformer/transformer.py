import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator, window_size):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.window_size = window_size

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)

    def forward(self, x):

        src = x[:,0:17*self.window_size]
        src = src.view([-1, self.window_size, 17])
        tgt = x[:,17*self.window_size:]
        tgt = tgt.view([-1, self.window_size, 1])
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out)
        out = out.squeeze()
        # out = F.log_softmax(out, dim = -1)
        return out

    # Masking
    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask


    def make_tgt_mask(self, tgt):
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        return pad_mask & seq_mask


    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask


    def make_pad_mask(self, query, key, pad_idx=-1):
        # # query: (n_batch, query_seq_len)
        # # key: (n_batch, key_seq_len)
        # query_seq_len, key_seq_len = query.size(1), key.size(1)

        # # pad_idx랑 같으면 False => mask
        # # 우리 데이터의 경우 padding 필요 없으니까 죄다 True (1) 로!!
        # key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
        # key_mask = key_mask.repeat(1, 1, query_seq_len, 1)    # (n_batch, 1, query_seq_len, key_seq_len)

        # query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
        # query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

        # mask = key_mask & query_mask    # (n_batch, 1, query_seq_len, key_seq_len)
        # mask.requires_grad = False

        is_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if is_cuda else 'cpu')
        mask = torch.ones(size=(query.size(0), 1, self.window_size, self.window_size), dtype=bool).to(device)
        return mask


    def make_subsequent_mask(self, query, key):
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8') # lower triangle without diagonal
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
        return mask