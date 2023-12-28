import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask=None, dropout=None):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        if dropout is not None:
            attn = dropout(attn)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.W_Q = nn.Parameter(torch.Tensor(n_heads, d_model, self.d_k))
        self.W_K = nn.Parameter(torch.Tensor(n_heads, d_model, self.d_k))
        self.W_V = nn.Parameter(torch.Tensor(n_heads, d_model, self.d_k))

        self.fc = nn.Linear(self.d_v * n_heads, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_uniform_(self.W_Q, gain=1.414)
        nn.init.xavier_uniform_(self.W_K, gain=1.414)
        nn.init.xavier_uniform_(self.W_V, gain=1.414)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        Q = torch.matmul(input_Q.unsqueeze(1), self.W_Q)  # Q: [batch_size, n_heads, len_q, d_k]
        K = torch.matmul(input_K.unsqueeze(1), self.W_K)  # K: [batch_size, n_heads, len_k, d_k]
        V = torch.matmul(input_V.unsqueeze(1), self.W_V)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask=attn_mask, dropout=self.dropout)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_k)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff=1024):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model)

    def forward(self, enc_inputs, enc_self_attn_mask=None):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, attn_mask=enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, max_len=20, if_emb=True, if_pos=True):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.if_emb = if_emb
        self.if_pos = if_pos
        if self.if_emb:
            self.src_emb = nn.Linear(input_dim, d_model)
            self.d_model = d_model
        else:
            self.d_model = input_dim
        if self.if_pos:
            self.pos_emb = PositionalEncoding(self.d_model, max_len=max_len)
        # print(self.d_model)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, n_heads) for _ in range(n_layers)])

    def forward(self, inputs, mask=None):
        '''
        enc_inputs: [batch_size, src_len, d]
        '''
        enc_outputs = inputs
        if self.if_emb:
            enc_outputs = self.src_emb(enc_outputs)  # [batch_size, src_len, d_model]
        if self.if_pos:
            enc_outputs = self.pos_emb(enc_outputs)  # [batch_size, src_len, d_model]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask=mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(DecoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model)

    def forward(self, dec_inputs, enc_outputs, enc_self_attn_mask=None, enc_dec_attn_mask=None):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        dec_outputs, attn1 = self.enc_self_attn(dec_inputs, dec_inputs, dec_inputs, attn_mask=enc_self_attn_mask)  # enc_inputs to same Q,K,V
        dec_outputs, attn2 = self.enc_dec_attn(dec_outputs, enc_outputs, enc_outputs, attn_mask=enc_dec_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return dec_outputs, attn1, attn2


class Decoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, max_len=20, if_emb=True, if_pos=True):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.if_emb = if_emb
        self.if_pos = if_pos
        if self.if_emb:
            self.src_emb = nn.Linear(input_dim, d_model)
            self.d_model = d_model
        else:
            self.d_model = input_dim
        if self.if_pos:
            self.pos_emb = PositionalEncoding(self.d_model, max_len=max_len)
        self.layers = nn.ModuleList([DecoderLayer(self.d_model, n_heads) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs, mask=None):
        '''
        enc_inputs: [batch_size, src_len, d]
        '''
        dec_outputs = dec_inputs
        if self.if_emb:
            dec_outputs = self.src_emb(dec_outputs)  # [batch_size, src_len, d_model]
        if self.if_pos:
            dec_outputs = self.pos_emb(dec_outputs)  # [batch_size, src_len, d_model]
        dec_self_attns = []
        enc_dec_attns = []
        for layer in self.layers:
            dec_outputs, enc_self_attn, enc_dec_attn = layer(dec_outputs, enc_outputs, enc_self_attn_mask=mask)
            dec_self_attns.append(enc_self_attn)
            enc_dec_attns.append(enc_dec_attn)

        return dec_outputs
