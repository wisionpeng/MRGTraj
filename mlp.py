import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x


class SocialLatentGenerator(nn.Module):  # KL(p(z|global social context), N(0, 1))
    def __init__(self, dim_in=4, d_model=256, dim_z=16, dff=1024, dropout=0.3):
        super(SocialLatentGenerator, self).__init__()
        self.dim_z = dim_z
        self.d_model = d_model
        self.emb_layer = nn.Sequential(
            nn.Linear(dim_in, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.latent = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(dff, dff),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(dff, dim_z * 2),
            nn.ReLU6()
        )

    def forward(self, input, batch_mask):
        # input: [batch_size, pred_len, 4]
        seq_len, batch_size = input.shape[1], input.shape[0]
        input_embedding = self.emb_layer(input)
        input_embedding_n = input_embedding.transpose(0, 1).unsqueeze(1).repeat(1, batch_size, 1, 1)
        input_embedding_n = input_embedding_n * batch_mask.unsqueeze(2).unsqueeze(0).repeat(
            seq_len, 1, 1, self.d_model)
        input_embeddings = input_embedding_n.max(2)[0].mean(0)
        latent_variables = self.latent(input_embeddings)
        mu = latent_variables[:, :self.dim_z]
        log_var = latent_variables[:, self.dim_z:]
        var = log_var.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        eps = eps.cuda()
        z = eps.mul(var).add_(mu)
        return z, mu, log_var

    def reparameters(self, batch_size):

        z = get_noise((1, self.dim_z), "gaussian")
        z = z.repeat(batch_size, 1)

        return z

