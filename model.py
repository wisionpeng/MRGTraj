from transformer import Encoder
from mlp import *


def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class MRGTraj(nn.Module):
    def __init__(self, args):
        super(MRGTraj, self).__init__()
        d_model, n_layers, n_heads, noise_dim, obs_len, pred_len \
            = args.d_model, args.n_layers, args.n_heads, args.noise_dim, args.obs_len, args.pred_len
        self.d_model = d_model
        self.noise_dim = noise_dim
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.past_encoder = Encoder(2, d_model, n_layers, n_heads)

        self.social_latent_generator = SocialLatentGenerator(dim_in=4, d_model=d_model, dim_z=noise_dim)

        self.temporal_mapper = MLP(obs_len, pred_len, hidden_size=(512, 1024, 512))

        self.social_refiner = Encoder(d_model + noise_dim, d_model, n_layers, n_heads, if_emb=False, if_pos=False)
        self.prediction_layer = nn.Linear(d_model + noise_dim, 2)

    def forward(self, past_traj_rel, future_traj, batch_mask=None):
        '''
        past_traj_rel: [batch_size, src_len, 2]
        future_traj: [batch_size, tgt_len, 4]
        '''

        past_encoding = self.past_encoder(past_traj_rel[..., 2:], mask=None)
        future_decoding = self.temporal_mapper(past_encoding.transpose(-1, -2)).transpose(-1, -2).transpose(0, 1)
        z, mu, log_var = self.social_latent_generator(future_traj, batch_mask)
        future_decoding = torch.cat((future_decoding, z.unsqueeze(0).repeat(self.pred_len, 1, 1)), dim=-1)
        batch_mask = torch.gt(batch_mask, 0).unsqueeze(0)
        future_decoding = self.social_refiner(future_decoding, batch_mask)

        pred_future_traj = self.prediction_layer(future_decoding)

        return pred_future_traj, mu, log_var

    def inference(self, past_traj_rel, batch_mask=None):
        '''
        enc_inputs: [batch_size, src_len, 2]
        dec_inputs: [batch_size, tgt_len, 2]
        '''
        batch_mask = torch.gt(batch_mask, 0).unsqueeze(0)

        past_encoding = self.past_encoder(past_traj_rel[..., 2:], mask=None)
        future_decoding = self.temporal_mapper(past_encoding.transpose(-1, -2)).transpose(-1, -2).transpose(0, 1)
        z = self.social_latent_generator.reparameters(past_traj_rel.shape[0])
        future_decoding = torch.cat((future_decoding, z.unsqueeze(0).repeat(self.pred_len, 1, 1)), dim=-1)
        future_decoding = self.social_refiner(future_decoding, batch_mask)
        pred_future_traj = self.prediction_layer(future_decoding)
        return pred_future_traj


