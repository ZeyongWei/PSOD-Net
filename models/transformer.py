import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pointnet2_utils import sample_and_group, index_points, farthest_point_sample, query_ball_point, knn_point

class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=64, dim=128, temperature=10000):
        super().__init__()
        self.token_projection = nn.Linear(hidden_dim * 3, dim)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim

    def forward(self, pos_embed, max_len=(1, 1, 1)):
        z_embed, y_embed, x_embed = pos_embed.chunk(3, 1)
        z_max, y_max, x_max = max_len

        eps = 1e-6
        z_embed = z_embed / (z_max + eps) * self.scale
        y_embed = y_embed / (y_max + eps) * self.scale
        x_embed = x_embed / (x_max + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=pos_embed.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t
        pos_z = z_embed / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(),
                             pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(),
                             pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(),
                             pos_z[:, 1::2].cos()), dim=2).flatten(1)

        pos = torch.cat((pos_z, pos_y, pos_x), dim=1)

        pos = self.token_projection(pos)
        return pos

class LinformerEncoderLayer(nn.Module):

    def __init__(self, src_len, ratio, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear_k = nn.Parameter(torch.empty(src_len // ratio, src_len))
        self.linear_v = nn.Parameter(torch.empty(src_len // ratio, src_len))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)

        self.activation = nn.ReLU(inplace=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_k)
        nn.init.xavier_uniform_(self.linear_v)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, is_causal = False, src_key_padding_mask=None):   ######################## is_causal = False
        src_temp = src.transpose(0, 1)
        #print(src_temp.shape,self.linear_k.shape, self.linear_v.shape)
        key = torch.matmul(self.linear_k, src_temp).transpose(0, 1)
        value = torch.matmul(self.linear_v, src_temp).transpose(0, 1)
        src2 = self.self_attn(src, key, value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class TransformerEncoderLayerPreNorm(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)

        self.activation = nn.ReLU(inplace=False)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, is_causal=  False, src_key_padding_mask=None):   ###################### is_causal=  False
        src = self.norm1(src)
        src2, mask = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        return src


class TransformerDecoderLayerPreNorm(nn.Module):

    def __init__(self, d_model, nc_mem, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_mem = nn.LayerNorm(nc_mem)

        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)

        self.activation = nn.ReLU(inplace=False)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = self.norm1(tgt)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm2(tgt)
        memory = self.norm_mem(memory)
        tgt2, mask = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt


class LocalTransformer(nn.Module):

    def __init__(self, npoint,  nsample, dim_feature, dim_out, nhead=4, num_layers=2, norm_cfg=dict(type='BN2d'),
                 ratio=1, drop=0.0, prenorm=True):
        super().__init__()

        self.npoint = npoint
        self.nsample = nsample
        self.nc_in = dim_feature
        self.nc_out = dim_out

        # self.sampler = Points_Sampler([self.npoint], ['D-FPS'])
        # self.grouper = QueryAndGroup(self.radius, self.nsample, use_xyz=False, return_grouped_xyz=True, normalize_xyz=False)

        self.pe = nn.Sequential(
            nn.Conv2d(3, self.nc_in // 2, 1),
            nn.BatchNorm2d(self.nc_in // 2),
            nn.ReLU(),
            nn.Conv2d(self.nc_in // 2, self.nc_in, 1)
        )

        # self.pe = PositionalEncodingFourier(int(self.nc_in // 2), self.nc_in)

        # BSC_Encoder = TransformerEncoderLayerPreNorm if prenorm else nn.TransformerEncoderLayer

        self.chunk = nn.TransformerEncoder(
            TransformerEncoderLayerPreNorm(d_model=self.nc_in, dim_feedforward=2 * self.nc_in, dropout=drop, nhead=nhead) if ratio == 1 else
            LinformerEncoderLayer(src_len=nsample, ratio=ratio, d_model=self.nc_in, nhead=nhead, dropout=drop, dim_feedforward=2 * self.nc_in),
            num_layers=num_layers)

        self.fc = nn.Conv2d(self.nc_in, self.nc_out, 1)

    def forward(self, xyz, features):
        xyz_flipped = xyz.transpose(1, 2).contiguous()  # B, N, 3
        fps_idx = farthest_point_sample(xyz_flipped, self.npoint)
        new_xyz = index_points(xyz_flipped, fps_idx)  # B, npoint, 3
        # group_features, group_xyz = self.grouper(xyz.contiguous(), new_xyz.contiguous(),
        #                                          features.contiguous())  # (B, 3, npoint, nsample) (B, C, npoint, nsample)
        #group_idx = query_ball_point(self.radius, self.nsample, xyz_flipped, new_xyz)
        group_idx = knn_point(self.nsample, xyz_flipped, new_xyz)
        grouped_xyz = index_points(xyz_flipped, group_idx).permute(0, 3, 1, 2).contiguous()
        grouped_features = index_points(features.transpose(1, 2).contiguous(), group_idx).permute(0, 3, 1, 2).contiguous()

        # Fourier Position Embedding
        # b, _, n, s = grouped_xyz.size()
        # grouped_xyz = grouped_xyz.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        position_encoding = self.pe(grouped_xyz)
        # position_encoding = position_encoding.view(b, n, s, -1).permute(0, 3, 1, 2).contiguous()

        input_features = grouped_features + position_encoding
        B, D, np, ns = input_features.shape

        input_features = input_features.permute(0, 2, 1, 3).reshape(-1, D, ns).permute(2, 0, 1)  # (ns, B*np, D)
        transformed_feats = self.chunk(input_features).permute(1, 2, 0).reshape(B, np, D, ns).transpose(1, 2)
        output_features = F.max_pool2d(transformed_feats, kernel_size=[1, ns])  # (B, C, npoint)
        output_features = self.fc(output_features).squeeze(-1)

        return new_xyz.transpose(1, 2).contiguous(), output_features


class GlobalTransformer(nn.Module):

    def __init__(self, dim_feature, dim_out, nhead=4, num_layers=2, norm_cfg=dict(type='BN2d'), ratio=1, src_pts=2048,
                 drop=0.0, prenorm=True):
        super().__init__()

        self.nc_in = dim_feature
        self.nc_out = dim_out
        self.nhead = nhead

        self.pe = nn.Sequential(
            nn.Conv2d(3, self.nc_in // 2, 1),
            nn.BatchNorm2d(self.nc_in // 2),
            nn.ReLU(),
            nn.Conv2d(self.nc_in // 2, self.nc_in, 1)
        )

        # BSC_Encoder = TransformerEncoderLayerPreNorm if prenorm else nn.TransformerEncoderLayer

        self.chunk = nn.TransformerEncoder(
            TransformerEncoderLayerPreNorm(d_model=self.nc_in, dim_feedforward=2 * self.nc_in, dropout=drop, nhead=nhead) if ratio == 1 else
            LinformerEncoderLayer(src_len=src_pts, ratio=ratio, d_model=self.nc_in, nhead=nhead, dropout=drop, dim_feedforward=2 * self.nc_in),
            num_layers=num_layers)

        self.fc = nn.Conv2d(self.nc_in, self.nc_out, 1)

    def forward(self, xyz, features):
        xyz_flipped = xyz.unsqueeze(-1)
        input_features = features.unsqueeze(-1) + self.pe(xyz_flipped)
        input_features = input_features.squeeze(-1).permute(2, 0, 1)
        transformed_feats = self.chunk(input_features).permute(1, 2, 0)
        output_features = self.fc(transformed_feats.unsqueeze(-1)).squeeze(-1)

        return output_features
