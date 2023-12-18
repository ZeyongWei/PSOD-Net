import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from pointnet2_utils import index_points, PointNetFeaturePropagation
from transformer import LocalTransformer, GlobalTransformer

class get_model(nn.Module):
    def __init__(self, num_classes=2, nsample=[32, 32, 16, 16]):
        super(get_model, self).__init__()
        self.nsample = nsample
        self.local_transformer = nn.ModuleList()
        self.decode_list = nn.ModuleList()
        self.fea_cov = nn.Sequential(
            nn.Conv1d(9, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        last_channel = 64
        last_point = 4096
        en_dims = [last_channel]
        # Encoder
        out_channel = last_channel
        for i in range(len(self.nsample)):
            last_point = int(last_point / 4)
            self.local_transformer.append(LocalTransformer(last_point, self.nsample[i], en_dims[-1], out_channel, ratio=4))
            last_channel = out_channel
            en_dims.append(last_channel)
            out_channel = last_channel * 2


        self.rfb1 = GlobalTransformer(512, 512, drop=0.2, ratio=4, src_pts=16)

        self.rfb3 = GlobalTransformer(128, 128, drop=0.2, ratio=4, src_pts=1024)

        # Decoder
        de_dims = [256, 256, 128, 128]
        en_dims.reverse()    
        de_dims.insert(0, en_dims[0])  
        #print(en_dims,de_dims)
        for i in range(len(en_dims)-1):
            self.decode_list.append(
                PointNetFeaturePropagation(de_dims[i]+en_dims[i+1], [de_dims[i+1], de_dims[i+1]])
            )


        # Global context mapping
        self.gcm_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gcm_list.append(nn.Sequential(
                nn.Conv1d(en_dim, 64, 1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)
            ))
        self.gcm_end = nn.Sequential(
                nn.Conv1d(64*5, 64, 1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)
            )

        # Predict head
        self.pred_head = nn.Sequential(
            nn.Conv1d(64+de_dims[-1], 64, 1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Conv1d(64, num_classes, 1)
        )

    def forward(self, x):
        xyz = x[:, :3, :]
        x = self.fea_cov(x)
        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, D, N]

        #encoder
        for i in range(len(self.nsample)):
            xyz, x = self.local_transformer[i](xyz, x)
            xyz_list.append(xyz)
            x_list.append(x)

        x = self.rfb1(xyz_list[-1], x_list[-1])
        x_list[-1] = x
        #decoder
        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]
        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i+1], xyz_list[i], x_list[i+1], x)
            if(i == 2):
                x = self.rfb3(xyz_list[i+1], x)


        #global context
        gcp_list = []
        for i in range(len(x_list)):
            gcp_list.append(F.adaptive_max_pool1d(self.gcm_list[i](x_list[i]), 1))
        global_context = self.gcm_end(torch.cat(gcp_list, dim=1))

        #pred
        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]])], dim=1)
        #x = self.rfb2(xyz_list[-1], x)
        pred = self.pred_head(x)
        pred_o = pred.transpose(2, 1).contiguous()
        pred = F.log_softmax(pred_o, dim=-1)
        return pred, pred_o#, x.permute(0, 2, 1)

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss
