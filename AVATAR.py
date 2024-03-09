import gc
import sys

import torch.nn
from pytorch_tcn import TCN
from torch import nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP

from htm_pytorch import HTMAttention
import torch.nn.functional as F

from TCN3d import TCN3d
from utils import create_2d_views

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
torch.autograd.set_detect_anomaly(True)


# class dynamicAct(nn.Module):
#     def __init__(self):
#         super(dynamicAct, self).__init__()
#
#     def activation(self, x, offset_x, offset_y, scale, low_cut_off, high_cut_off):
#         c = []
#         for i in range(x.shape[0]):
#             a = (torch.tanh(x + offset_x[i]) + offset_y[i]) * scale[i]
#             b = torch.nn.functional.hardtanh(x, min_val=-low_cut_off[i].item(), max_val=high_cut_off[i].item())
#             c.append(a + b)
#         return torch.tensor(c[0], device=device)


class AvatarUNRES(nn.Module):
    def __init__(self, meta, pos, vel, acc, force):
        super().__init__()
        gc.collect()
        torch.cuda.empty_cache()

        self.meta = meta
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.force = force
        self.uplift_dim = 100
        self.drop_dim = 100
        self.modes = 16
        self.directions_num = 1
        self.lstm_layers = 2
        self.drop = 0.05
        self.bidirectional = True
        self.kernel_s = 3
        self.filters = 10
        self.dropout = 0.1
        self.grid_step = 0.01
        self.num_of_views = 8
        self.dist_coef = torch.tensor([0., 0., 0., 0., 0.], dtype=torch.float32)
        self.rot_ang = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1020], dtype=torch.float32)
        # self.rot_ang = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        self.distances = torch.tensor([0., 0., 2.], dtype=torch.float32)
        self.camera_params = torch.tensor([1., 1., 1., 1.], dtype=torch.float32)
        self.uplift_meta = nn.Linear(self.meta.shape[1], self.uplift_dim, bias=True)

        self.conv_p_uplift = nn.Conv2d(self.num_of_views, self.filters, kernel_size=self.kernel_s,
                                       dilation=1,
                                       padding=(self.kernel_s - 1) // 2, stride=1)

        self.bn1p = nn.BatchNorm2d(self.filters)
        self.dropout1p = nn.Dropout2d(self.dropout)

        self.conv_v_uplift = nn.Conv2d(self.num_of_views, self.filters, kernel_size=self.kernel_s,
                                       dilation=1,
                                       padding=(self.kernel_s - 1) // 2, stride=1)
        self.bn1v = nn.BatchNorm2d(self.filters)
        self.dropout1v = nn.Dropout2d(self.dropout)

        self.conv_a_uplift = nn.Conv2d(self.num_of_views, self.filters, kernel_size=self.kernel_s,
                                       dilation=1,
                                       padding=(self.kernel_s - 1) // 2, stride=1)
        self.bn1a = nn.BatchNorm2d(self.filters)
        self.dropout1a = nn.Dropout2d(self.dropout)

        self.conv_f_uplift = nn.Conv2d(self.num_of_views, self.filters, kernel_size=self.kernel_s,
                                       dilation=1,
                                       padding=(self.kernel_s - 1) // 2, stride=1)
        self.bn1f = nn.BatchNorm2d(self.filters)
        self.dropout1f = nn.Dropout2d(self.dropout)

        self.conv_p_downlift = nn.Conv2d(self.filters, self.num_of_views, kernel_size=self.kernel_s,
                                         dilation=(1,),
                                         padding=(self.kernel_s - 1) // 2, stride=1)
        self.conv_v_downlift = nn.Conv2d(self.filters, self.num_of_views, kernel_size=self.kernel_s,
                                         dilation=(1,),
                                         padding=(self.kernel_s - 1) // 2, stride=1)
        self.conv_a_downlift = nn.Conv2d(self.filters, self.num_of_views, kernel_size=self.kernel_s,
                                         dilation=(1,),
                                         padding=(self.kernel_s - 1) // 2, stride=1)
        self.conv_f_downlift = nn.Conv2d(self.filters, self.num_of_views, kernel_size=self.kernel_s,
                                         dilation=(1,),
                                         padding=(self.kernel_s - 1) // 2, stride=1)

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # nn.init.xavier_uniform_(module.weight)
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 1)

    def forward(self, meta, pos, vel, acc, force):

        p, _ = create_2d_views(self.grid_step, pos, self.dist_coef, self.rot_ang, self.distances, self.camera_params,
                                device)
        v, _ = create_2d_views(self.grid_step, vel, self.dist_coef, self.rot_ang, self.distances, self.camera_params,
                               device)
        a, _ = create_2d_views(self.grid_step, acc, self.dist_coef, self.rot_ang, self.distances, self.camera_params,
                               device)
        f, _ = create_2d_views(self.grid_step, force, self.dist_coef, self.rot_ang, self.distances, self.camera_params,
                               device)

        p = F.relu(self.conv_p_uplift(p))
        v = F.relu(self.conv_v_uplift(v))
        a = F.relu(self.conv_a_uplift(a))
        f = F.relu(self.conv_f_uplift(f))

        p = self.bn1p(p)
        v = self.bn1p(v)
        a = self.bn1p(a)
        f = self.bn1p(f)

        p = self.dropout1p(p)
        v = self.dropout1p(v)
        a = self.dropout1p(a)
        f = self.dropout1p(f)

        p = F.sigmoid(self.conv_p_downlift(p))
        v = F.sigmoid(self.conv_v_downlift(v))
        a = F.sigmoid(self.conv_a_downlift(a))
        f = F.sigmoid(self.conv_f_downlift(f))

        # m = self.uplift_meta(meta)
        # pred_p, pred_v, pred_a, pred_f = p.squeeze(1), v.squeeze(1), a.squeeze(
        #     1), f.squeeze(1)
        return p, v, a, f
