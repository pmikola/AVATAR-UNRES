import gc
import math
import sys

import torch.nn
from pytorch_tcn import TCN
from torch import nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP

import torch.nn.functional as F

from TCN3d import TCN3d
from utils import create_2d_views


# torch.autograd.set_detect_anomaly(True)


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
        self.kernel_s = 3
        self.dropout = 0.05
        self.grid_step = 0.005
        self.grid_padding = int((4. / self.grid_step))
        self.fx = 1.  # / (2 * math.tan(0.5 * self.fov)) # aspect ratio x
        self.fy = 1.  # / (2 * math.tan(0.5 * self.fov)) # aspect ratio y
        self.fz = 1.  # / (2 * math.tan(0.5 * 90.))
        self.Cx, self.Cy, self.Cz = 1., 1., 0.  # principal points where optic axis intersect with the image plane
        self.gamma = 0.
        self.tx, self.ty, self.tz = -0.1, 0.1, 0.1
        self.no_quadrants = 4
        self.dist_coef = torch.tensor([20., 20., 30., 40., 80.], dtype=torch.float32)
        self.rot_ang = torch.tensor([20, 40, 10, 15, 5], dtype=torch.float32)
        self.num_of_views = self.rot_ang.shape[0] * self.no_quadrants
        self.filters = self.num_of_views
        self.translation = torch.tensor([self.tx, self.ty, self.tz], dtype=torch.float32)
        self.camera_params = torch.tensor([self.fx, self.fy, self.fz, self.Cx, self.Cy, self.Cz],
                                          dtype=torch.float32)
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
        p, pz = create_2d_views(pos, self.grid_step, self.grid_padding, self.dist_coef, self.rot_ang, self.translation,
                                self.camera_params, self.device)
        v, vz = create_2d_views(vel, self.grid_step, self.grid_padding, self.dist_coef, self.rot_ang, self.translation,
                                self.camera_params, self.device)
        a, az = create_2d_views(acc, self.grid_step, self.grid_padding, self.dist_coef, self.rot_ang, self.translation,
                                self.camera_params, self.device)
        f, fz = create_2d_views(force, self.grid_step, self.grid_padding, self.dist_coef, self.rot_ang,
                                self.translation,
                                self.camera_params, self.device)

        p = p.reshape(shape=(p.shape[0], p.shape[1] * p.shape[2], p.shape[3], p.shape[4]))
        v = v.reshape(shape=(v.shape[0], v.shape[1] * v.shape[2], v.shape[3], v.shape[4]))
        a = a.reshape(shape=(a.shape[0], a.shape[1] * a.shape[2], a.shape[3], a.shape[4]))
        f = f.reshape(shape=(f.shape[0], f.shape[1] * f.shape[2], f.shape[3], f.shape[4]))

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

        p = self.conv_p_downlift(p)
        v = self.conv_v_downlift(v)
        a = self.conv_a_downlift(a)
        f = self.conv_f_downlift(f)

        p = p.reshape(shape=(p.shape[0], self.rot_ang.shape[0], self.no_quadrants, p.shape[2], p.shape[3]))
        v = v.reshape(shape=(v.shape[0], self.rot_ang.shape[0], self.no_quadrants, v.shape[2], v.shape[3]))
        a = a.reshape(shape=(a.shape[0], self.rot_ang.shape[0], self.no_quadrants, a.shape[2], a.shape[3]))
        f = f.reshape(shape=(f.shape[0], self.rot_ang.shape[0], self.no_quadrants, f.shape[2], f.shape[3]))

        p = torch.sigmoid(p)
        v = torch.sigmoid(v)
        a = torch.sigmoid(a)
        f = torch.sigmoid(f)

        # m = self.uplift_meta(meta)
        # pred_p, pred_v, pred_a, pred_f = p.squeeze(1), v.squeeze(1), a.squeeze(
        #     1), f.squeeze(1)
        return p, v, a, f, pz, vz, az, fz
