import gc

import torch.nn
from pytorch_tcn import TCN
from torch import nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP

from htm_pytorch import HTMAttention
import torch.nn.functional as F

from TCN3d import TCN3d

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

        self.uplift_meta = nn.Linear(self.meta.shape[1], self.uplift_dim, bias=True)


        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # nn.init.xavier_uniform_(module.weight)
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 1)

    def forward(self, meta, pos, vel, acc, force):



        # m = self.uplift_meta(meta)


        return p, v, a, f
