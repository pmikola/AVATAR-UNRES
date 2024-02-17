import torch.nn
from torch import nn


class AvatarUNRES(nn.Module):
    def __init__(self, meta, pos, vel, acc, force):
        super().__init__()
        self.meta = meta
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.force = force
        self.uplift_out_dim = 300
        self.out = self.pos.shape[1] + self.vel.shape[1] + self.acc.shape[1] + self.force.shape[1]

        self.uplift_meta = nn.Linear(self.meta.shape[1], self.uplift_out_dim, bias=True)
        self.uplift_pos = nn.Linear(self.pos.shape[1], self.uplift_out_dim, bias=True)
        self.uplift_vel = nn.Linear(self.vel.shape[1], self.uplift_out_dim, bias=True)
        self.uplift_acc = nn.Linear(self.acc.shape[1], self.uplift_out_dim, bias=True)
        self.uplift_force = nn.Linear(self.force.shape[1], self.uplift_out_dim, bias=True)

        self.h1 = nn.Linear(self.uplift_out_dim*5, self.uplift_out_dim, )
        self.output = nn.Linear(self.uplift_out_dim, self.out)

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, meta, pos, vel, acc, force):
        m = self.uplift_meta(meta)
        p = self.uplift_pos(pos)
        v = self.uplift_vel(vel)
        a = self.uplift_acc(acc)
        f = self.uplift_force(force)

        up_out = torch.cat([m, p, v, a, f])
        h1 = torch.relu(self.h1(up_out))
        out = self.output(h1)

        return out
