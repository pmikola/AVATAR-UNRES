import gc

import torch.nn
from torch import nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP

if torch.cuda.is_available():
    device = torch.device('cuda')  # GPU available
else:
    device = torch.device('cpu')  # No GPU available, fallback to CPU


class dynamicAct(nn.Module):
    def __init__(self):
        super(dynamicAct, self).__init__()

    def activation(self, x, offset_x, offset_y, scale, low_cut_off, high_cut_off):
        c = []
        for i in range(x.shape[0]):
            a = (torch.tanh(x + offset_x[i]) + offset_y[i]) * scale[i]
            b = torch.nn.functional.hardtanh(x, min_val=-low_cut_off[i].item(), max_val=high_cut_off[i].item())
            c.append(a + b)
        return torch.tensor(c[0], device=device)

#
# class HopfieldNet:
#     def __init__(self, init):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.memory = torch.tensor(init, dtype=torch.float, device=self.device)
#         if self.memory.size()[0] > 1:
#             self.n = self.memory.size()[1]
#         else:
#             self.n = len(self.memory)
#         self.state = torch.randint(-2, 2, (self.n, 1), dtype=torch.float, device=self.device)  # state vector
#         self.weights = torch.zeros(self.n, self.n, dtype=torch.float, device=self.device)  # weights vector
#         self.energies = torch.tensor([], dtype=torch.float, device=self.device)  # tensor for tracking of energy
#
#     def learning(self):
#         self.weights = (1 / self.memory.size()[0]) * torch.matmul(self.memory.t(), self.memory)  # Hebbian learning
#         self.weights.fill_diagonal_(0)
#
#     def update_state(self, n_update):
#         for _ in range(n_update):  # update n neurons randomly
#             rand_index = torch.randint(0, self.n, (1,), dtype=torch.long,
#                                        device=self.device)  # pick a random neuron in the state vector
#             index_activation = torch.matmul(self.weights[rand_index, :],
#                                             self.state)  # compute activation for randomly indexed neuron
#             # threshold function for binary state change
#             rand_index_value = rand_index.item()  # Extract the scalar value from rand_index
#             self.state[rand_index_value] = torch.where(index_activation < 0, -1.0, 1.0)
#
#     def energyUpdate(self):
#         energy = -0.5 * torch.matmul(torch.matmul(self.state.t(), self.weights), self.state)
#         self.energies = torch.cat((self.energies, energy.view(1)), dim=0)


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
        self.uplift_dim = 500
        self.drop_dim = 250
        self.dynamics = self.pos.shape[1] + self.vel.shape[1] + self.acc.shape[1] + self.force.shape[1]

        self.uplift_meta = nn.Linear(self.meta.shape[1], self.uplift_dim, bias=True)
        self.uplift_pos = nn.Linear(self.pos.shape[1], self.uplift_dim, bias=True)
        self.uplift_vel = nn.Linear(self.vel.shape[1], self.uplift_dim, bias=True)
        self.uplift_acc = nn.Linear(self.acc.shape[1], self.uplift_dim, bias=True)
        self.uplift_force = nn.Linear(self.force.shape[1], self.uplift_dim, bias=True)

        self.astrA0 = nn.Linear(self.uplift_dim, self.uplift_dim, bias=True)
        self.astrB0 = nn.Linear(self.uplift_dim, 5, bias=True)
        self.h0 = nn.Linear(self.uplift_dim * 4, self.drop_dim, bias=True)
        self.d0 = dynamicAct()

        self.astrA1 = nn.Linear(self.uplift_dim, self.uplift_dim, bias=True)
        self.astrB1 = nn.Linear(self.uplift_dim, 5, bias=True)

        self.ARU1Uplif = nn.Linear(self.drop_dim, self.drop_dim, bias=True)
        # self.Hopfield1 = HopfieldNet(torch.zeros((10, 10)))  # reshaped self.drop_dim
        self.LSTM1 = nn.LSTM(input_size=self.drop_dim, hidden_size=self.drop_dim, num_layers=1, batch_first=True)
        self.ARU1Drop = nn.Linear(self.drop_dim, 5, bias=True)

        self.h1 = nn.Linear(self.drop_dim, self.drop_dim, bias=True)
        self.d1 = dynamicAct()

        self.astrA2 = nn.Linear(self.uplift_dim, self.uplift_dim, bias=True)
        self.astrB2 = nn.Linear(self.uplift_dim, 5, bias=True)
        self.h2 = nn.Linear(self.drop_dim, self.drop_dim, bias=True)
        self.d2 = dynamicAct()
        # self.hx0 = torch.zeros(self.drop_dim).to(device)
        # self.wiring = AutoNCP(self.drop_dim*2, self.drop_dim)
        # self.liquid0 = LTC(self.drop_dim, self.drop_dim)
        self.output = nn.Linear(self.drop_dim, self.dynamics, bias=True)

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

        # TODO : FOURIER or WNO

        up_out = torch.cat([p, v, a, f], dim=1)

        # ASTROCYTES CONTROL UNITS
        astrA0 = torch.tanh(self.astrA0(m))
        astrB0 = self.astrB0(astrA0)
        astrA1 = torch.tanh(self.astrA1(m))
        astrB1 = self.astrB1(astrA1)
        astrA2 = torch.tanh(self.astrA2(m))
        astrB2 = self.astrB2(astrA2)
        # ASTROCYTES CONTROL UNITS

        h0 = torch.tanh(self.h0(up_out))

        # ARU UNIT
        ARU1UP = torch.tanh(self.ARU1Uplif(h0))
        # lstm1, _ = self.LSTM1(ARU1UP)
        # TODO Recurent Hopefield network or LSTM
        ARU1DOWN = self.ARU1Drop(ARU1UP)
        # ARU UNIT

        h1 = torch.tanh(self.h1(h0))
        h2 = torch.tanh(self.h2(h1))

        # lq0, _ = self.liquid0(h0, self.hx0)
        out = self.output(h2)
        return out
