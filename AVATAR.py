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


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(out_channels, self.modes, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        out = torch.einsum("bi,ow->bw", input, weights)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x, dim=1)
        out_ft = self.compl_mul1d(x_ft[:, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


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
        self.uplift_dim = 1500
        self.drop_dim = 1500
        self.dynamics = self.pos.shape[1] + self.vel.shape[1] + self.acc.shape[1] + self.force.shape[1]

        self.LSTM_pos = nn.LSTM(input_size=self.pos.shape[1], hidden_size=self.uplift_dim, num_layers=2, batch_first=True)
        self.LSTM_vel = nn.LSTM(input_size=self.vel.shape[1], hidden_size=self.uplift_dim, num_layers=2, batch_first=True)
        self.LSTM_acc = nn.LSTM(input_size=self.acc.shape[1], hidden_size=self.uplift_dim, num_layers=2, batch_first=True)
        self.LSTM_force = nn.LSTM(input_size=self.force.shape[1], hidden_size=self.uplift_dim, num_layers=2,
                                  batch_first=True)

        self.uplift_meta = nn.Linear(self.meta.shape[1], self.uplift_dim, bias=True)
        self.uplift_pos = nn.Linear(self.uplift_dim, self.uplift_dim, bias=True)
        self.uplift_vel = nn.Linear(self.uplift_dim, self.uplift_dim, bias=True)
        self.uplift_acc = nn.Linear(self.uplift_dim, self.uplift_dim, bias=True)
        self.uplift_force = nn.Linear(self.uplift_dim, self.uplift_dim, bias=True)
        self.modes = 16
        self.spectralConvp0 = SpectralConv1d(self.uplift_dim, self.uplift_dim, self.modes)
        self.wp0 = nn.Linear(self.uplift_dim, self.uplift_dim)
        self.spectralConvp1 = SpectralConv1d(self.uplift_dim, self.uplift_dim, self.modes)
        self.wp1 = nn.Linear(self.uplift_dim, self.uplift_dim)
        self.spectralConvp2 = SpectralConv1d(self.uplift_dim, self.uplift_dim, self.modes)
        self.wp2 = nn.Linear(self.uplift_dim, self.uplift_dim)

        self.spectralConvv0 = SpectralConv1d(self.uplift_dim, self.uplift_dim, self.modes)
        self.wv0 = nn.Linear(self.uplift_dim, self.uplift_dim)
        self.spectralConvv1 = SpectralConv1d(self.uplift_dim, self.uplift_dim, self.modes)
        self.wv1 = nn.Linear(self.uplift_dim, self.uplift_dim)
        self.spectralConvv2 = SpectralConv1d(self.uplift_dim, self.uplift_dim, self.modes)
        self.wv2 = nn.Linear(self.uplift_dim, self.uplift_dim)

        self.spectralConva0 = SpectralConv1d(self.uplift_dim, self.uplift_dim, self.modes)
        self.wa0 = nn.Linear(self.uplift_dim, self.uplift_dim)
        self.spectralConva1 = SpectralConv1d(self.uplift_dim, self.uplift_dim, self.modes)
        self.wa1 = nn.Linear(self.uplift_dim, self.uplift_dim)
        self.spectralConva2 = SpectralConv1d(self.uplift_dim, self.uplift_dim, self.modes)
        self.wa2 = nn.Linear(self.uplift_dim, self.uplift_dim)

        self.spectralConvf0 = SpectralConv1d(self.uplift_dim, self.uplift_dim, 16)
        self.wf0 = nn.Linear(self.uplift_dim, self.uplift_dim)
        self.spectralConvf1 = SpectralConv1d(self.uplift_dim, self.uplift_dim, 16)
        self.wf1 = nn.Linear(self.uplift_dim, self.uplift_dim)
        self.spectralConvf2 = SpectralConv1d(self.uplift_dim, self.uplift_dim, 16)
        self.wf2 = nn.Linear(self.uplift_dim, self.uplift_dim)

        self.astrA0 = nn.Linear(self.uplift_dim, self.uplift_dim, bias=True)
        self.astrB0 = nn.Linear(self.uplift_dim, 5, bias=True)
        self.h0 = nn.Linear(self.uplift_dim * 4, self.drop_dim, bias=True)
        self.d0 = dynamicAct()

        self.astrA1 = nn.Linear(self.uplift_dim, self.uplift_dim, bias=True)
        self.astrB1 = nn.Linear(self.uplift_dim, 5, bias=True)

        self.ARU1Uplif = nn.Linear(self.drop_dim, self.drop_dim, bias=True)
        # self.Hopfield1 = HopfieldNet(torch.zeros((10, 10)))  # reshaped self.drop_dim
        self.LSTM1 = nn.LSTM(input_size=self.uplift_dim, hidden_size=self.uplift_dim, num_layers=1, batch_first=True)
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
        self.common_output = nn.Linear(self.drop_dim, self.dynamics, bias=True)

        self.pos_out = nn.Linear(self.dynamics, self.pos.shape[1], bias=True)
        self.vel_out = nn.Linear(self.dynamics, self.vel.shape[1], bias=True)
        self.acc_out = nn.Linear(self.dynamics, self.acc.shape[1], bias=True)
        self.force_out = nn.Linear(self.dynamics, self.force.shape[1], bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, meta, pos, vel, acc, force):

        p = pos.unsqueeze(1)
        p,_ = self.LSTM_pos(p)
        v = vel.unsqueeze(1)
        v, _ = self.LSTM_vel(v)
        a = acc.unsqueeze(1)
        a, _ = self.LSTM_acc(a)
        f = force.unsqueeze(1)
        f, _ = self.LSTM_force(f)
        m = self.uplift_meta(meta)
        p = torch.tanh(self.uplift_pos(p.squeeze(1)))

        v = torch.tanh(self.uplift_vel(v.squeeze(1)))
        a = torch.tanh(self.uplift_acc(a.squeeze(1)))
        f = torch.tanh(self.uplift_force(f.squeeze(1)))

        # TODO :WNO - compare with FNO
        fpos = self.spectralConvp0(p)
        wpos = self.wp0(p)
        fwpos = torch.tanh(fpos + wpos)

        fpos = self.spectralConvp1(fwpos)
        wpos = self.wp1(fwpos)
        fwpos = torch.tanh(fpos + wpos)

        fpos = self.spectralConvp2(fwpos)
        wpos = self.wp2(fwpos)
        fwpos = torch.tanh(fpos + wpos)

        fvel = self.spectralConvv0(v)
        wvel = self.wv0(v)
        fwvel = torch.tanh(fvel + wvel)

        fvel = self.spectralConvv1(fwvel)
        wvel = self.wv1(fwvel)
        fwvel = torch.tanh(fvel + wvel)

        fvel = self.spectralConvv2(fwvel)
        wvel = self.wv2(fwvel)
        fwvel = torch.tanh(fvel + wvel)

        facc = self.spectralConva0(a)
        wacc = self.wa0(a)
        fwacc = torch.tanh(facc + wacc)

        facc = self.spectralConva1(fwacc)
        wacc = self.wa1(fwacc)
        fwacc = torch.tanh(facc + wacc)

        facc = self.spectralConva2(fwacc)
        wacc = self.wa2(fwacc)
        fwacc = torch.tanh(facc + wacc)

        fforce = self.spectralConvf0(f)
        wforce = self.wf0(f)
        fwforce = torch.tanh(fforce + wforce)

        fforce = self.spectralConvf1(fwforce)
        wforce = self.wf1(fwforce)
        fwforce = torch.tanh(fforce + wforce)

        fforce = self.spectralConvf2(fwforce)
        wforce = self.wf2(fwforce)
        fwforce = torch.tanh(fforce + wforce)

        up_out = torch.cat([fwpos, fwvel, fwacc, fwforce], dim=1)

        # ASTROCYTES CONTROL UNITS
        # astrA0 = torch.tanh(self.astrA0(m))
        # astrB0 = self.astrB0(astrA0)
        # astrA1 = torch.tanh(self.astrA1(m))
        # astrB1 = self.astrB1(astrA1)
        # astrA2 = torch.tanh(self.astrA2(m))
        # astrB2 = self.astrB2(astrA2)
        # ASTROCYTES CONTROL UNITS

        h0 = torch.tanh(self.h0(up_out))

        # ARU UNIT
        # ARU1UP = torch.tanh(self.ARU1Uplif(h0))
        # lstm1, _ = self.LSTM1(ARU1UP)
        # TODO Recurent Hopefield network or LSTM or Liquid NN
        # ARU1DOWN = self.ARU1Drop(ARU1UP)
        # ARU UNIT
        h1 = torch.tanh(self.h1(h0))
        h2 = torch.tanh(self.h2(h1))

        # lq0, _ = self.liquid0(h0, self.hx0)
        co = torch.tanh(self.common_output(h2))
        pred_pos = self.pos_out(co)
        pred_vel = self.vel_out(co)
        pred_acc = self.acc_out(co)
        pred_force = self.force_out(co)
        return pred_pos, pred_vel, pred_acc, pred_force
