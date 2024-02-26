import gc

import torch.nn
from torch import nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP
from torch.nn.functional import relu

if torch.cuda.is_available():
    device = torch.device('cuda')  # GPU available
else:
    device = torch.device('cpu')  # No GPU available, fallback to CPU


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
def real_imaginary_relu(z):
    return relu(z.real) + 1.j * relu(z.imag)

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
        self.uplift_dim = 1000
        self.drop_dim = 1000
        self.modes = 16
        self.directions_num = 2
        self.lstm_layers = 2
        self.drop = 0.05
        self.bidirectional = True
        self.dynamics = self.pos.shape[1] + self.vel.shape[1] + self.acc.shape[1] + \
                        self.force.shape[1]

        self.LSTM_meta = nn.LSTM(input_size=self.meta.shape[1], hidden_size=self.meta.shape[1],
                                 num_layers=self.lstm_layers,
                                 batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        self.LSTM_pos = nn.LSTM(input_size=self.pos.shape[1], hidden_size=self.pos.shape[1],
                                num_layers=self.lstm_layers,
                                batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        self.LSTM_vel = nn.LSTM(input_size=self.vel.shape[1], hidden_size=self.vel.shape[1],
                                num_layers=self.lstm_layers,
                                batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        self.LSTM_acc = nn.LSTM(input_size=self.acc.shape[1], hidden_size=self.acc.shape[1],
                                num_layers=self.lstm_layers,
                                batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        self.LSTM_force = nn.LSTM(input_size=self.force.shape[1], hidden_size=self.force.shape[1],
                                  num_layers=self.lstm_layers,
                                  batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)

        self.uplift_meta = nn.Linear(self.meta.shape[1] * self.directions_num, self.uplift_dim, bias=True)
        self.uplift_pos = nn.Linear(self.pos.shape[1] * self.directions_num, self.uplift_dim, bias=True)
        self.uplift_vel = nn.Linear(self.vel.shape[1] * self.directions_num, self.uplift_dim, bias=True)
        self.uplift_acc = nn.Linear(self.acc.shape[1] * self.directions_num, self.uplift_dim, bias=True)
        self.uplift_force = nn.Linear(self.force.shape[1] * self.directions_num, self.uplift_dim, bias=True)

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

        self.spectralConvf0 = SpectralConv1d(self.uplift_dim, self.uplift_dim, self.modes)
        self.wf0 = nn.Linear(self.uplift_dim, self.uplift_dim)
        self.spectralConvf1 = SpectralConv1d(self.uplift_dim, self.uplift_dim, self.modes)
        self.wf1 = nn.Linear(self.uplift_dim, self.uplift_dim)
        self.spectralConvf2 = SpectralConv1d(self.uplift_dim, self.uplift_dim, self.modes)
        self.wf2 = nn.Linear(self.uplift_dim, self.uplift_dim)

        self.astrA0 = nn.Linear(self.uplift_dim, self.uplift_dim, bias=True)
        self.astrB0 = nn.Linear(self.uplift_dim, self.drop_dim, bias=True)
        self.h0 = nn.Linear(self.uplift_dim * 5, self.drop_dim, bias=True)
        # self.d0 = dynamicAct()

        # self.ARU1Uplif = nn.Linear(self.drop_dim, self.drop_dim, bias=True)
        # # self.Hopfield1 = HopfieldNet(torch.zeros((10, 10)))  # reshaped self.drop_dim
        # self.LSTM1 = nn.LSTM(input_size=self.uplift_dim, hidden_size=self.uplift_dim, num_layers=1, batch_first=True)
        # self.ARU1Drop = nn.Linear(self.drop_dim, 5, bias=True)

        self.astrA1 = nn.Linear(self.uplift_dim, self.uplift_dim, bias=True)
        self.astrB1 = nn.Linear(self.uplift_dim, self.drop_dim, bias=True)
        self.h1 = nn.Linear(self.drop_dim, self.drop_dim, bias=True)
        # self.d1 = dynamicAct()

        self.astrA2 = nn.Linear(self.uplift_dim, self.uplift_dim, bias=True)
        self.astrB2 = nn.Linear(self.uplift_dim, self.drop_dim, bias=True)
        self.h2 = nn.Linear(self.drop_dim, self.drop_dim, bias=True)
        # self.d2 = dynamicAct()
        # self.hx0 = torch.zeros(self.drop_dim).to(device)
        # self.wiring = AutoNCP(self.drop_dim*2, self.drop_dim)
        # self.liquid0 = LTC(self.drop_dim, self.drop_dim)
        self.common_output = nn.Linear(self.drop_dim, self.dynamics, bias=True)

        self.pos_out = nn.LSTM(input_size=self.dynamics, hidden_size=self.pos.shape[1], num_layers=self.lstm_layers,
                               batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        self.vel_out = nn.LSTM(input_size=self.dynamics, hidden_size=self.vel.shape[1], num_layers=self.lstm_layers,
                               batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        self.acc_out = nn.LSTM(input_size=self.dynamics, hidden_size=self.acc.shape[1], num_layers=self.lstm_layers,
                               batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        self.force_out = nn.LSTM(input_size=self.dynamics, hidden_size=self.force.shape[1], num_layers=self.lstm_layers,
                                 batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)

        self.p_out = nn.Linear(self.pos.shape[1] * self.directions_num, self.pos.shape[1], bias=True)
        self.v_out = nn.Linear(self.vel.shape[1] * self.directions_num, self.vel.shape[1], bias=True)
        self.a_out = nn.Linear(self.acc.shape[1] * self.directions_num, self.acc.shape[1], bias=True)
        self.f_out = nn.Linear(self.force.shape[1] * self.directions_num, self.force.shape[1], bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # nn.init.xavier_uniform_(module.weight)
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 1)

    def init_h0_c0ForLSTMs(self, no_steps):
        h0m = torch.empty(self.lstm_layers * self.directions_num, no_steps, self.meta.shape[1]).to(device)
        c0m = torch.empty(self.lstm_layers * self.directions_num, no_steps, self.meta.shape[1]).to(device)
        h0p = torch.empty(self.lstm_layers * self.directions_num, no_steps, self.pos.shape[1]).to(device)
        c0p = torch.empty(self.lstm_layers * self.directions_num, no_steps, self.pos.shape[1]).to(device)
        h0v = torch.empty(self.lstm_layers * self.directions_num, no_steps, self.vel.shape[1]).to(device)
        c0v = torch.empty(self.lstm_layers * self.directions_num, no_steps, self.vel.shape[1]).to(device)
        h0a = torch.empty(self.lstm_layers * self.directions_num, no_steps, self.acc.shape[1]).to(device)
        c0a = torch.empty(self.lstm_layers * self.directions_num, no_steps, self.acc.shape[1]).to(device)
        h0f = torch.empty(self.lstm_layers * self.directions_num, no_steps, self.force.shape[1]).to(device)
        c0f = torch.empty(self.lstm_layers * self.directions_num, no_steps, self.force.shape[1]).to(device)
        nn.init.uniform_(h0m, a=0, b=1)
        nn.init.uniform_(c0m, a=0, b=1)
        nn.init.uniform_(h0p, a=0, b=1)
        nn.init.uniform_(c0p, a=0, b=1)
        nn.init.uniform_(h0v, a=0, b=1)
        nn.init.uniform_(c0v, a=0, b=1)
        nn.init.uniform_(h0a, a=0, b=1)
        nn.init.uniform_(c0a, a=0, b=1)
        nn.init.uniform_(h0f, a=0, b=1)
        nn.init.uniform_(c0f, a=0, b=1)

        return h0m, c0m, h0p, c0p, h0v, c0v, h0a, c0a, h0f, c0f

    def forward(self, meta, pos, vel, acc, force, hc4lstm, b_st_flag=0):
        h0m, c0m, h0p, c0p, h0v, c0v, h0a, c0a, h0f, c0f = hc4lstm

        m = meta.unsqueeze(1)
        p = pos.unsqueeze(1)
        v = vel.unsqueeze(1)
        a = acc.unsqueeze(1)
        f = force.unsqueeze(1)

        # if b_st_flag == 1:
        #     m = torch.swapaxes(m, 0, 1)
        #     p = torch.swapaxes(m, 0, 1)
        #     v = torch.swapaxes(m, 0, 1)
        #     a = torch.swapaxes(m, 0, 1)
        #     f = torch.swapaxes(m, 0, 1)

        m, (hnm, cnm) = self.LSTM_meta(m, (h0m, c0m))
        p, (hnp, cnp) = self.LSTM_pos(p, (h0p, c0p))
        v, (hnv, cnv) = self.LSTM_vel(v, (h0v, c0v))
        a, (hna, cna) = self.LSTM_acc(a, (h0a, c0a))
        f, (hnf, cnf) = self.LSTM_force(f, (h0f, c0f))
        # print(p)
        m = torch.nn.functional.gelu(self.uplift_meta(m.squeeze(1)))
        p = torch.nn.functional.gelu(self.uplift_pos(p.squeeze(1)))

        v = torch.nn.functional.gelu(self.uplift_vel(v.squeeze(1)))
        a = torch.nn.functional.gelu(self.uplift_acc(a.squeeze(1)))
        f = torch.nn.functional.gelu(self.uplift_force(f.squeeze(1)))

        # TODO :WNO - compare with FNO
        fpos = self.spectralConvp0(p)
        wpos = self.wp0(p)
        fwpos = torch.nn.functional.gelu(fpos + wpos)

        fpos = self.spectralConvp1(fwpos)
        wpos = self.wp1(fwpos)
        fwpos = torch.nn.functional.gelu(fpos + wpos)

        fpos = self.spectralConvp2(fwpos)
        wpos = self.wp2(fwpos)
        fwpos = torch.nn.functional.gelu(fpos + wpos)

        fvel = self.spectralConvv0(v)
        wvel = self.wv0(v)
        fwvel = torch.nn.functional.gelu(fvel + wvel)

        fvel = self.spectralConvv1(fwvel)
        wvel = self.wv1(fwvel)
        fwvel = torch.nn.functional.gelu(fvel + wvel)

        fvel = self.spectralConvv2(fwvel)
        wvel = self.wv2(fwvel)
        fwvel = torch.nn.functional.gelu(fvel + wvel)

        facc = self.spectralConva0(a)
        wacc = self.wa0(a)
        fwacc = torch.nn.functional.gelu(facc + wacc)

        facc = self.spectralConva1(fwacc)
        wacc = self.wa1(fwacc)
        fwacc = torch.nn.functional.gelu(facc + wacc)

        facc = self.spectralConva2(fwacc)
        wacc = self.wa2(fwacc)
        fwacc = torch.nn.functional.gelu(facc + wacc)

        fforce = self.spectralConvf0(f)
        wforce = self.wf0(f)
        fwforce = torch.nn.functional.gelu(fforce + wforce)

        fforce = self.spectralConvf1(fwforce)
        wforce = self.wf1(fwforce)
        fwforce = torch.nn.functional.gelu(fforce + wforce)

        fforce = self.spectralConvf2(fwforce)
        wforce = self.wf2(fwforce)
        fwforce = torch.nn.functional.gelu(fforce + wforce)

        up_out = torch.cat([m, fwpos, fwvel, fwacc, fwforce], dim=1)

        # ASTROCYTES CONTROL UNITS
        astrA0 = torch.nn.functional.gelu(self.astrA0(m))
        astrB0 = torch.nn.functional.gelu(self.astrB0(astrA0))
        # astrA1 = torch.nn.functional.gelu(self.astrA1(m))
        # astrB1 = torch.nn.functional.gelu(self.astrB1(astrA1))
        # astrA2 = torch.nn.functional.gelu(self.astrA2(m))
        # astrB2 = torch.nn.functional.gelu(self.astrB2(astrA2))
        # ASTROCYTES CONTROL UNITS
        # ARU UNITs
        h0 = torch.nn.functional.gelu(self.h0(up_out)) * astrB0
        # h1 = torch.nn.functional.gelu(self.h1(h0)) * astrB1
        # h2 = torch.nn.functional.gelu(self.h2(h1)) * astrB2

        # lq0, _ = self.liquid0(h0, self.hx0)

        co = torch.nn.functional.gelu(self.common_output(h0))
        co = co.unsqueeze(1)
        pred_pos, (hnp, cnp) = self.pos_out(co, (hnp, cnp))
        pred_vel, (hnv, cnv) = self.vel_out(co, (hnv, cnv))
        pred_acc, (hna, cna) = self.acc_out(co, (hna, cna))
        pred_force, (hnf, cnf) = self.force_out(co, (hnf, cnf))
        pred_pos, pred_vel, pred_acc, pred_force = pred_pos.squeeze(1), pred_vel.squeeze(1), pred_acc.squeeze(
            1), pred_force.squeeze(1)
        # print(hnp.shape,pred_pos.shape)
        hc4lstm = hnm, cnm, hnp, cnp, hnv, cnv, hna, cna, hnf, cnf
        pred_pos = self.p_out(pred_pos)
        pred_vel = self.v_out(pred_vel)
        pred_acc = self.a_out(pred_acc)
        pred_force = self.f_out(pred_force)
        return pred_pos, pred_vel, pred_acc, pred_force, hc4lstm
