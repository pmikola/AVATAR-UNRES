import gc

import torch.nn
from pytorch_tcn import TCN
from torch import nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP
from torch.nn.functional import relu
from htm_pytorch import HTMAttention
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
torch.autograd.set_detect_anomaly(True)


class TCN(nn.Module):
    def __init__(self, features, output_size, layers=2, kernel_s=3, filters=100, dropout=0.2):
        super(TCN, self).__init__()
        self.output_size = output_size
        self.features = features
        self.layers = layers
        self.kernel_s = kernel_s
        self.filters = filters
        self.dropout = dropout

        self.conv1 = nn.Conv1d(1, self.filters, kernel_size=(kernel_s,), dilation=1,
                               padding=(kernel_s - 1) // 2, stride=(1,))
        # self.conv1 = nn.Conv1d(self.features, self.filters, kernel_size=(kernel_s, kernel_s), dilation=(1, 1),
        #                        padding=(kernel_s - 1) // 2, stride=(1,))
        self.bn1 = nn.BatchNorm1d(self.filters)
        self.dropout1 = nn.Dropout(self.dropout)

        self.conv2 = nn.Conv1d(self.filters, self.features, kernel_size=(kernel_s,), dilation=1,
                               padding=(kernel_s - 1) // 2, stride=(1,))
        self.bn2 = nn.BatchNorm1d(self.features)
        self.dropout2 = nn.Dropout(self.dropout)

        self.conv_1x1 = nn.Conv1d(1, self.features, kernel_size=(1,), )

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.bns2 = nn.ModuleList()
        self.dropouts2 = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()

        for i in range(layers - 1):
            dilation_factor = 2 ** (i + 1)
            self.convs.append(
                nn.Conv1d(self.features, self.filters, kernel_size=(kernel_s,), dilation=dilation_factor,
                          padding=(kernel_s - 1) * dilation_factor // 2, stride=(1,)))
            self.bns.append(nn.BatchNorm1d(self.filters))
            self.convs2.append(
                nn.Conv1d(self.filters, self.features, kernel_size=(kernel_s,), dilation=dilation_factor,
                          padding=(kernel_s - 1) * dilation_factor // 2, stride=(1,)))
            self.bns2.append(nn.BatchNorm1d(self.features))
            self.convs_1x1.append(
                nn.Conv1d(self.features, self.features, kernel_size=(1,)))

        self.out = nn.Linear(self.features, self.output_size,bias=False)

    def forward(self, x):
        # x = torch.swapaxes(x, 2, 1)
        # x = x.permute(0, 2, 1)
        res = self.conv_1x1(x)  # optional?
        # print(x.shape, 'x_res')

        x = self.conv1(x)
        # print(x.shape,'x1')
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.gelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        o = F.gelu(x)
        # print(o.shape,'o0')
        x = o + res

        for i in range(self.layers - 1):
            res = self.convs_1x1[i](x)
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.gelu(x)
            x = self.convs2[i](x)
            x = self.bns2[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            o = F.gelu(x)
            # print(x.shape, i)
            x = o + res
            x = F.gelu(x)

        # x = x.permute(0, 2, 1)
        x = x[:, -1, :]
        # print(x.shape)
        x = F.gelu(self.out(x))
        return x


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
        self.weights_real = nn.Parameter(self.scale * torch.rand(out_channels, self.modes, dtype=torch.cfloat))
        self.weights_imag = nn.Parameter(self.scale * torch.rand(out_channels, self.modes, dtype=torch.cfloat))
        self.combine_ri = nn.Linear(4 * out_channels, out_channels)

    def compl_mul1d(self, input, weights):
        out = torch.einsum("bi,ow->bw", input, weights)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.fft(x, dim=1)
        x_ft_real = x_ft.real
        x_ft_imag = x_ft.imag
        out_real = self.compl_mul1d(x_ft_real[:, :self.modes], self.weights_real)
        out_imag = self.compl_mul1d(x_ft_imag[:, :self.modes], self.weights_imag)
        x_real = torch.fft.ifft(out_real, n=x.size(-1))
        x_imag = torch.fft.ifft(out_imag, n=x.size(-1))
        x_r = torch.cat([x_real.real, x_imag.real], dim=1)  # only x_r is better than cat(x_r,x_i)
        x_i = torch.cat([x_real.imag, x_imag.imag], dim=1)
        x_ri = torch.cat([x_r, x_i], dim=1)
        x = self.combine_ri(x_ri)
        # This aproach is way better than only real fft domain (more weights or FFT plane vectors (so for eg. phase
        # info) just not lost?)
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
        self.directions_num = 1
        self.lstm_layers = 2
        self.drop = 0.05
        self.bidirectional = True
        self.dynamics = self.pos.shape[1] + self.vel.shape[1] + self.acc.shape[1] + \
                        self.force.shape[1]

        # self.LSTM_meta = nn.LSTM(input_size=self.meta.shape[1], hidden_size=self.meta.shape[1],
        #                          num_layers=self.lstm_layers,
        #                          batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        # self.LSTM_pos = nn.LSTM(input_size=self.pos.shape[1], hidden_size=self.pos.shape[1],
        #                         num_layers=self.lstm_layers,
        #                         batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        # self.LSTM_vel = nn.LSTM(input_size=self.vel.shape[1], hidden_size=self.vel.shape[1],
        #                         num_layers=self.lstm_layers,
        #                         batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        # self.LSTM_acc = nn.LSTM(input_size=self.acc.shape[1], hidden_size=self.acc.shape[1],
        #                         num_layers=self.lstm_layers,
        #                         batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        # self.LSTM_force = nn.LSTM(input_size=self.force.shape[1], hidden_size=self.force.shape[1],
        #                           num_layers=self.lstm_layers,
        #                           batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        self.m_in = TCN(self.meta.shape[1], self.uplift_dim)
        self.p_in = TCN(self.pos.shape[1], self.uplift_dim)
        self.v_in = TCN(self.vel.shape[1], self.uplift_dim)
        self.a_in = TCN(self.acc.shape[1], self.uplift_dim)
        self.f_in = TCN(self.force.shape[1], self.uplift_dim)

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
        self.h0 = nn.Linear(self.uplift_dim * 4, self.drop_dim, bias=True)
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
        self.astrB2 = nn.Linear(self.uplift_dim, self.uplift_dim, bias=True)
        self.h2 = nn.Linear(self.drop_dim, self.drop_dim, bias=True)
        # self.d2 = dynamicAct()
        # self.hx0 = torch.zeros(self.drop_dim).to(device)
        # self.wiring = AutoNCP(self.drop_dim*2, self.drop_dim)
        # self.liquid0 = LTC(self.drop_dim, self.drop_dim)
        self.common_output = nn.Linear(self.uplift_dim*4, self.uplift_dim, bias=True)

        # self.pos_out = nn.LSTM(input_size=self.dynamics, hidden_size=self.pos.shape[1], num_layers=self.lstm_layers,
        #                        batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        # self.vel_out = nn.LSTM(input_size=self.dynamics, hidden_size=self.vel.shape[1], num_layers=self.lstm_layers,
        #                        batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        # self.acc_out = nn.LSTM(input_size=self.dynamics, hidden_size=self.acc.shape[1], num_layers=self.lstm_layers,
        #                        batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)
        # self.force_out = nn.LSTM(input_size=self.dynamics, hidden_size=self.force.shape[1], num_layers=self.lstm_layers,
        #                          batch_first=True, bidirectional=self.bidirectional, dropout=self.drop)

        self.meta_out = TCN(self.uplift_dim, self.meta.shape[1])
        self.pos_out = TCN(self.uplift_dim, self.pos.shape[1])
        self.vel_out = TCN(self.uplift_dim, self.vel.shape[1])
        self.acc_out = TCN(self.uplift_dim, self.acc.shape[1])
        self.force_out = TCN(self.uplift_dim, self.force.shape[1])

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

    def forward(self, meta, pos, vel, acc, force, flag=0):

        m = meta.unsqueeze(1)
        p = pos.unsqueeze(1)
        v = vel.unsqueeze(1)
        a = acc.unsqueeze(1)
        f = force.unsqueeze(1)
        flag = 0
        if flag == 1:
            m = torch.swapaxes(m, 0, 1)
            p = torch.swapaxes(p, 0, 1)
            v = torch.swapaxes(v, 0, 1)
            a = torch.swapaxes(a, 0, 1)
            f = torch.swapaxes(f, 0, 1)

        m = self.m_in(m)
        p = self.p_in(p)
        v = self.v_in(v)
        a = self.a_in(a)
        f = self.f_in(f)
        if flag == 1:
            m = torch.swapaxes(m, 0, 1)
            p = torch.swapaxes(p, 0, 1)
            v = torch.swapaxes(v, 0, 1)
            a = torch.swapaxes(a, 0, 1)
            f = torch.swapaxes(f, 0, 1)
        m = m.squeeze(1)
        p = p.squeeze(1)
        v = v.squeeze(1)
        a = a.squeeze(1)
        f = f.squeeze(1)
        # m = F.gelu(self.uplift_meta(m))
        # p = F.gelu(self.uplift_pos(p))
        # v = F.gelu(self.uplift_vel(v))
        # a = F.gelu(self.uplift_acc(a))
        # f = F.gelu(self.uplift_force(f))

        # TODO :WNO - compare with FNO
        # fpos = self.spectralConvp0(p)
        # wpos = self.wp0(p)
        # fwpos = F.gelu(fpos + wpos)
        #
        # fpos = self.spectralConvp1(fwpos)
        # wpos = self.wp1(fwpos)
        # fwpos = F.gelu(fpos + wpos)
        #
        # fpos = self.spectralConvp2(fwpos)
        # wpos = self.wp2(fwpos)
        # fwpos = F.gelu(fpos + wpos)
        #
        # fvel = self.spectralConvv0(v)
        # wvel = self.wv0(v)
        # fwvel = F.gelu(fvel + wvel)
        #
        # fvel = self.spectralConvv1(fwvel)
        # wvel = self.wv1(fwvel)
        # fwvel = F.gelu(fvel + wvel)
        #
        # fvel = self.spectralConvv2(fwvel)
        # wvel = self.wv2(fwvel)
        # fwvel = F.gelu(fvel + wvel)
        #
        # facc = self.spectralConva0(a)
        # wacc = self.wa0(a)
        # fwacc = F.gelu(facc + wacc)
        #
        # facc = self.spectralConva1(fwacc)
        # wacc = self.wa1(fwacc)
        # fwacc = F.gelu(facc + wacc)
        #
        # facc = self.spectralConva2(fwacc)
        # wacc = self.wa2(fwacc)
        # fwacc = F.gelu(facc + wacc)
        #
        # fforce = self.spectralConvf0(f)
        # wforce = self.wf0(f)
        # fwforce = F.gelu(fforce + wforce)
        #
        # fforce = self.spectralConvf1(fwforce)
        # wforce = self.wf1(fwforce)
        # fwforce = F.gelu(fforce + wforce)
        #
        # fforce = self.spectralConvf2(fwforce)
        # wforce = self.wf2(fwforce)
        # fwforce = F.gelu(fforce + wforce)

        up_out = torch.cat([p, v, a, f], dim=1)

        # ASTROCYTES CONTROL UNITS
        astrA0 = F.gelu(self.astrA0(m))
        astrB0 = F.gelu(self.astrB0(astrA0))
        # astrA1 = F.gelu(self.astrA1(m))
        # astrB1 = F.gelu(self.astrB1(astrA1))
        # astrA2 = F.gelu(self.astrA2(m))
        # astrB2 = F.gelu(self.astrB2(astrA2))
        # ASTROCYTES CONTROL UNITS
        # ARU UNITs
        # h0 = torch.nn.functional.gelu(self.h0(up_out))
        # h1 = F.gelu(self.h1(h0)) * astrB1
        # h2 = F.gelu(self.h2(h1)) * astrB2

        # lq0, _ = self.liquid0(h0, self.hx0)

        co = torch.nn.functional.gelu(self.common_output(up_out)) * astrB0
        co = co.unsqueeze(1)
        if flag == 1:
            co = torch.swapaxes(co, 0, 1)

        pred_p = self.pos_out(co)
        pred_v = self.vel_out(co)
        pred_a = self.acc_out(co)
        pred_f = self.force_out(co)
        if flag == 1:
            pred_p = torch.swapaxes(pred_p, 0, 1)
            pred_v = torch.swapaxes(pred_v, 0, 1)
            pred_a = torch.swapaxes(pred_a, 0, 1)
            pred_f = torch.swapaxes(pred_f, 0, 1)

        pred_p, pred_v, pred_a, pred_f = pred_p.squeeze(1), pred_v.squeeze(1), pred_a.squeeze(
            1), pred_f.squeeze(1)
        # print(hnp.shape,pred_pos.shape)

        # pred_pos = self.p_out(pred_p)
        # pred_vel = self.v_out(pred_v)
        # pred_acc = self.a_out(pred_a)
        # pred_force = self.f_out(pred_f)

        return pred_p, pred_v, pred_a, pred_f
