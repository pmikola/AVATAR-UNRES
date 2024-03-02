import torch
from torch import nn


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