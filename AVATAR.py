import math, gc, torch, torch.nn as nn, torch.nn.functional as F
from timm.layers import DropPath
from linformer import Linformer

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(rms + self.eps)

class FNOBlock(nn.Module):
    def __init__(self, d_model, seq_len, fourier_dim=None):
        super().__init__()
        self.seq_len = seq_len
        self.fdim    = fourier_dim or d_model
        self.lin1    = nn.Linear(d_model, self.fdim)
        self.lin2    = nn.Linear(self.fdim, d_model)
        self.scale   = 1 / math.sqrt(2)

    def forward(self, x):
        B, L, D = x.shape
        xf = torch.fft.rfft(x, dim=1)
        nf = xf.size(1)
        real = xf.real.flatten(0,1)
        imag = xf.imag.flatten(0,1)
        real = self.lin1(real)
        imag = self.lin1(imag)
        xf_hat = torch.complex(real, imag).view(B, nf, -1)  # (B, nf, fdim)
        xi = torch.fft.irfft(xf_hat, n=self.seq_len, dim=1)
        return F.gelu(x + self.lin2(xi) * self.scale)

class Residual1D(nn.Module):
    def __init__(self, c, k=3, p=0.1):
        super().__init__()
        pad = (k - 1) // 2
        self.c1 = nn.Conv1d(c, c, k, padding=pad)
        self.c2 = nn.Conv1d(c, c, k, padding=pad)
        self.norm = RMSNorm(c)
        self.act = nn.SiLU()
        self.dp = nn.Dropout(p)
        self.scale = 1 / math.sqrt(2)
    def forward(self, x):
        y = self.c1(x)
        y = self.act(self.norm(y.transpose(1, 2))).transpose(1, 2)
        y = self.dp(y)
        y = self.c2(y)
        return x + y * self.scale

class Stem(nn.Module):
    def __init__(self, d_in, d_model, n_blocks, p):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([Residual1D(d_model, p=p) for _ in range(n_blocks)])
    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        return x.transpose(1, 2)

class SwiGLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_out * 2)
        self.w2 = nn.Linear(d_out, d_in)
    def forward(self, x):
        a, b = self.w1(x).chunk(2, -1)
        return self.w2(F.silu(b) * a)

class FeedForward(nn.Module):
    def __init__(self, d_model, p=0.1):
        super().__init__()
        self.net = nn.Sequential(SwiGLU(d_model, d_model * 2), nn.Dropout(p))
        self.scale = 1 / math.sqrt(2)
    def forward(self, x):
        return x + self.net(x) * self.scale

class LinformerBlock(nn.Module):
    def __init__(self, d_model, seq_len, heads, k, p=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Linformer(dim=d_model, seq_len=seq_len, depth=1, heads=heads, k=k, one_kv_head=True, share_kv=True, dropout=p)
        self.drop = DropPath(0.05)
        self.gamma1 = nn.Parameter(torch.ones(d_model) * 1e-4)
        self.ff = FeedForward(d_model, p)
    def forward(self, x):
        x = x + self.drop(self.gamma1 * self.attn(self.norm1(x)))
        x = self.ff(x)
        return x

class Head(nn.Module):
    def __init__(self, d_model, out_scale):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(d_model), nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 3))
        self.out_scale = out_scale
    def forward(self, h, mu, sig, gate=None):
        x = self.net(h) * self.out_scale * sig + mu
        if gate is not None:
            x = x * gate.view(-1, 1, 1)
        return x

class NCALayer(nn.Module):
    def __init__(self, d_state):
        super().__init__()
        self.perceive = nn.Conv1d(d_state, d_state * 2, 3, padding=1, groups=d_state)
        self.update = nn.Sequential(nn.Conv1d(d_state * 2, d_state, 3, padding=1), nn.LeakyReLU(0.2))
    def forward(self, x, steps=4):
        B, N, C = x.shape
        x = x.transpose(1, 2)
        for _ in range(steps):
            y = self.perceive(x)
            dx = self.update(y)
            x = x + dx
        return x.transpose(1, 2)

class AvatarUNRES(nn.Module):
    def __init__(self, pos, vel, acc, d_model=256, n_stem_blocks=4, n_attn_blocks=4, attn_heads=4, lin_k=32, p=0.1, nca_steps=4):
        super().__init__()
        gc.collect(); torch.cuda.empty_cache()
        self.N = pos.shape[1]
        self.d = d_model
        self.nca_steps = nca_steps
        cat = torch.cat([pos, vel, acc], 0)
        mu = cat.mean((0, 1), keepdim=True)
        sg = cat.std((0, 1), keepdim=True).clamp(1e-6)
        self.register_buffer('mu', mu)
        self.register_buffer('sig', sg)
        self.stem_p = Stem(3, d_model, n_stem_blocks, p)
        self.stem_v = Stem(3, d_model, n_stem_blocks, p)
        self.stem_a = Stem(3, d_model, n_stem_blocks, p)
        self.atom_emb = nn.Parameter(torch.randn(self.N, d_model))
        self.type_emb = nn.Parameter(torch.randn(3, d_model))
        self.global_emb = nn.Parameter(torch.randn(1, 1, d_model))
        seq_len = self.N * 3 + 1
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.encoder = nn.ModuleList([LinformerBlock(d_model, seq_len, attn_heads, lin_k, p) for _ in range(n_attn_blocks)])
        self.fno_blocks = nn.ModuleList([FNOBlock(d_model, seq_len) for _ in range(n_attn_blocks)])
        self.nca = NCALayer(d_model)
        self.gate_p0 = nn.Conv1d(277, 1, 3, padding=1)
        self.gate_v0 = nn.Conv1d(277, 1, 3, padding=1)
        self.gate_a0 = nn.Conv1d(277, 1, 3, padding=1)
        self.gate_p = nn.Linear(d_model, 1)
        self.gate_v = nn.Linear(d_model, 1)
        self.gate_a = nn.Linear(d_model, 1)
        out_scale = nn.Parameter(torch.ones(1, 1, 1))
        self.head_p = Head(d_model, out_scale)
        self.head_v = Head(d_model, out_scale)
        self.head_a = Head(d_model, out_scale)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _norm(self, x):
        return (x - self.mu) / self.sig

    def forward(self, pos, vel, acc):
        hp = self.stem_p(self._norm(pos))
        hv = self.stem_v(self._norm(vel))
        ha = self.stem_a(self._norm(acc))
        h_seq = torch.cat([hp, hv, ha], 1)

        B = h_seq.size(0)
        atom = self.atom_emb.unsqueeze(0).repeat(B, 3, 1, 1)
        atom[:,0] += self.type_emb[0]
        atom[:,1] += self.type_emb[1]
        atom[:,2] += self.type_emb[2]
        atom = atom.view(B, -1, self.d)
        gtok = self.global_emb.expand(B, -1, -1)
        h = torch.cat([gtok, h_seq + atom], 1)
        h = h + self.pos_emb

        for blk in self.encoder:
            h = blk(h)

        h_f = h
        for fb in self.fno_blocks:
            h_f = fb(h_f)

        h_gate = self.nca(h_f, steps=self.nca_steps)

        h_p = F.leaky_relu(self.gate_p0(h_f+h_gate).squeeze(1), 0.2)
        h_v = F.leaky_relu(self.gate_v0(h_f+h_gate).squeeze(1), 0.2)
        h_a = F.leaky_relu(self.gate_a0(h_f+h_gate).squeeze(1), 0.2)
        gate_p = self.gate_p(h_p)
        gate_v = self.gate_v(h_v)
        gate_a = self.gate_a(h_a)

        h = h[:,1:]
        hp_cat, hv_cat, ha_cat = h.split(self.N, 1)

        p_out = self.head_p(hp_cat, self.mu, self.sig, gate_p)
        v_out = self.head_v(hv_cat, self.mu, self.sig, gate_v)
        a_out = self.head_a(ha_cat, self.mu, self.sig, gate_a)
        return p_out, v_out, a_out
