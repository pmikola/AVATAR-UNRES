import gc, torch, torch.nn as nn, torch.nn.functional as F
from linformer import Linformer
import math

# -- FNO block for spectral processing --
class FNOBlock(nn.Module):
    def __init__(self, d_model, seq_len, fourier_dim=None):
        super().__init__()
        self.seq_len = seq_len
        self.fourier_dim = fourier_dim or d_model
        # spectral conv: project to fourier_dim and back
        self.lin1 = nn.Linear(d_model, self.fourier_dim)
        self.lin2 = nn.Linear(self.fourier_dim, d_model)
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        x_f = torch.fft.rfft(x, dim=1)
        x_real = self.lin1(x_f.real)
        x_imag = self.lin1(x_f.imag)
        x_f = torch.complex(x_real, x_imag)
        x_if = torch.fft.irfft(x_f, n=self.seq_len, dim=1)
        x_if = self.lin2(x_if)
        return x + x_if * self.scale

class Residual1D(nn.Module):
    def __init__(self, c, k=3, p=0.1):
        super().__init__()
        pad = (k - 1) // 2
        self.c1 = nn.Conv1d(c, c, k, padding=pad)
        self.c2 = nn.Conv1d(c, c, k, padding=pad)
        self.n1 = nn.LayerNorm(c)
        self.act = nn.SiLU()
        self.dp = nn.Dropout(p)
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        y = self.c1(x)
        y = self.act(self.n1(y.transpose(1, 2))).transpose(1, 2)
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

class FeedForward(nn.Module):
    def __init__(self, d_model, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p)
        )
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        return x + self.net(x) * self.scale

class LinformerBlock(nn.Module):
    def __init__(self, d_model, seq_len, heads, k, p=0.1):
        super().__init__()
        self.attn = Linformer(dim=d_model, seq_len=seq_len, depth=1,
                              heads=heads, k=k, one_kv_head=True,
                              share_kv=True, dropout=p)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, p)

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) * (1 / math.sqrt(2))
        x = self.ff(x)
        return x

class Head(nn.Module):
    def __init__(self, d_model, out_scale):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3)
        )
        self.out_scale = out_scale

    def forward(self, h, mu, sig, gate=None):
        x = self.net(h) * self.out_scale * sig + mu
        if gate is not None:
            # gate: (B,1) -> broadcast over seq and coords
            x = x * gate.view(-1, 1, 1)
        return x

class AvatarUNRES(nn.Module):
    def __init__(self, pos, vel, acc,
                 d_model=256,
                 n_stem_blocks=4,
                 n_attn_blocks=8,
                 attn_heads=16,
                 lin_k=256,
                 p=0.1):
        super().__init__()
        gc.collect(); torch.cuda.empty_cache()

        self.N = pos.shape[1]
        self.d_model = d_model
        # normalize stats
        cat = torch.cat([pos, vel, acc], dim=0)
        mu = cat.mean((0,1), keepdim=True)
        sg = cat.std((0,1), keepdim=True).clamp(1e-6)
        self.register_buffer('mu', mu)
        self.register_buffer('sig', sg)

        # per-branch stems
        self.stem_p = Stem(3, d_model, n_stem_blocks, p)
        self.stem_v = Stem(3, d_model, n_stem_blocks, p)
        self.stem_a = Stem(3, d_model, n_stem_blocks, p)

        # global and atom/type embeddings
        self.atom_emb  = nn.Parameter(torch.randn(self.N, d_model))
        self.type_emb  = nn.Parameter(torch.randn(3, d_model))
        self.global_emb = nn.Parameter(torch.randn(1, 1, d_model))

        seq_len = self.N*3 + 1
        # main Linformer encoder
        self.encoder = nn.ModuleList([
            LinformerBlock(d_model, seq_len, attn_heads, lin_k, p)
            for _ in range(n_attn_blocks)
        ])

        # FNO blocks for gate computation
        self.fno_blocks = nn.ModuleList([
            FNOBlock(d_model, seq_len) for _ in range(n_attn_blocks)
        ])
        self.gate_p = nn.Linear(d_model, 1)
        self.gate_v = nn.Linear(d_model, 1)
        self.gate_a = nn.Linear(d_model, 1)

        # prediction heads
        out_scale = nn.Parameter(torch.ones(1,1,1))
        self.head_p = Head(d_model, out_scale)
        self.head_v = Head(d_model, out_scale)
        self.head_a = Head(d_model, out_scale)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _norm_in(self, x):
        return (x - self.mu) / self.sig

    def forward(self, pos, vel, acc):
        # extract features per branch
        hp = self.stem_p(self._norm_in(pos))
        hv = self.stem_v(self._norm_in(vel))
        ha = self.stem_a(self._norm_in(acc))
        # concatenate along sequence
        h_seq = torch.cat([hp, hv, ha], dim=1)  # (B,3N,d)

        # build embeddings
        B = h_seq.size(0)
        # atom+type embeddings broadcast
        atom = self.atom_emb.unsqueeze(0).repeat(B,3,1,1)
        atom[:,0] += self.type_emb[0]
        atom[:,1] += self.type_emb[1]
        atom[:,2] += self.type_emb[2]
        atom = atom.view(B, -1, self.d_model)  # (B,3N,d)
        # global token
        gtok = self.global_emb.expand(B, -1, -1)  # (B,1,d)

        # full sequence
        h = torch.cat([gtok, h_seq + atom], dim=1)  # (B,3N+1,d)

        # Linformer encoder
        for blk in self.encoder:
            h = blk(h)

        # parallel FNO for gates
        h_f = h
        for fblk in self.fno_blocks:
            h_f = fblk(h_f)
        g_repr = h_f[:,0]  # (B,d)
        gate_p = torch.sigmoid(self.gate_p(g_repr))
        gate_v = torch.sigmoid(self.gate_v(g_repr))
        gate_a = torch.sigmoid(self.gate_a(g_repr))

        # strip global token and split
        h = h[:,1:]  # (B,3N,d)
        hp_cat, hv_cat, ha_cat = h.split(self.N, dim=1)

        # apply gated heads
        p_out = self.head_p(hp_cat, self.mu, self.sig, gate_p)
        v_out = self.head_v(hv_cat, self.mu, self.sig, gate_v)
        a_out = self.head_a(ha_cat, self.mu, self.sig, gate_a)
        return p_out, v_out, a_out