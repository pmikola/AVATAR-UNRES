import gc, math, torch, torch.nn as nn, torch.nn.functional as F
from linformer import Linformer           # pip install linformer

class Residual1D(nn.Module):
    def __init__(self, c, k=3, p=0.1):
        super().__init__()
        pad = (k - 1) // 2
        self.c1 = nn.Conv1d(c, c, k, padding=pad)
        self.c2 = nn.Conv1d(c, c, k, padding=pad)
        self.n1 = nn.LayerNorm(c)
        self.act, self.dp = nn.SiLU(), nn.Dropout(p)

    def forward(self, x):
        y = self.c1(x)
        y = self.act(self.n1(y.transpose(1, 2))).transpose(1, 2)
        y = self.dp(y)
        y = self.c2(y)
        return x + y

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
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(p)
        )

    def forward(self, x):
        return x + self.net(x)

class LinformerBlock(nn.Module):
    def __init__(self, d_model, seq_len, heads, k, p=0.1):
        super().__init__()
        self.attn = Linformer(dim=d_model, seq_len=seq_len, depth=1,
                              heads=heads, k=k, one_kv_head=True,
                              share_kv=True, dropout=p)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = FeedForward(d_model, p)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = self.ff(self.norm2(x))
        return x

class Head(nn.Module):
    def __init__(self, d_model, out_scale):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//2),
            nn.SiLU(),
            nn.Linear(d_model//2, 3)
        )
        self.out_scale = out_scale

    def forward(self, h, mu, sig):
        return self.net(h) * self.out_scale * sig + mu

# ───────────────────────────── AvatarUNRES v3 ───────────────────────────────
class AvatarUNRES(nn.Module):
    def __init__(self, pos, vel, acc,
                 d_model=256,
                 n_stem_blocks=4,
                 n_attn_blocks=4,
                 attn_heads=8,
                 lin_k=128,
                 p=0.1):
        super().__init__()
        gc.collect(); torch.cuda.empty_cache()

        self.N = pos.shape[1]
        with torch.no_grad():
            cat = torch.cat([pos, vel, acc], dim=0)
            mu  = cat.mean((0,1), keepdim=True)
            sg  = cat.std ((0,1), keepdim=True).clamp(1e-6)
        self.register_buffer('mu',  mu)
        self.register_buffer('sig', sg)

        self.stem_p = Stem(3, d_model, n_stem_blocks, p)
        self.stem_v = Stem(3, d_model, n_stem_blocks, p)
        self.stem_a = Stem(3, d_model, n_stem_blocks, p)

        self.atom_emb  = nn.Parameter(torch.randn(self.N, d_model))
        self.type_emb  = nn.Parameter(torch.randn(3,   d_model))

        seq_len = self.N * 3
        self.encoder = nn.ModuleList([
            LinformerBlock(d_model, seq_len, attn_heads, lin_k, p)
            for _ in range(n_attn_blocks)
        ])

        out_scale = nn.Parameter(torch.ones(1,1,1))
        self.head_p = Head(d_model, out_scale)
        self.head_v = Head(d_model, out_scale)
        self.head_a = Head(d_model, out_scale)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def _norm_in(self, x):
        return (x - self.mu) / self.sig

    def forward(self, pos, vel, acc):
        hp = self.stem_p(self._norm_in(pos))
        hv = self.stem_v(self._norm_in(vel))
        ha = self.stem_a(self._norm_in(acc))

        h = torch.cat([hp, hv, ha], dim=1)

        B = h.size(0)
        atom_embeddings = self.atom_emb.unsqueeze(0).repeat(B, 3, 1, 1)
        atom_embeddings[:,1] += self.type_emb[1]
        atom_embeddings[:,2] += self.type_emb[2]
        atom_embeddings[:,0] += self.type_emb[0]
        atom_embeddings = atom_embeddings.view(B, -1, self.atom_emb.size(-1))
        h = h + atom_embeddings

        for blk in self.encoder:
            h = blk(h)

        hp_cat, hv_cat, ha_cat = h.split(self.N, dim=1)

        p_out = self.head_p(hp_cat, self.mu, self.sig)
        v_out = self.head_v(hv_cat, self.mu, self.sig)
        a_out = self.head_a(ha_cat, self.mu, self.sig)
        return p_out, v_out, a_out
