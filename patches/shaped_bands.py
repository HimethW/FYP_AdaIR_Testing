"""Shaped frequency bands + band self-attention for AdaIR's AFLB (FMiM* and BSA).

Replaces each FreModule (AFLB) in an AdaIR model with FreModuleBands:

    I --conv3x3--> P --FFT--> F
    P --SGB--> 8 shape params --> 4 soft masks: dot, star, lobe, rest (sum to 1)
    X_b = CA(q = |IFFT(M_b * F)|, kv = X)          one cross-attention per band
    X'_b = BSA(X_dot, X_star, X_lobe, X_rest)       per-pixel 4x4 self-attention over bands
    X_m = conv1x1(sum_b X'_b)
    out = CA(q = X, kv = X_m) * para1 + X * para2

The two-band H-L / L-H modulation (FreRefine) is removed: with 4 bands the mixing
recipe is learned by BSA instead.

Coordinates: u (rows) and v (columns) are centred frequencies normalised to
[-1, 1] (0 = DC, 1 = Nyquist), as in mask_fix.py, so shapes mean the same at
every AFLB resolution. Edges are soft: sigmoid(signed distance / tau).

The 4 bands (masks M_b, in this priority, so they partition the spectrum):
    dot   D: disc |f| < r, r = r50 * exp(s0), r50 = radius holding 50% of the
             non-DC energy of this image's spectrum (detached), s0 learned
    star  S: rotated superellipse (|x'|/a)^p + (|y'|/b)^p <= 1,
             a = size * aspect, b = size / aspect, angle theta_s
    lobe  L: side lobes |v'| >= vertex + curv * u'^2, angle theta_l
    rest   : everything else
    M_dot = D, M_star = S(1-D), M_lobe = L(1-D)(1-S), M_rest = (1-D)(1-S)(1-L)

Shape params are not bounded yet (exp / identity, with numeric safety clamps only).

Usage:
    from net.model import AdaIR
    from patches import shaped_bands
    model = shaped_bands.apply(AdaIR(decoder=True))
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.model import Chanel_Cross_Attention, LayerNorm

DEFAULT_TAU = 0.05
BANDS = ("dot", "star", "lobe", "rest")
PARAM_NAMES = ("dot_scale", "star_theta", "star_p", "star_size", "star_aspect",
               "lobe_vertex", "lobe_curv", "lobe_theta")

# Starting shapes (fractions of Nyquist). The SGB's base is set so that every
# image gets exactly these shapes before training.
INIT_SHAPE = dict(dot_scale=1.0, star_theta=0.0, star_p=0.5, star_size=0.75, star_aspect=1.0,
                  lobe_vertex=0.6, lobe_curv=0.4, lobe_theta=0.0)
_LOG_PARAMS = ("dot_scale", "star_p", "star_size", "star_aspect", "lobe_vertex", "lobe_curv")


def init_raw():
    return torch.tensor([math.log(INIT_SHAPE[n]) if n in _LOG_PARAMS else INIT_SHAPE[n] for n in PARAM_NAMES])


def decode_params(raw, r50):
    """raw: (B, 8) from the SGB; r50: (B,) 50%-energy radius. Returns dict of (B,) tensors."""
    p = dict(zip(PARAM_NAMES, raw.unbind(1)))
    out = {n: p[n].exp() if n in _LOG_PARAMS else p[n] for n in PARAM_NAMES}
    # numeric safety only (keeps powers / divisions finite), not shape bounds
    out["star_p"] = out["star_p"].clamp(0.1, 10.0)
    out["star_size"] = out["star_size"].clamp(min=1e-3)
    out["star_aspect"] = out["star_aspect"].clamp(1e-2, 1e2)
    out["dot_r"] = r50 * out.pop("dot_scale")
    return out


def centred_grid(h, w, device, dtype):
    """u (rows), v (cols) in the FreModule.shift layout: index h//2 is DC."""
    u = (torch.arange(h, device=device, dtype=dtype) - h // 2) / (h / 2)
    v = (torch.arange(w, device=device, dtype=dtype) - w // 2) / (w / 2)
    return u[:, None].expand(h, w), v[None, :].expand(h, w)


def band_masks(params, h, w, tau=DEFAULT_TAU):
    """Soft masks for the 4 bands, (B, 4, h, w), centred layout. They sum to 1."""
    ref = params["dot_r"]
    u, v = centred_grid(h, w, ref.device, ref.dtype)
    u, v = u[None], v[None]
    col = lambda t: t[:, None, None]  # noqa: E731

    dot = torch.sigmoid((col(params["dot_r"]) - torch.sqrt(u ** 2 + v ** 2)) / tau)

    th, size, aspect, pw = (col(params[k]) for k in ("star_theta", "star_size", "star_aspect", "star_p"))
    x = torch.cos(th) * v + torch.sin(th) * u
    y = -torch.sin(th) * v + torch.cos(th) * u
    eps = 1e-4
    g = (((x.abs() + eps) / (size * aspect)) ** pw + ((y.abs() + eps) / (size / aspect)) ** pw) ** (1 / pw)
    star = torch.sigmoid(size * (1 - g) / tau)  # size * g ~ distance from centre along the star

    th, vertex, curv = (col(params[k]) for k in ("lobe_theta", "lobe_vertex", "lobe_curv"))
    ul = torch.cos(th) * u - torch.sin(th) * v
    vl = torch.sin(th) * u + torch.cos(th) * v
    lobe = torch.sigmoid((vl.abs() - vertex - curv * ul ** 2) / tau)

    m_dot = dot
    m_star = star * (1 - dot)
    m_lobe = lobe * (1 - dot) * (1 - star)
    m_rest = (1 - dot) * (1 - star) * (1 - lobe)
    return point_symmetrize(torch.stack([m_dot, m_star, m_lobe, m_rest], 1))


def point_symmetrize(m):
    """Average each mask with its DFT mirror (k -> -k), so a real input gives real bands.

    The shapes are point-symmetric in continuous coordinates, but on an even grid the
    Nyquist row/column (u = -1) has no +1 partner, so rotated shapes are slightly off.
    Linear, so the masks still sum to 1.
    """
    h, w = m.shape[-2:]
    mirror = m.flip(-2)
    if h % 2 == 0:
        mirror = mirror.roll(1, -2)
    mirror = mirror.flip(-1)
    if w % 2 == 0:
        mirror = mirror.roll(1, -1)
    return 0.5 * (m + mirror)


_ORDER_CACHE = {}


def energy_radius(spec_centred, share=0.5):
    """Radius (fraction of Nyquist) holding `share` of the non-DC energy, per sample. Detached."""
    with torch.no_grad():
        b, _, h, w = spec_centred.shape
        energy = spec_centred.abs().float().pow(2).mean(1).flatten(1)  # (B, h*w)
        key = (h, w, energy.device)
        if key not in _ORDER_CACHE:
            u, v = centred_grid(h, w, energy.device, torch.float32)
            rho = torch.sqrt(u ** 2 + v ** 2).flatten()
            order = torch.argsort(rho)
            _ORDER_CACHE[key] = (order, rho[order])
        order, rho_sorted = _ORDER_CACHE[key]
        e = energy[:, order]
        e[:, 0] = 0  # DC is the smallest radius
        cum = e.cumsum(1)
        cum = cum / cum[:, -1:].clamp(min=1e-12)
        idx = torch.searchsorted(cum, torch.full((b, 1), share, device=cum.device)).clamp(max=h * w - 1)
        return rho_sorted[idx[:, 0]].to(spec_centred.real.dtype)


class ShapeGenerator(nn.Module):
    """SGB: [GAP(P), STD(P)] -> 1x1 -> GELU -> 1x1 -> + learned base -> 8 raw shape params."""

    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or max(dim // 8, 8)
        self.body = nn.Sequential(
            nn.Conv2d(2 * dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, len(PARAM_NAMES), 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        self.base = nn.Parameter(init_raw())

    def forward(self, p):
        stats = torch.cat([p.mean((2, 3), keepdim=True), p.std((2, 3), keepdim=True)], 1)
        return self.base + self.body(stats).flatten(1)


class BandSelfAttention(nn.Module):
    """BSA: per-pixel self-attention over the bands (n x n at every pixel, per head)."""

    def __init__(self, dim, n_bands=len(BANDS), num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.band_emb = nn.Parameter(torch.randn(n_bands, dim, 1, 1) * 0.02)
        self.norm = LayerNorm(dim, 'WithBias')
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)  # starts as identity (residual only)
        nn.init.zeros_(self.proj.bias)

    def forward(self, bands):
        x = torch.stack(bands, 1)  # (B, n, C, H, W)
        b, n, c, h, w = x.shape
        t = self.norm((x + self.band_emb).flatten(0, 1))
        d = c // self.num_heads
        q, k, v = (f(t).view(b, n, self.num_heads, d, h * w) for f in (self.q, self.k, self.v))
        attn = torch.einsum('bihdp,bjhdp->bhpij', q, k) * d ** -0.5
        attn = attn.softmax(-1)  # over the bands
        out = torch.einsum('bhpij,bjhdp->bihdp', attn, v).reshape(b * n, c, h, w)
        out = self.proj(out).view(b, n, c, h, w) + x
        self.last_attn = attn.detach().mean(dim=(1, 2))  # (B, n, n): mean mixing recipe
        return list(out.unbind(1))


class FreModuleBands(nn.Module):
    """AFLB with 4 shaped bands (FMiM*) and band self-attention (BSA)."""

    def __init__(self, dim, num_heads, bias, in_dim=3, tau=DEFAULT_TAU, bsa_heads=4):
        super().__init__()
        self.tau = tau
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.sgb = ShapeGenerator(dim)
        self.band_ca = nn.ModuleList([Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias) for _ in BANDS])
        self.bsa = BandSelfAttention(dim, len(BANDS), bsa_heads)
        self.merge = nn.Conv2d(dim, dim, kernel_size=1)
        self.channel_cross_agg = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x, y):
        _, _, H, W = y.size()
        x = F.interpolate(x, (H, W), mode='bilinear')

        bands = self.split(x)
        bands = [ca(band, y) for ca, band in zip(self.band_ca, bands)]
        bands = self.bsa(bands)
        agg = self.merge(sum(bands))
        out = self.channel_cross_agg(y, agg)

        return out * self.para1 + y * self.para2

    def split(self, x):
        """|IFFT(M_b * F)| for the 4 bands, each (B, C, h, w)."""
        p = self.conv1(x)
        h, w = p.shape[-2:]
        spec = torch.fft.fft2(p, norm='forward', dim=(-2, -1))
        spec = torch.roll(spec, shifts=(h // 2, w // 2), dims=(2, 3))

        r50 = energy_radius(spec)
        params = decode_params(self.sgb(p), r50)
        masks = band_masks(params, h, w, self.tau)  # (B, 4, h, w)

        # kept for logging / mask plots
        self.last_params = {k: t.detach() for k, t in params.items()}
        self.last_r50 = r50
        self.last_band_area = masks.detach().mean(dim=(2, 3))  # (B, 4)

        banded = spec[:, None] * masks[:, :, None]  # (B, 4, C, h, w)
        banded = torch.roll(banded, shifts=(-(h // 2), -(w // 2)), dims=(3, 4))
        bands = torch.fft.ifft2(banded, norm='forward', dim=(-2, -1)).abs()
        return list(bands.unbind(1))


AFLBS = ("fre1", "fre2", "fre3")


def apply(model, tau=DEFAULT_TAU, bsa_heads=4):
    """Replace fre1/fre2/fre3 of an AdaIR model with FreModuleBands (new, untrained). Returns model."""
    for name in AFLBS:
        old = getattr(model, name, None)
        if old is None:
            raise ValueError(f"shaped_bands.apply: model has no {name} (built with decoder=False?)")
        new = FreModuleBands(dim=old.conv1.out_channels, num_heads=old.channel_cross_agg.num_head,
                             bias=old.channel_cross_agg.q.bias is not None, in_dim=old.conv1.in_channels,
                             tau=tau, bsa_heads=bsa_heads)
        setattr(model, name, new.to(old.para1.device))
    return model
