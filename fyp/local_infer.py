"""Test-time local aggregation for AdaIR (the idea of TLC, Chu et al., ECCV 2022).

The model is trained on 128-px crops, but tested on whole 1080 x 720 images. Its whole-image operations then see far
more image than in training:
  - channel attention (Restormer's MDTA in every transformer block, and the channel cross-attention in the
    frequency blocks): the C x C attention is computed from all pixels of the feature map;
  - (optional) the frequency blocks themselves: FFT, the 50%-energy radius and the shape generator's channel
    statistics are all computed over the whole feature map.
Inside `local_inference(model, base)` these are computed on overlapping windows instead, `base` pixels wide at full
resolution (base / 2, / 4, / 8 at the lower levels), and the window outputs are averaged where they overlap. Every
convolution still sees the whole image, unlike cutting the image into tiles. With base >= the image size the model
is exactly the original one. Nothing is trained; the weights are unchanged.

  with local_inference(model, base=256, aflb=True):
      y = model(x)
"""
import contextlib
import types

import torch
import torch.nn.functional as F
from einops import rearrange

from net.model import Attention, Chanel_Cross_Attention

# downsampling factor of each part of AdaIR (fre1 sits on the latent, fre2 on decoder level 3, fre3 on level 2)
LEVEL = {"encoder_level1": 1, "decoder_level1": 1, "refinement": 1, "encoder_level2": 2, "decoder_level2": 2,
         "encoder_level3": 4, "decoder_level3": 4, "latent": 8, "fre1": 8, "fre2": 4, "fre3": 2}


def _starts(n, win, stride):
    if n <= win:
        return [0]
    s = list(range(0, n - win + 1, stride))
    if s[-1] != n - win:
        s.append(n - win)
    return s


def _chan_attn(q, k, v, heads, temperature):
    h, w = q.shape[-2:]
    q, k, v = (rearrange(t, 'b (head c) h w -> b head c (h w)', head=heads) for t in (q, k, v))
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    attn = (q @ k.transpose(-2, -1) * temperature).softmax(dim=-1)
    return rearrange(attn @ v, 'b head c (h w) -> b (head c) h w', head=heads, h=h, w=w)


def _local_attn(q, k, v, heads, temperature, win, stride):
    """Channel attention computed on overlapping win x win windows, outputs averaged over the overlaps."""
    b, _, h, w = q.shape
    if h <= win and w <= win:
        return _chan_attn(q, k, v, heads, temperature)
    wh, ww = min(win, h), min(win, w)
    out = torch.zeros_like(v)
    cnt = torch.zeros((1, 1, h, w), device=v.device, dtype=v.dtype)
    xs = _starts(w, win, stride)
    for y0 in _starts(h, win, stride):  # one row of windows at a time, batched
        crop = lambda t: torch.cat([t[..., y0:y0 + wh, x0:x0 + ww] for x0 in xs], 0)  # noqa: E731
        o = _chan_attn(crop(q), crop(k), crop(v), heads, temperature).split(b, 0)
        for x0, oi in zip(xs, o):
            out[..., y0:y0 + wh, x0:x0 + ww] += oi
            cnt[..., y0:y0 + wh, x0:x0 + ww] += 1
    return out / cnt


def _attention_forward(self, x):
    q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
    return self.project_out(_local_attn(q, k, v, self.num_heads, self.temperature, self._win, self._stride))


def _cross_forward(self, x, y):
    q = self.q_dwconv(self.q(x))
    k, v = self.kv_dwconv(self.kv(y)).chunk(2, dim=1)
    return self.project_out(_local_attn(q, k, v, self.num_head, self.temperature, self._win, self._stride))


def _aflb_forward(self, x, y):
    """The whole frequency block on overlapping windows of its feature map (and of the image resized to it)."""
    _, _, h, w = y.shape
    x = F.interpolate(x, (h, w), mode='bilinear')  # the block resizes x to y's size; on a crop that is a no-op
    full = type(self).forward
    win, stride = self._win, self._stride
    if h <= win and w <= win:
        return full(self, x, y)
    wh, ww = min(win, h), min(win, w)
    out = torch.zeros_like(y)
    cnt = torch.zeros((1, 1, h, w), device=y.device, dtype=y.dtype)
    for y0 in _starts(h, win, stride):
        for x0 in _starts(w, win, stride):
            sl = (..., slice(y0, y0 + wh), slice(x0, x0 + ww))
            out[sl] += full(self, x[sl], y[sl])
            cnt[sl] += 1
    return out / cnt


@contextlib.contextmanager
def local_inference(model, base=256, overlap=0.5, attention=True, aflb=False):
    """base: window width in input pixels (at full resolution); overlap: share of a window shared with the next."""
    patched = []
    for name, m in model.named_modules():
        top = name.split(".")[0]
        if top not in LEVEL:
            continue
        win = max(1, round(base / LEVEL[top]))
        stride = max(1, round(win * (1 - overlap)))
        if attention and isinstance(m, Attention):
            fwd = _attention_forward
        elif attention and isinstance(m, Chanel_Cross_Attention):
            fwd = _cross_forward
        elif aflb and name in ("fre1", "fre2", "fre3"):
            fwd = _aflb_forward
        else:
            continue
        m._win, m._stride = win, stride
        m.forward = types.MethodType(fwd, m)
        patched.append(m)
    try:
        yield model
    finally:
        for m in patched:
            del m.forward  # back to the class's forward
            del m._win, m._stride
