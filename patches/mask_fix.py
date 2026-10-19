"""Runtime fix for the frequency mask in AdaIR's FreModule (AFLB).

The bug
-------
The released FreModule.fft builds the low-frequency mask as

    threshold = sigmoid(rate_conv(GAP(x)))           # in (0, 1)
    h_ = (h // 128 * threshold).int()
    mask[..., h//2-h_ : h//2+h_, w//2-w_ : w//2+w_] = 1

The three AFLBs run at H/8, H/4 and H/2 of the input, so for 128 training
crops the maps are 16/32/64 px and h // 128 == 0. Even on 720p inputs it is
at most 2-5, and int(k * sigmoid) rounds down to ~0. The mask is empty, the low
branch is all zeros and the high branch is the full spectrum. On top of that,
.int() plus slice assignment has no gradient, so rate_conv never trains.

The fix
-------
Only the mask changes; the rest of FreModule.fft is the original code.

    cutoff = sigmoid(rate_conv(GAP(x)))               # (B, 2), fraction of Nyquist per axis
    mask   = sigmoid((cutoff_h - |u|) / tau) * sigmoid((cutoff_w - |v|) / tau)

u, v are the centred frequency coordinates normalised to [-1, 1] (0 = DC,
1 = Nyquist), so the cutoff means the same thing on a 16 px map and a 640 px
map (resolution-independent). The soft edge makes the mask differentiable, so
rate_conv learns the cutoff. tau = 0.05 keeps coverage within 0.4% of the
cutoff and d(coverage)/d(cutoff) within 0.94-1.01 down to 16x16 maps.

No parameters are added, so released checkpoints load unchanged.

Usage
-----
net/model.py is not modified. The patch swaps the class of each FreModule
instance in a given model, so a patched and an original AdaIR can live in the
same process (needed for benchmarking against the released model).

    from net.model import AdaIR
    from patches import mask_fix

    model = mask_fix.apply(AdaIR(decoder=True))       # fixed
    mask_fix.remove(model)                            # back to original AdaIR
"""
import torch
import torch.nn.functional as F

from net.model import FreModule

DEFAULT_TAU = 0.05


def soft_rect_mask(cutoff, h, w, tau=DEFAULT_TAU):
    """Soft low-pass rectangle in the centred (FreModule.shift) layout.

    cutoff: (B, 2) tensor in (0, 1), [height cutoff, width cutoff].
    Returns a (B, 1, h, w) mask, broadcast over channels like the original.
    """
    u = (torch.arange(h, device=cutoff.device, dtype=cutoff.dtype) - h // 2) / (h / 2)
    v = (torch.arange(w, device=cutoff.device, dtype=cutoff.dtype) - w // 2) / (w / 2)
    mask_h = torch.sigmoid((cutoff[:, 0, None] - u.abs()) / tau)  # (B, h)
    mask_w = torch.sigmoid((cutoff[:, 1, None] - v.abs()) / tau)  # (B, w)
    return (mask_h[:, :, None] * mask_w[:, None, :])[:, None]


class FreModuleMaskFix(FreModule):
    """FreModule with the soft, resolution-independent mask. Same parameters."""

    mask_tau = DEFAULT_TAU

    def fft(self, x, n=None):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        h, w = x.shape[-2:]
        cutoff = F.adaptive_avg_pool2d(x, 1)
        cutoff = self.rate_conv(cutoff).sigmoid().flatten(1)

        mask = soft_rect_mask(cutoff, h, w, self.mask_tau)

        # kept for logging / mask visualisation
        self.last_cutoff = cutoff.detach()
        self.last_mask_area = mask.detach().mean(dim=(1, 2, 3))

        fft = torch.fft.fft2(x, norm='forward', dim=(-2,-1))
        fft = self.shift(fft)

        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2,-1))
        high = torch.abs(high)

        fft_low = fft * mask

        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2,-1))
        low = torch.abs(low)

        return high, low


def apply(model, tau=DEFAULT_TAU):
    """Switch every FreModule in model to the fixed mask (in place). Returns model."""
    count = 0
    for m in model.modules():
        if isinstance(m, FreModule):
            m.__class__ = FreModuleMaskFix
            m.mask_tau = tau
            count += 1
    if count == 0:
        raise ValueError("mask_fix.apply: no FreModule found in model")
    return model


def remove(model):
    """Undo apply(): every patched FreModule goes back to the original class."""
    for m in model.modules():
        if isinstance(m, FreModuleMaskFix):
            m.__class__ = FreModule
            for attr in ('mask_tau', 'last_cutoff', 'last_mask_area'):
                m.__dict__.pop(attr, None)
    return model
