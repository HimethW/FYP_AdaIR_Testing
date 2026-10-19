"""Mixed-precision helper for the original / mask-fix AFLBs.

Under bf16 autocast the released FreModule fails ("Unsupported dtype BFloat16": torch.fft has
no bf16 kernels), and fp16 FFTs only work for power-of-2 sizes. fp32_frequency(model) makes the
frequency step (conv on I, FFT, mask, IFFT) of every FreModule / FreModuleMaskFix run in fp32,
so baselines can be trained and tested with autocast like the shaped-band model
(FreModuleBands already keeps its frequency path in fp32 by itself). No parameters change.
"""
import torch

from net.model import FreModule
from patches.mask_fix import FreModuleMaskFix


class _FP32FFT:
    def fft(self, x, *args, **kwargs):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().fft(x.float(), *args, **kwargs)


class FreModuleFP32(_FP32FFT, FreModule):
    pass


class FreModuleMaskFixFP32(_FP32FFT, FreModuleMaskFix):
    pass


def fp32_frequency(model):
    """In place; returns model."""
    for m in model.modules():
        if type(m) is FreModule:
            m.__class__ = FreModuleFP32
        elif type(m) is FreModuleMaskFix:
            m.__class__ = FreModuleMaskFixFP32
    return model
