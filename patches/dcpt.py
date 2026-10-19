"""DCPT: degradation-classification pre-training for AdaIR.

Method: Hu et al., "Universal Image Restoration Pre-training via Degradation Classification",
ICLR 2025 (arXiv 2501.15510). This is our own implementation, written from the paper and from
reading the official code (github.com/MILab-PKU/dcpt, which has no license file, so no code is
copied). Where the paper text and the official code differ, we follow the code:

    each iteration, one backward on  L = pix_weight * L_pix + L_cls
      generation      x_hat = net(gt),  L_pix = L1(x_hat, gt)        (clean in -> clean out)
      classification  F = decoder features of net(lq),  logits = classifier(F),
                      L_cls = focal loss (gamma 2) vs the degradation label
    optimisers        net: AdamW lr 2e-4, wd 0;  classifier: AdamW lr 5e-5, wd 1e-4 (betas 0.9, 0.999)
    (paper text: the two stages alternate and generation uses a conv on the deepest feature)

Features: the outputs of decoder_level3 (4C, H/4), decoder_level2 (2C, H/2) and decoder_level1
(2C, H), the hook points of the official Restormer config (AdaIR uses the same backbone), fed to
the classifier finest first. All three AFLBs sit upstream of these, so the classification loss also
trains the frequency blocks. The classification pass stops right after decoder_level1.

Classifier (paper Fig. 2a / official "NoImg" decoder): per level, add softmax(w)_i * F_i, two ResNet
bottleneck blocks (1x1 -> 3x3 -> 1x1, width 2x, LayerNorm instead of BatchNorm), then 1x1 conv to
the next level's width + 2x2 max-pool + ReLU; two more bottlenecks, global average pool, linear.

After pre-training: fine-tune `dcpt.net` (all weights, strict load) for restoration; the classifier
is dropped.

Usage:
    model = shaped_bands.apply(AdaIR(decoder=True))
    dcpt = DCPT(model, num_classes=5).cuda()
    opt_net, opt_cls = dcpt.optimizers()
    loss, logs = dcpt(lq, gt, labels); loss.backward(); opt_net.step(); opt_cls.step()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

HOOKS = ("decoder_level1", "decoder_level2", "decoder_level3")  # finest first

# AdaIR's de_id (utils/dataset_utils.py): 0-2 denoise sigma 15/25/50, 3 derain, 4 dehaze,
# 5 deblur, 6 enhance. Default: one class per task, noise levels merged (as DCPT's 5D setting).
ADAIR_DE_ID_TO_CLASS = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4}


def labels_from_de_id(de_id, mapping=None):
    mapping = mapping or ADAIR_DE_ID_TO_CLASS
    return torch.as_tensor([mapping[int(d)] for d in de_id], dtype=torch.long)


def focal_loss(logits, target, gamma=2.0):
    logp = F.log_softmax(logits.float(), dim=-1)
    logp_t = logp.gather(1, target[:, None])[:, 0]
    return (-(1 - logp_t.exp()) ** gamma * logp_t).mean()


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels of an NCHW tensor."""

    def __init__(self, ch, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias = nn.Parameter(torch.zeros(ch))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight[:, None, None] + self.bias[:, None, None]


class Bottleneck(nn.Module):
    def __init__(self, ch, mid):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, mid, 1, bias=False)
        self.norm1 = ChannelLayerNorm(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.norm2 = ChannelLayerNorm(mid)
        self.conv3 = nn.Conv2d(mid, ch, 1, bias=False)
        self.norm3 = ChannelLayerNorm(ch)
        for conv in (self.conv1, self.conv2, self.conv3):
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        return F.relu(out + x)


class DegradationClassifier(nn.Module):
    def __init__(self, feature_dims, num_classes, blocks=2):
        super().__init__()
        dims = list(feature_dims)
        self.stages = nn.ModuleList(
            nn.Sequential(*[Bottleneck(d, 2 * d) for _ in range(blocks)]) for d in dims)
        self.downs = nn.ModuleList(
            nn.Sequential(nn.Conv2d(d, dims[i + 1] if i + 1 < len(dims) else d, 1, bias=False),
                          nn.MaxPool2d(2, 2), nn.ReLU()) for i, d in enumerate(dims))
        self.last = nn.Sequential(*[Bottleneck(dims[-1], 2 * dims[-1]) for _ in range(blocks)])
        self.mix = nn.Parameter(torch.ones(len(dims)))
        self.fc = nn.Linear(dims[-1], num_classes)

    def forward(self, feats):
        """feats: list of NCHW features, finest first, each level half the size of the previous."""
        w = self.mix.softmax(0)
        h = 0
        for i, f in enumerate(feats):
            h = self.downs[i](self.stages[i](h + w[i] * f))
        return self.fc(self.last(h).mean((2, 3)))


class _StopForward(Exception):
    pass


class DCPT(nn.Module):
    def __init__(self, net, num_classes, gamma=2.0, pix_weight=1.0):
        super().__init__()
        self.net = net
        self.gamma = gamma
        self.pix_weight = pix_weight
        dims = [getattr(net, name)[0].ffn.project_in.in_channels for name in HOOKS]
        self.classifier = DegradationClassifier(dims, num_classes)

    def features(self, lq):
        """Decoder features of net(lq), finest first. Stops the forward after decoder_level1."""
        feats = {}

        def hook(name):
            def fn(module, inputs, output):
                feats[name] = output
                if name == HOOKS[0]:
                    raise _StopForward
            return fn

        handles = [getattr(self.net, name).register_forward_hook(hook(name)) for name in HOOKS]
        try:
            self.net(lq)
        except _StopForward:
            pass
        finally:
            for h in handles:
                h.remove()
        return [feats[name] for name in HOOKS]

    def classify(self, lq):
        return self.classifier(self.features(lq))

    def forward(self, lq, gt, labels):
        """Returns (loss, logs). Generation on the clean image, classification on the degraded one."""
        x_hat = self.net(gt)
        l_pix = F.l1_loss(x_hat, gt)
        logits = self.classify(lq)
        l_cls = focal_loss(logits, labels, self.gamma)
        acc = (logits.argmax(1) == labels).float().mean()
        loss = self.pix_weight * l_pix + l_cls
        return loss, {"l_pix": l_pix.detach(), "l_cls": l_cls.detach(), "acc": acc.detach()}

    def optimizers(self, lr_net=2e-4, lr_cls=5e-5, wd_cls=1e-4):
        opt_net = torch.optim.AdamW(self.net.parameters(), lr=lr_net, weight_decay=0, betas=(0.9, 0.999))
        opt_cls = torch.optim.AdamW(self.classifier.parameters(), lr=lr_cls, weight_decay=wd_cls,
                                    betas=(0.9, 0.999))
        return opt_net, opt_cls
