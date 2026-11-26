"""Learn corrections to a frozen native TRM answer at a fixed fork."""

import torch
import torch.nn.functional as F
from torch import nn

from .model import Reasoner


class TRMRefiner(Reasoner):
    """Use the same frozen context and output interface for both update rules."""

    def __init__(self, architecture, native_width, method):
        if method not in {"unrestricted", "stateless", "recurrent"}:
            raise ValueError("Expected a direct, stateless, or recurrent refiner")
        super().__init__(architecture, 11, 11, method)
        # Both refiners start at the native answer, with no trainable initializer.
        del self.initial
        self.native_context = nn.Linear(2 * native_width + 11, architecture.width)
        self.gain = nn.Parameter(torch.zeros(()))

    def prepare(self, inputs, z_h, z_l, logits):
        length = inputs.shape[1]
        # Native registers precede the cells. Native continuation keeps the full carry.
        h = z_h[:, -length:].detach().float()
        l = z_l[:, -length:].detach().float()
        base = logits.detach().float()
        context = torch.cat(
            (F.layer_norm(h, (h.shape[-1],)), F.layer_norm(l, (l.shape[-1],)),
             F.layer_norm(base, (base.shape[-1],))), dim=-1,
        )
        context = self.encode(inputs) + self.native_context(context)
        delta = base.new_zeros((*inputs.shape, self.a.latent))
        return context, base, delta, self.zero_memory(context)

    def readout(self, base, delta):
        # Decode a change, so a zero correction leaves every native logit intact.
        return base + (2 * (delta @ self.codes.T) / self.a.temperature)

    def advance(self, context, delta, workspace, *, train):
        proposal, workspace = self.step(context, delta, workspace, train=train)
        delta_next = delta + self.gain * (proposal - delta)
        return (delta_next, workspace) if train else (delta_next.detach(), workspace)

    def training_loss(self, inputs, z_h, z_l, logits, labels, loops):
        context, base, delta, workspace = self.prepare(inputs, z_h, z_l, logits)
        losses = []
        for _ in range(loops):
            delta, workspace = self.advance(context, delta, workspace, train=True)
            losses.append(F.cross_entropy(self.readout(base, delta).flatten(0, 1), labels.flatten()))
        return torch.stack(losses).mean()
