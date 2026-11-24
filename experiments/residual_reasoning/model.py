"""Rule-free structured predictor. The only arguments to inference are x and model state."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from .operators import bounded_residual_mix, residual_mixing_force


METHODS = (
    "identity",
    "direct",
    "fixed",
    "random",
    "fixed_learned",
    "stateless",
    "recurrent",
    "diagonal",
    "state_side",
    "gated",
    "residual_map",
    "snr",
    "adam_energy",
    "free_energy",
    "unrestricted",
    "unrestricted_free",
)
HISTORY = {
    "recurrent",
    "diagonal",
    "state_side",
    "gated",
    "residual_map",
    "unrestricted",
    "unrestricted_free",
}
NO_CONTROLLER = {
    "identity",
    "direct",
    "fixed",
    "random",
    "snr",
    "adam_energy",
    "free_energy",
}
NO_FIELD = {"unrestricted", "unrestricted_free"}


@dataclass(frozen=True)
class Architecture:
    latent: int = 12
    width: int = 64
    heads: int = 4
    layers: int = 2
    rank: int = 2
    alpha: float = 0.5
    m: float = 1.0
    b: float = 2.0
    damping: float = 0.1
    temperature: float = 0.3
    max_length: int = 256
    interface_seed: int = 812

    def __post_init__(self):
        if (
            min(
                self.latent,
                self.width,
                self.heads,
                self.layers,
                self.rank,
                self.max_length,
            )
            < 1
        ):
            raise ValueError("All model dimensions must be positive")
        if self.width % self.heads or self.width % 2:
            raise ValueError("width must be even and divisible by heads")
        if not (
            self.m > 0
            and self.b >= 0
            and self.alpha > 0
            and self.temperature > 0
            and self.damping > 0
        ) or not all(
            math.isfinite(x)
            for x in (self.m, self.b, self.alpha, self.temperature, self.damping)
        ):
            raise ValueError("Invalid metric, step, damping or temperature")


class Block(nn.Module):
    """Explicit attention supports double backward on CPU and CUDA."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )

    def forward(self, h):
        batch, length, width = h.shape
        q, k, v = (
            self.qkv(self.norm(h)).view(batch, length, 3, self.heads, -1).unbind(2)
        )
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        weights = ((q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])).softmax(-1)
        h = h + self.out((weights @ v).transpose(1, 2).reshape(batch, length, width))
        return h + self.ff(h)


def stack(a):
    return nn.Sequential(
        *(Block(a.width, a.heads) for _ in range(a.layers)), nn.LayerNorm(a.width)
    )


class Field(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.candidate = nn.Linear(a.latent, a.width)
        self.trunk = stack(a)
        self.geometry = nn.Linear(a.width, a.latent)
        self.score = nn.Linear(a.width, 1)

    def forward(self, xh, q):
        h = self.trunk(xh + self.candidate(q))
        s = self.score(h.mean(1)) * math.sqrt(q.shape[1])
        return torch.cat((self.geometry(h).flatten(1), s), dim=1)


def vjp(c, q, cotangent, train, retain=False):
    return torch.autograd.grad(
        c, q, cotangent, create_graph=train, retain_graph=train or retain
    )[0]


class Reasoner(nn.Module):
    def __init__(self, a, vin, vout, method):
        super().__init__()
        if method not in METHODS or a.latent < vout:
            raise ValueError(
                "Unknown method or latent dimension smaller than output vocabulary"
            )
        self.a, self.method = a, method
        # A fixed orthonormal codeword per token, distance sqrt(2) between every pair.
        gen = torch.Generator().manual_seed(a.interface_seed)
        codes = torch.linalg.qr(
            torch.randn(a.latent, vout, generator=gen)
        ).Q.T.contiguous()
        self.register_buffer("codes", codes)
        self.input = nn.Embedding(vin, a.width)
        t = torch.arange(a.max_length)[:, None]
        angles = t * torch.exp(
            torch.arange(0, a.width, 2) * (-math.log(10000) / a.width)
        )
        self.register_buffer(
            "position", torch.stack((angles.sin(), angles.cos()), -1).flatten(1)
        )
        self.initial = nn.Sequential(stack(a), nn.Linear(a.width, a.latent))
        if method not in NO_FIELD:
            self.field = Field(a)
        if method in {"fixed", "fixed_learned"}:
            u = torch.randn(a.max_length * a.latent + 1, a.rank, generator=gen)
            if method == "fixed":
                self.register_buffer("fixed", u)
            else:
                self.fixed = nn.Parameter(u)
        if method in NO_CONTROLLER or method == "fixed_learned":
            return
        self.control_in = nn.Linear(a.latent * 2 + 1, a.width)
        self.controller = stack(a)
        if method in HISTORY:
            self.memory = nn.Linear(a.width, a.width)
        if method in {
            "stateless",
            "recurrent",
            "residual_map",
            "state_side",
            "diagonal",
        }:
            self.factor = nn.Linear(a.width, a.latent * a.rank)
            if method != "state_side":
                self.score_factor = nn.Linear(a.width, a.rank)
        if method == "residual_map":
            self.left = nn.Linear(a.width, a.latent * a.rank)
            self.direct_scale = nn.Linear(a.width, a.latent)
        if method in {"gated", "unrestricted", "unrestricted_free"}:
            self.proposal = nn.Linear(a.width, a.latent)
        if method == "unrestricted_free":
            self.readout = nn.Linear(a.latent, vout)

    def encode(self, x):
        if x.shape[1] > self.a.max_length:
            raise ValueError("Input length exceeds configured max_length")
        return self.input(x) + self.position[: x.shape[1]]

    def zero_memory(self, xh):
        if self.method == "adam_energy":
            return xh.new_zeros((*xh.shape[:2], 2 * self.a.latent + 1))
        return torch.zeros_like(xh)

    def start(self, xh):
        return self.initial(xh), self.zero_memory(xh)

    def decode(self, q):
        if self.method == "unrestricted_free":
            return self.readout(q)
        # The omitted -||q||^2 term is shared by every vocabulary entry.
        return (2 * q @ self.codes.T - self.codes.square().sum(-1)) / self.a.temperature

    def features(self, xh, q, w, c):
        g = c[:, :-1].reshape_as(q)
        score = (
            (c[:, -1:] / math.sqrt(q.shape[1])).unsqueeze(1).expand(-1, q.shape[1], -1)
        )
        h = xh + self.control_in(torch.cat((q, g, score), -1))
        if self.method in HISTORY:
            h = h + w
        return self.controller(h)

    def step(self, xh, q, w, *, train, intervention=None):
        """Return (q_next, w_next); preserve the outer graph only for training."""
        q = q if train else q.detach()
        if not q.requires_grad:
            q = q.requires_grad_(True)
        method, a = self.method, self.a
        if intervention == "reset_memory":
            w = torch.zeros_like(w)
        bs, length, d = q.shape
        if method in NO_FIELD:
            c = q.new_zeros(bs, length * d + 1)
        else:
            c = self.field(xh, q)
        if method not in NO_CONTROLLER and method != "fixed_learned":
            if intervention == "recompute_memory":
                w = torch.tanh(
                    self.memory(self.features(xh, q, torch.zeros_like(w), c))
                )
            h = self.features(xh, q, w, c)
            if method in HISTORY:
                w = torch.tanh(self.memory(h))
        if method in NO_FIELD:
            force = self.proposal(h)
        elif method == "direct":
            force = c[:, :-1].reshape_as(q)
        elif method == "gated":
            force = self.proposal(h) * c[:, -1, None, None]
        elif method in {"identity", "free_energy"} or intervention == "identity":
            force = vjp(c, q, c, train)
        elif method == "adam_energy":
            gradient = vjp(c, q, c, train)
            first, second, count = w.split((d, d, 1), dim=-1)
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient.square()
            count = count + 1
            w = torch.cat((first, second, count), dim=-1)
            # Smooth epsilon inside sqrt permits finite double backward at zero.
            force = (first / (1 - 0.9**count)) / (
                second / (1 - 0.999**count) + 1e-12
            ).sqrt()
        elif method == "snr":
            sketch = torch.randn(bs, c.shape[1], a.rank, device=q.device) / math.sqrt(
                c.shape[1]
            )
            rows = torch.stack(
                [
                    vjp(c, q, u, train, retain=True).flatten(1)
                    for u in sketch.unbind(-1)
                ],
                1,
            )
            gram = rows @ rows.transpose(1, 2)
            rhs = sketch.transpose(1, 2) @ c.unsqueeze(-1)
            coeff = torch.linalg.solve(
                gram + a.damping * torch.eye(a.rank, device=q.device), rhs
            )
            force = (rows.transpose(1, 2) @ coeff).reshape_as(q)
        else:
            if method in {"fixed", "fixed_learned"}:
                u = torch.cat((self.fixed[: length * d], self.fixed[-1:])).expand(
                    bs, -1, -1
                )
            elif method == "random" or intervention == "random":
                u = torch.randn(bs, c.shape[1], a.rank, device=q.device)
            else:
                u = self.factor(h).reshape(bs, length * d, a.rank)
                if method != "state_side":
                    u = torch.cat((u, self.score_factor(h.mean(1)).unsqueeze(1)), 1)
            if method == "residual_map":
                left = self.left(h).reshape(bs, length * d, a.rank)
                proj = u.transpose(1, 2) @ c.unsqueeze(-1)
                force = (left @ proj).reshape_as(q) / math.sqrt(length * d)
                force = force + self.direct_scale(h) * c[:, :-1].reshape_as(q)
            elif method == "diagonal":
                weights = a.m + a.b * u.square().sum(-1) / (
                    1 + u.square().sum((1, 2))[:, None]
                )
                force = vjp(c, q, weights * c, train)
            elif method == "state_side":
                plain = vjp(c, q, c, train).flatten(1)
                force = bounded_residual_mix(plain, u, m0=a.m, b=a.b).reshape_as(q)
            else:
                force = residual_mixing_force(
                    q, c, u, m0=a.m, b=a.b, create_graph=train
                )
        # The direct q-g comparator is parameter-free and takes a full correction.
        q = q - (1.0 if method == "direct" else a.alpha) * force
        return (q, w) if train else (q.detach(), w.detach())
