"""Supervision is confined to this module. Inner steps use only the observed input."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import NO_FIELD


@dataclass(frozen=True)
class Training:
    steps: int = 5000
    batch: int = 64  # global batch, sampled with replacement
    loops: int = 4
    lr: float = 0.0003
    weight_decay: float = 0.01
    clip: float = 1.0
    calibration: float = 1.0
    negative: float = 1.0
    margin: float = 1.0
    radius: float = 0.25  # fraction of half minimum codeword distance
    negative_loss: str = "signed"
    edits: str = "swap"
    save_every: int = 250
    log_every: int = 25
    max_seconds: float | None = None  # optional common elapsed-time training budget

    def __post_init__(self):
        if min(self.steps, self.batch, self.loops, self.log_every, self.save_every) < 1:
            raise ValueError(
                "Positive steps, batch, loops and logging intervals required"
            )
        if not (
            0 < self.radius < 1
            and self.lr > 0
            and self.clip > 0
            and self.margin > 0
            and min(self.weight_decay, self.calibration, self.negative) >= 0
        ):
            raise ValueError("Invalid optimizer or loss settings")
        if self.max_seconds is not None and not (
            math.isfinite(self.max_seconds) and self.max_seconds > 0
        ):
            raise ValueError("max_seconds must be finite and positive")
        if self.negative_loss not in {"signed", "energy"} or self.edits not in {
            "swap",
            "random",
        }:
            raise ValueError("Unknown negative objective or paired edit")


def calibration(model, xh, anchor, radius):
    # Bound each token's displacement strictly within its Voronoi cell.
    delta = F.normalize(torch.randn_like(anchor), dim=-1) * (
        radius / math.sqrt(2) * torch.rand_like(anchor[..., :1])
    )
    clean = model.field(xh, anchor)
    noisy = model.field(xh, anchor + delta)
    target = torch.cat((delta.flatten(1), torch.zeros_like(noisy[:, -1:])), 1)
    return (clean.square() + (noisy - target).square()).sum(-1).mean() / anchor.shape[
        1
    ], clean


def paired_edit(y, editable, vocab, mode):
    """Same two free sites for both corruption arms; distinct symbols for a real swap."""
    bs, length = y.shape
    row = torch.arange(bs, device=y.device)

    def pick(mask):
        return (
            torch.rand((bs, length), device=y.device).masked_fill(~mask, -1).argmax(1)
        )

    i = pick(editable)
    second = editable & (y != y[row, i, None])
    j = pick(second)
    eligible = second.any(1)
    out = y.clone()
    if mode == "swap":
        out[row, i], out[row, j] = y[row, j], y[row, i]
    else:
        for pos in (i, j):
            out[row, pos] = (
                y[row, pos] + torch.randint(1, vocab, (bs,), device=y.device)
            ) % vocab
    return out, eligible


class Objective(nn.Module):
    def __init__(self, model, training, frozen):
        super().__init__()
        self.model, self.cfg, self.frozen = model, training, frozen

    def forward(self, x, y, editable):
        m, cfg = self.model, self.cfg
        xh = m.encode(x)
        q, w = m.start(xh)
        initial_correct = m.decode(q.detach()).detach().argmax(-1).eq(y)
        anchor = m.codes[y]
        target = (
            (2 * anchor @ m.codes.T - m.codes.square().sum(-1)) / m.a.temperature
        ).softmax(-1)
        task = xh.new_zeros(())
        # Store one uniformly sampled reached state (including q0 and the endpoint).
        sample_at = int(torch.randint(cfg.loops + 1, ()).item())
        reached = q.detach()
        for step in range(1, cfg.loops + 1):
            q, w = m.step(xh, q, w, train=True)
            if step == sample_at:
                reached = q.detach()
            logits = m.decode(q)
            if m.method == "unrestricted_free":
                task = task + F.cross_entropy(logits.flatten(0, 1), y.flatten())
            else:
                task = (
                    task
                    + F.kl_div(logits.log_softmax(-1), target, reduction="none")
                    .sum(-1)
                    .mean()
                )
        task = task / cfg.loops
        correct = logits.detach().argmax(-1).eq(y)
        stats = {
            "task": task.detach(),
            "initial_exact": initial_correct.all(1).float().mean(),
            "final_token": correct.float().mean(),
            "final_exact": correct.all(1).float().mean(),
        }
        total = task
        if not self.frozen and m.method not in NO_FIELD | {"free_energy"}:
            if cfg.calibration:
                cal, clean = calibration(m, xh, anchor, cfg.radius)
                total = total + cfg.calibration * cal
                stats["calibration"] = cal.detach()
            elif cfg.negative and cfg.negative_loss == "energy":
                clean = m.field(xh, anchor)
        if (
            not self.frozen
            and m.method not in NO_FIELD | {"free_energy"}
            and cfg.negative
        ):
            wrong = m.decode(reached).detach().argmax(-1).ne(y).any(1)
            other = y.roll(1, dims=0)
            edited, eligible = paired_edit(y, editable, len(m.codes), cfg.edits)
            parts = (
                (reached, wrong, "reached"),
                (m.codes[other], other.ne(y).any(1), "other"),
                (m.codes[edited], eligible, "edited"),
            )
            neg = xh.new_zeros(())
            length = y.shape[1]
            for candidate, mask, name in parts:
                c = m.field(xh, candidate)
                if cfg.negative_loss == "signed":
                    violation = cfg.margin - c[:, -1] / math.sqrt(length)
                else:
                    violation = cfg.margin + (clean.square() - c.square()).sum(1) / (
                        2 * length
                    )
                count = mask.sum()
                world = 1
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(count)
                    world = torch.distributed.get_world_size()
                # DDP averages gradients; normalize by the global eligible sample count.
                neg = neg + world * (violation.relu() * mask).sum() / count.clamp_min(1)
                stats[f"negative_{name}_fraction"] = mask.float().mean()
            neg = neg / len(parts)
            total = total + cfg.negative * neg
            stats["negative"] = neg.detach()
        stats["total"] = total.detach()
        return total, stats
