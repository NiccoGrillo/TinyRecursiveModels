"""Bounded workspace-dependent corrections to the native answer proposal."""

from torch import nn
import torch
import torch.nn.functional as F


def mix_proposal(answer, proposal, gate, directions=None):
    delta = proposal.float() - answer.float()
    if directions is not None:
        u, v = directions
        delta = u * (v * delta).sum(dim=-1, keepdim=True)
    # Keep the original proposal as the base to preserve zero-gate parity in BF16.
    return proposal + (gate * delta).to(proposal.dtype)


class AnswerMixer(nn.Module):
    def __init__(self, width, method, hidden=64):
        super().__init__()
        if method not in ("scalar", "directional"):
            raise ValueError("Expected scalar or directional answer mixing")
        self.features = nn.Linear(2 * width, hidden)
        self.gate = nn.Linear(hidden, 1)
        self.directions = nn.Linear(hidden, 2 * width) if method == "directional" else None
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, answer, workspace, proposal):
        h, l = answer.float(), workspace.float()
        features = F.gelu(self.features(torch.cat((h, l), dim=-1)))
        gate = 0.5 * self.gate(features).tanh()
        directions = None
        if self.directions is not None:
            directions = tuple(vector / (vector.norm(dim=-1, keepdim=True) + 1e-6)
                               for vector in self.directions(features).chunk(2, dim=-1))
        return mix_proposal(answer, proposal, gate, directions)
