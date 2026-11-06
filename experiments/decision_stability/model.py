"""An explicit adapter to the pinned upstream TRM, including exact full gradients."""
from pathlib import Path
import hashlib

import torch
import yaml

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1 as TRM,
    TinyRecursiveReasoningModel_ACTV1Config as Config,
    TinyRecursiveReasoningModel_ACTV1InnerCarry as State,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smoke_config(mlp_t=True):
    return dict(batch_size=2, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
                puzzle_emb_ndim=32, puzzle_emb_len=1, H_cycles=2, L_cycles=2,
                H_layers=0, L_layers=1, hidden_size=32, expansion=1, num_heads=2,
                pos_encodings="none" if mlp_t else "rope", mlp_t=mlp_t,
                halt_max_steps=4, halt_exploration_prob=0, forward_dtype="float32")


def read_config(path, metadata, batch_size, dtype=None):
    """Read upstream resolved all_config.yaml or a flat, complete model JSON/YAML."""
    raw = Path(path).read_text()
    if "${" in raw:
        raise ValueError("Use a resolved all_config.yaml, not a Hydra interpolation template")
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("Model configuration must be a mapping")
    if "arch" in config:
        config = dict(config["arch"])
        name = config.pop("name", None)
        if name != "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1":
            raise ValueError(f"Only the original TRM architecture is supported: {name}")
        config.pop("loss", None)
    else:
        config = dict(config)
    config.pop("causal", None)  # passed but unused by upstream pretrain.py
    unknown = set(config) - set(Config.model_fields)
    if unknown:
        raise ValueError(f"Unknown model options: {sorted(unknown)}")
    for key in ("seq_len", "vocab_size", "num_puzzle_identifiers"):
        if key in config and config[key] != metadata[key]:
            raise ValueError(f"Model/data mismatch for {key}")
        config[key] = metadata[key]
    config["batch_size"] = batch_size  # eval sparse embedding buffers are not persistent
    if dtype is not None:
        config["forward_dtype"] = dtype
    config = Config(**config).model_dump()
    if config["forward_dtype"] not in ("float32", "bfloat16"):
        raise ValueError("Diagnostics currently support float32 and bfloat16 forward precision")
    if config["H_cycles"] < 1 or config["L_cycles"] < 1 or config["L_layers"] < 1:
        raise ValueError("Recursion and layer counts must be positive")
    if config["puzzle_emb_ndim"] == 0 and config["puzzle_emb_len"] != 0:
        raise ValueError("Upstream requires puzzle_emb_len=0 when puzzle embeddings are disabled")
    return config


def load_model(config, checkpoint=None, device="cpu"):
    model = TRM(config).to(device).eval()
    if checkpoint is not None:
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Expected an upstream state_dict, not a training bundle")
        # These are the two actual wrappers in upstream pretrain.py, in this order.
        for prefix in ("_orig_mod.", "model."):
            if all(isinstance(k, str) and k.startswith(prefix) for k in weights):
                weights = {k[len(prefix):]: v for k, v in weights.items()}
        model.load_state_dict(weights, strict=True)
    model.requires_grad_(False)
    return Adapter(model)


class Adapter:
    def __init__(self, model):
        self.model = model
        self.inner = model.inner
        self.config = model.config

    @property
    def module_calls_per_macro(self):
        return self.config.H_cycles * (self.config.L_cycles + 1)

    @property
    def block_calls_per_macro(self):
        return self.module_calls_per_macro * self.config.L_layers

    def initial(self, batch):
        inner = self.inner
        shape = (len(batch["inputs"]), self.config.seq_len + inner.puzzle_emb_len,
                 self.config.hidden_size)
        return State(inner.H_init.expand(shape).clone(), inner.L_init.expand(shape).clone())

    def readout(self, state):
        return self.inner.lm_head(state.z_H)[:, self.inner.puzzle_emb_len:]

    def q_logit(self, state):
        return self.inner.q_head(state.z_H[:, 0]).float()[:, 0]

    @torch.no_grad()
    def outer_context(self, batch):
        """Input/position tensors are constant during this frozen-model experiment."""
        return (self.inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"]),
                self.inner.rotary_emb() if hasattr(self.inner, "rotary_emb") else None)

    @torch.no_grad()
    def outer_step(self, state, batch, low_cycles=None, context=None):
        """Exactly one z_H update. Native L repeated H times equals one native macro.

        Overriding L changes the recurrence; observing native L at this finer phase
        does not. This inference-only path intentionally never invokes ACT resets.
        """
        cycles = self.config.L_cycles if low_cycles is None else low_cycles
        if not isinstance(cycles, int) or cycles < 1:
            raise ValueError("low_cycles must be a positive integer")
        injection, cos_sin = self.outer_context(batch) if context is None else context
        high, low = state.z_H, state.z_L
        for _ in range(cycles):
            low = self.inner.L_level(low, high + injection, cos_sin=cos_sin)
        high = self.inner.L_level(high, low, cos_sin=cos_sin)
        result = State(high, low)
        return result, self.readout(result)

    def step(self, state, batch, differentiable=False):
        if not differentiable:
            with torch.no_grad():
                new_state, logits, _ = self.inner(state, batch)
            return new_state, logits
        # Upstream's training truncation is intentionally absent here. This is the
        # SAME forward recurrence, differentiated through every cycle and macro-step.
        inner = self.inner
        seq = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
        injection = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        high, low = state.z_H, state.z_L
        for _ in range(self.config.H_cycles):
            for _ in range(self.config.L_cycles):
                low = inner.L_level(low, high + injection, **seq)
            high = inner.L_level(high, low, **seq)
        result = State(high, low)
        return result, self.readout(result)


def clone_state(state):
    return State(state.z_H.detach().clone(), state.z_L.detach().clone())


def reference_margins(logits, reference):
    """Logit of the chosen token minus the largest competing logit at each cell."""
    chosen = logits.float().gather(-1, reference[..., None]).squeeze(-1)
    competitors = logits.float().scatter(-1, reference[..., None], -torch.inf)
    return chosen - competitors.max(dim=-1).values


def rms(tensor):
    return tensor.float().square().flatten(1).mean(1).sqrt()
