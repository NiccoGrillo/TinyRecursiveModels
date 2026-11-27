"""Train digit-conditioned working-state edits and compare matched-work native paths."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from evaluate_matched_sudoku import model_from_checkpoint, read_boards, sha
from experiments.decision_stability.model import Adapter, State
from probe_native_trm_branches import audit
from run_trm_refinement_dev import native_training_bank

FORK, STEPS, BATCH, SEED = 12, 256, 8, 11
ARMS = ("serial", "clean_alternate", "two_hypotheses")


def hypothesis_choices(inputs, logits):
    values, digits = logits.float()[..., 2:].topk(2, dim=-1)
    margin = values[..., 0] - values[..., 1]
    cell = margin.masked_fill(inputs.ne(1), torch.inf).argmin(-1)
    row = torch.arange(len(inputs), device=inputs.device)
    return cell, digits[row, cell] + 2


class WorkspaceEdit(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * width + 22, 128), nn.GELU(), nn.Linear(128, width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state, logits, cell, digit):
        row = torch.arange(len(cell), device=cell.device)
        position = cell + state.z_L.shape[1] - logits.shape[1]
        high = state.z_H[row, position].float()
        low = state.z_L[row, position].float()
        scores = logits[row, cell].float()
        condition = F.one_hot(digit, 11).float()
        features = torch.cat((F.layer_norm(high, high.shape[-1:]), F.layer_norm(low, low.shape[-1:]),
                              F.layer_norm(scores, scores.shape[-1:]), condition), dim=-1)
        delta = self.net(features).tanh() * low.square().mean(-1, keepdim=True).sqrt()
        site = F.one_hot(position, state.z_L.shape[1]).unsqueeze(-1)
        updated = (state.z_L.float() + site * delta.unsqueeze(1)).to(state.z_L.dtype)
        return State(state.z_H, updated)


def batch_for(inputs):
    return {"inputs": inputs, "puzzle_identifiers": inputs.new_zeros(len(inputs))}


def rollout(adapter, state, batch, steps):
    outputs = []
    for _ in range(steps):
        state, score = adapter.step(state, batch)
        outputs.append(score)
    return state, outputs


def fit(adapter, edit, bank, device):
    optimizer = torch.optim.AdamW(edit.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(SEED)
    history = []
    for step in range(1, STEPS + 1):
        indices = torch.randint(len(bank[0]), (BATCH,), generator=generator)
        inputs, h, l, logits, labels = [t[indices].to(device) for t in bank]
        batch = batch_for(inputs)
        cell, digits = hypothesis_choices(inputs, logits)
        row = torch.arange(BATCH, device=device)
        true = labels[row, cell]
        alternate = torch.where(digits[:, 0] == true, digits[:, 1], digits[:, 0])
        # Correct hypotheses supervise the full board. Wrong hypotheses supervise only their own cell.
        condition_true = row < BATCH // 2
        proposed = torch.where(condition_true, true, alternate)
        start = edit(State(h, l), logits, cell, proposed)
        _, scores = adapter.step(start, batch, differentiable=True)
        selected_loss = F.cross_entropy(scores[row, cell].float(), proposed)
        board_loss = F.cross_entropy(scores[:BATCH // 2].float().flatten(0, 1), labels[:BATCH // 2].flatten())
        loss = board_loss + 0.25 * selected_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(edit.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 32 == 0:
            record = dict(step=step, board_loss=board_loss.item(), hypothesis_loss=selected_loss.item())
            history.append(record)
            print(record, flush=True)
    return history


@torch.inference_mode()
def evaluate(adapter, edit, saved, traces, device):
    x, y, h, l, logits = [saved[key].to(device) for key in ("inputs", "labels", "z_H", "z_L", "logits")]

    def board(i):
        inputs = x[i:i + 1]
        batch = batch_for(inputs)
        label = y[i]
        start, prefix = rollout(adapter, State(h[i:i + 1], l[i:i + 1]), batch, FORK - saved["fork"])
        expected = traces[i]["correct"]
        if (torch.cat(prefix).argmax(-1).eq(label).all(-1).tolist() != expected[saved["fork"]:FORK]
                or bool(logits[i].argmax(-1).eq(label).all()) != expected[saved["fork"] - 1]):
            raise ValueError(f"Common prefix disagrees with calibration on board {i}")
        base = prefix[-1]
        result = {"id": saved["ids"][i], "arms": {}}
        offset = i % len(ARMS)
        for arm in ARMS[offset:] + ARMS[:offset]:
            torch.cuda.synchronize()
            begin = perf_counter()
            if arm == "serial":
                _, outputs = rollout(adapter, start, batch, 4)
            else:
                cell, digits = hypothesis_choices(inputs, base)
                second = edit(start, base, cell, digits[:, 1])
                first = start if arm == "clean_alternate" else edit(start, base, cell, digits[:, 0])
                _, first_outputs = rollout(adapter, first, batch, 2)
                _, second_outputs = rollout(adapter, second, batch, 2)
                outputs = first_outputs + second_outputs
            scores = torch.cat([base, *outputs]).float()
            confidence, tokens = scores.log_softmax(-1).max(-1)
            choice = confidence[:, inputs[0].eq(1)].mean(-1).argmax()
            torch.cuda.synchronize()
            elapsed = perf_counter() - begin
            if not bool(scores.isfinite().all()):
                raise ValueError(f"Non-finite {arm} logits on board {i}")
            result["arms"][arm] = dict(exact=tokens.eq(label).all(-1).tolist(),
                                        selected=int(choice), seconds=elapsed)
            if arm == "two_hypotheses":
                if not torch.equal(adapter.readout(first), base) or not torch.equal(adapter.readout(second), base):
                    raise ValueError(f"An intervention changed the immediate readout on board {i}")
                index = int(cell.item())
                position = index + start.z_L.shape[1] - base.shape[1]
                original = start.z_L[0, position].float()
                rms = original.square().mean().sqrt().clamp_min(1e-30)
                realized = [((state.z_L[0, position].float() - original).square().mean().sqrt() / rms).item()
                            for state in (first, second)]
                result.update(cell=index, proposed=digits[0].tolist(), target=int(label[index]),
                              hypothesis_readouts=tokens[[1, 2, 3, 4], index].tolist(),
                              branches_disagree_at_end=not torch.equal(tokens[2], tokens[4]),
                              different_post_cast_states=not torch.equal(first.z_L, second.z_L),
                              realized_relative_cell_rms=realized)
        if result["arms"]["serial"]["exact"] != expected[FORK - 1:FORK + 4]:
            raise ValueError(f"Serial continuation disagrees with calibration on board {i}")
        if result["arms"]["clean_alternate"]["exact"][:3] != result["arms"]["serial"]["exact"][:3]:
            raise ValueError(f"Clean branch disagrees with serial on board {i}")
        return result

    for i in range(4):
        board(i)
    return [board(i) for i in range(len(traces))]


def run(root, output):
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    calibration, trace_audit = audit(root / "dev-calibration/calibration.json", root / "dev-refiners/dev-results.json")
    prior = json.loads((root / "dev-refiners/dev-results.json").read_text())
    config, weights = (root / "checkpoint" / name for name in ("all_config.yaml", "checkpoint.pt"))
    bank_path = root / "dev-calibration/fork_states.pt"
    if (sha(config) != calibration["config_sha256"] or sha(weights) != calibration["checkpoint_sha256"]
            or sha(bank_path) != calibration["bank_sha256"] or torch.cuda.get_device_name() != calibration["device"]
            or str(torch.__version__) != calibration["torch"]):
        raise ValueError("Expected the archived checkpoint, development bank and GH200 runtime")
    saved = torch.load(bank_path, map_location="cpu", weights_only=True)
    if saved["ids"] != prior["ids"] or saved["fork"] != prior["fork"]:
        raise ValueError("Saved fork cohort mismatch")
    data = root / "train-only"
    cohort = json.loads((data / "cohort.json").read_text())
    inputs, labels, ids, sha_train = read_boards(data, "train", "trm",
                                                 slice(0, None, cohort["augments"] + 1), device)
    if ids != prior["train_ids"] or sha_train != prior["train_sha256"] or set(ids) & set(saved["ids"]):
        raise ValueError("Training cohort mismatch")
    native, _ = model_from_checkpoint(SimpleNamespace(family="trm", weights=weights, config=config), device)
    native.requires_grad_(False)
    adapter = Adapter(native)
    output.mkdir()
    bank = native_training_bank(native, inputs, labels, FORK)
    test_x, test_h, test_l = [t[:BATCH].to(device) for t in bank[:3]]
    state, batch = State(test_h, test_l), batch_for(test_x)
    _, full = adapter.step(state, batch, differentiable=True)
    _, native_scores = adapter.step(state, batch)
    if not torch.equal(full, native_scores):
        raise ValueError("Full-gradient forward disagrees with native forward")
    edit = WorkspaceEdit(native.config.hidden_size).to(device)
    print("Training digit-conditioned edit from original training board states", flush=True)
    history = fit(adapter, edit, bank, device)
    checkpoint = output / "workspace-edit.pt"
    torch.save(edit.state_dict(), checkpoint)
    print("Evaluating all development trajectories", flush=True)
    rows = evaluate(adapter, edit.eval(), saved, calibration["traces"], device)
    report = dict(fork=FORK, steps=STEPS, training_batch=BATCH, seed=SEED, train_sha256=sha_train,
                  checkpoint_sha256=sha(checkpoint), native_sha256=trace_audit["checkpoint_sha256"],
                  calibration_sha256=trace_audit["calibration_sha256"], script_sha256=sha(Path(__file__)),
                  native_blocks_per_arm=adapter.block_calls_per_macro * 4, history=history, rows=rows)
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print({arm: sum(any(row["arms"][arm]["exact"]) for row in rows) for arm in ARMS}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
