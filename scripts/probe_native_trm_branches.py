"""Audit fixed native Sudoku trajectories and probe two branches at equal native work."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

DEPTH = 16
WORK = 4
NOISE_RMS = 0.10
SEED = 11


def audit(calibration_path, prior_path):
    digest = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    calibration = json.loads(calibration_path.read_text())
    prior = json.loads(prior_path.read_text())
    traces = calibration["traces"]
    ids = [row["id"] for row in traces]
    correct = [row["correct"] for row in traces]
    fork = prior["fork"]
    budget = prior["budget_seconds"]
    if (digest != prior["calibration_sha256"] or calibration["cohort"] != "dev"
            or len(ids) != len(set(ids)) or len(ids) != calibration["n"] or ids != prior["ids"]
            or calibration["choice"]["fork"] != fork or calibration["choice"]["seconds"] != budget
            or calibration["checkpoint_sha256"] != prior["checkpoint_sha256"]
            or any(len(row) != DEPTH or any(type(ok) is not bool for ok in row) for row in correct)
            or [row[fork - 1] for row in correct] != prior["outcomes"]["stop"]):
        raise ValueError("Expected the sealed native development cohort and matching comparison")
    per_depth = []
    for depth in range(1, DEPTH + 1):
        ends = [row[depth - 1] for row in correct]
        visited = [any(row[:depth]) for row in correct]
        per_depth.append(dict(depth=depth, exact=sum(ends), ever_exact=sum(visited),
                              wrong_after_solve=sum(visited) - sum(ends),
                              adjacent_breaks=sum(row[depth - 2] and not row[depth - 1]
                                                  for row in correct) if depth > 1 else 0))
    timing = prior["timing_rows"]["native"]
    timed_lost, timed_wrong = [], []
    for board_id, row, timed, expected in zip(ids, correct, timing, prior["outcomes"]["native"], strict=True):
        depth = fork + timed["steps"]
        if not fork <= depth <= DEPTH or timed["nonfinite"]:
            raise ValueError("Timed endpoint differs from the sealed trajectory")
        if row[depth - 1] != timed["correct"] or timed["correct"] != expected:
            raise ValueError("Timed endpoint differs from the sealed trajectory")
        if not expected:
            timed_wrong.append(board_id)
            if any(row[:depth]):
                timed_lost.append(board_id)
    return calibration, dict(
        calibration_sha256=digest, checkpoint_sha256=calibration["checkpoint_sha256"],
        data_sha256=calibration["data_sha256"], count=len(ids), fork=fork, per_depth=per_depth,
        final_wrong_after_solve_ids=[key for key, row in zip(ids, correct) if any(row) and not row[-1]],
        never_solved_ids=[key for key, row in zip(ids, correct) if not any(row)],
        timed=dict(budget_seconds=budget, exact=len(ids) - len(timed_wrong),
                   wrong_ids=timed_wrong, wrong_after_solve_ids=timed_lost),
    )


def probe(root, calibration, requested_fork=None):
    from time import perf_counter
    from types import SimpleNamespace

    import torch

    from evaluate_matched_sudoku import model_from_checkpoint, sha
    from experiments.decision_stability.interventions import perturb, random_direction
    from experiments.decision_stability.model import State
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1InnerCarry as InnerCarry

    bank_path = root / "dev-calibration/fork_states.pt"
    config = root / "checkpoint/all_config.yaml"
    weights = root / "checkpoint/checkpoint.pt"
    device = torch.device("cuda")
    if (sha(bank_path) != calibration["bank_sha256"] or sha(weights) != calibration["checkpoint_sha256"]
            or sha(config) != calibration["config_sha256"] or str(torch.__version__) != calibration["torch"]
            or torch.cuda.get_device_name() != calibration["device"]):
        raise ValueError("Expected the original native model, fork bank and GH200 environment")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _ = model_from_checkpoint(SimpleNamespace(family="trm", weights=weights, config=config), device)
    saved = torch.load(bank_path, map_location="cpu", weights_only=True)
    saved_fork = calibration["choice"]["fork"]
    fork = saved_fork if requested_fork is None else requested_fork
    traces = calibration["traces"]
    if saved["ids"] != [row["id"] for row in traces] or saved["fork"] != saved_fork:
        raise ValueError("Saved fork IDs or depth differ from the calibration")
    if not saved_fork <= fork <= DEPTH - WORK:
        raise ValueError("The common prefix and serial continuation must fit in the calibrated trajectory")
    x, y, h, l, logits = [saved[k].to(device) for k in ("inputs", "labels", "z_H", "z_L", "logits")]

    def roll(state, batch, steps):
        scores = []
        for _ in range(steps):
            state, score, _ = model.inner(state, batch)
            scores.append(score)
        return state, scores

    def readout(state):
        return model.inner.lm_head(state.z_H)[:, model.inner.puzzle_emb_len:]

    def select(scores, blanks, started):
        stack = torch.cat(scores).float()
        confidence, tokens = stack.log_softmax(-1).max(-1)
        chosen = confidence[:, blanks].mean(-1).argmax()
        torch.cuda.synchronize()
        elapsed = perf_counter() - started
        if not bool(stack.isfinite().all()):
            raise FloatingPointError("Non-finite native logits")
        return tokens, int(chosen), elapsed

    def run_board(i):
        board = x[i:i + 1]
        batch = {"inputs": board, "puzzle_identifiers": board.new_zeros(1)}
        state = InnerCarry(h[i:i + 1], l[i:i + 1])
        base = logits[i:i + 1]
        native = traces[i]["correct"]
        if fork > saved_fork:
            state, prefix = roll(state, batch, fork - saved_fork)
            if torch.cat(prefix).argmax(-1).eq(y[i]).all(-1).tolist() != native[saved_fork:fork]:
                raise ValueError(f"Reconstructed prefix disagrees with the archived trace on board {i}")
            base = prefix[-1]
        blanks = board[0].eq(1)
        seed = int(saved["ids"][i][:16], 16) ^ SEED
        result = {"id": saved["ids"][i]}
        for arm in (("serial", "split") if i % 2 == 0 else ("split", "serial")):
            torch.cuda.synchronize()
            started = perf_counter()
            if arm == "serial":
                _, serial = roll(state, batch, WORK)
                scores = [base, *serial]
            else:
                original = State(state.z_H, state.z_L)
                perturbed, norm = perturb(original, random_direction(original, seed), NOISE_RMS)
                _, clean = roll(state, batch, WORK // 2)
                _, alternate = roll(InnerCarry(perturbed.z_H, perturbed.z_L), batch, WORK // 2)
                scores = [base, *clean, *alternate]
            tokens, chosen, elapsed = select(scores, blanks, started)
            exact = tokens.eq(y[i]).all(-1).tolist()
            result[arm] = dict(exact=exact, selected=chosen, seconds=elapsed)
        if (not torch.equal(readout(perturbed), base) or not norm[0]["budget_ok"] or norm[0]["zero_delta"]):
            raise ValueError("Fork readout changed or the perturbation missed its RMS budget")
        if result["serial"]["exact"] != native[fork - 1:fork + WORK]:
            raise ValueError(f"Serial continuation disagrees with the archived trace on board {i}")
        if result["split"]["exact"][:1 + WORK // 2] != result["serial"]["exact"][:1 + WORK // 2]:
            raise ValueError(f"Clean branch disagrees with serial continuation on board {i}")
        return dict(**result, native_ever_exact=any(native), perturbation=norm[0])

    with torch.inference_mode():
        for i in range(4):
            run_board(i)
        rows = [run_board(i) for i in range(len(traces))]
    serial = [any(r["serial"]["exact"]) for r in rows]
    split = [any(r["split"]["exact"]) for r in rows]
    return dict(fork=fork, macro_updates=WORK, blocks_per_macro=model.config.H_cycles * (model.config.L_cycles + 1)
                * model.config.L_layers, seed=SEED, relative_rms=NOISE_RMS,
                serial_oracle=sum(serial), split_oracle=sum(split),
                split_only_ids=[r["id"] for r, s, b in zip(rows, serial, split) if b and not s],
                serial_only_ids=[r["id"] for r, s, b in zip(rows, serial, split) if s and not b],
                split_solves_never_native_ids=[r["id"] for r, b in zip(rows, split)
                                              if b and not r["native_ever_exact"]],
                serial_last=sum(r["serial"]["exact"][-1] for r in rows),
                selected_exact={arm: sum(r[arm]["exact"][r[arm]["selected"]] for r in rows)
                                for arm in ("serial", "split")},
                median_seconds={arm: median(r[arm]["seconds"] for r in rows) for arm in ("serial", "split")},
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true", help="Run the GPU branch comparison after the CPU audit.")
    parser.add_argument("--fork", type=int, help="GPU fork depth. Defaults to the archived fork.")
    args = parser.parse_args()
    if args.fork is not None and not args.probe:
        parser.error("--fork requires --probe")
    calibration, report = audit(args.root / "dev-calibration/calibration.json",
                                args.root / "dev-refiners/dev-results.json")
    if args.probe:
        report["branch_probe"] = probe(args.root, calibration, args.fork)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Depth {DEPTH}: {report['per_depth'][-1]}", flush=True)
    if args.probe:
        result = report["branch_probe"]
        print(f"Oracle candidates: serial {result['serial_oracle']}, split {result['split_oracle']}", flush=True)


if __name__ == "__main__":
    main()
