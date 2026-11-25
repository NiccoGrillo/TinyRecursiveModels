"""Fixed-budget solving, paired natural-error repair, and offline mechanism probes."""

import json
import math
import time

import torch

from .common import atomic, digest, exclusive, fingerprint, read, source, write
from .model import NO_FIELD, vjp
from .train import checkpoint, from_checkpoint, hardware, sync


def checked_output(out, spec):
    done = out / "complete.json"
    if done.exists():
        prior = read(done)
        if prior["spec"] != spec:
            raise ValueError(f"Output belongs to a different evaluation: {out}")
        for name, checksum in prior["files"].items():
            if digest(out / name) != checksum:
                raise ValueError(f"Corrupt result: {out / name}")
        return True
    return False


def elapsed_call(device, fn):
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return result, time.perf_counter() - start


def make_bank(path, model_path, ds, forks, batch, device):
    """Bank keys are dataset row IDs, never correctness. All examples are kept."""
    spec = dict(
        data=ds.fingerprint,
        ids=fingerprint(ds.ids.tolist()),
        ref=digest(model_path),
        forks=forks,
        batch=batch,
        hardware=hardware(device),
        source=source(),
    )
    with exclusive(path):
        if checked_output(path, spec):
            return
        model, ck = from_checkpoint(model_path, device)
        if ck["dataset"] != ds.fingerprint:
            raise ValueError("Reference was trained on a different dataset")
        buffers = {f: [] for f in forks}
        for start in range(0, len(ds.ids), batch):
            x, _, _ = ds.batch(slice(start, start + batch), device)
            with torch.no_grad():
                xh = model.encode(x)
                q, w = model.start(xh)
            for t in range(max(forks) + 1):
                if t in buffers:
                    buffers[t].append(q.detach().cpu())
                if t < max(forks):
                    q, w = model.step(xh, q, w, train=False)
        with atomic(path / "bank.pt") as tmp:
            torch.save(
                dict(
                    ids=ds.ids.tolist(),
                    q={t: torch.cat(v) for t, v in buffers.items()},
                    reference=spec["ref"],
                ),
                tmp,
            )
        write(
            path / "complete.json",
            dict(spec=spec, files={"bank.pt": digest(path / "bank.pt")}),
        )


class Search:
    """One generic proposal per step; score all restarts and preserve the incumbent."""

    def __init__(self, model, xh, q, edits, uphill, restart):
        self.m, self.xh, self.edits = model, xh, edits
        self.uphill, self.restart = uphill, restart
        self.best = self.current = model.decode(q).argmax(-1)
        self.best_score = self.score = self.value(self.current)
        self.finite = self.score.isfinite()

    def value(self, tokens):
        return self.m.field(self.xh, self.m.codes[tokens])[:, -1] / math.sqrt(
            tokens.shape[1]
        )

    @torch.no_grad()
    def step(self, t):
        bs, length = self.current.shape
        vocab = len(self.m.codes)
        if self.restart and t % self.restart == 0:
            prop = torch.randint_like(self.current, vocab)
            restart = True
        else:
            positions = torch.rand(bs, length, device=self.current.device).argsort(1)[
                :, : self.edits
            ]
            prop = self.current.clone()
            old = prop.gather(1, positions)
            prop.scatter_(
                1, positions, (old + torch.randint_like(old, 1, vocab)) % vocab
            )
            restart = False
        score = self.value(prop)
        self.finite = self.finite & score.isfinite()
        improve = score < self.best_score
        self.best = torch.where(improve[:, None], prop, self.best)
        self.best_score = torch.minimum(score, self.best_score)
        take = (score <= self.score) | restart
        if self.uphill > 0:
            take = take | (torch.rand_like(score) < self.uphill)
        self.current = torch.where(take[:, None], prop, self.current)
        self.score = torch.where(take, score, self.score)
        return self.best


def price_reference(reference, devset, cfg, length, device):
    """Dev timing only; same budget for all arms of one reference/seed/length group."""
    m, _ = from_checkpoint(reference, device)
    values = []
    b = min(cfg["batch"], len(devset.ids))
    for i in range(cfg["timing_batches"] + 1):
        idx = (torch.arange(b) + i * b) % len(devset.ids)
        x, _, _ = devset.batch(idx, device)
        x = x.repeat(1, math.ceil(length / x.shape[1]))[:, :length]
        with torch.no_grad():
            xh = m.encode(x)
            q, w = m.start(xh)

        def run():
            state, memory = q, w
            for _ in range(cfg["time_reference_steps"]):
                state, memory = m.step(xh, state, memory, train=False)
                m.decode(state).argmax(-1)

        _, seconds = elapsed_call(device, run)
        if i:
            values.append(seconds)
    return dict(
        seconds=float(torch.tensor(values).median()),
        samples_s=values,
        batch=b,
        length=length,
        device=hardware(device),
    )


def ops(method, rank, t, cfg):
    if method.startswith("search"):
        restarts = t // cfg["restart_every"] if cfg["restart_every"] else 0
        return dict(
            field=t + 1,
            vjp=0,
            solves=0,
            proposals=t - restarts,
            restarts=restarts,
            controller=0,
        )
    field = 0 if method in NO_FIELD else t
    grads = 0 if method in NO_FIELD | {"direct", "gated", "residual_map"} else t
    controller = (
        t
        if method
        in {
            "recurrent",
            "stateless",
            "diagonal",
            "state_side",
            "gated",
            "residual_map",
            "unrestricted",
            "unrestricted_free",
        }
        else 0
    )
    return dict(
        field=field,
        vjp=rank * grads if method == "snr" else grads,
        solves=t if method == "snr" else 0,
        proposals=0,
        restarts=0,
        controller=controller,
    )


def diagnostics(m, xh, q, y):
    q = q.detach().requires_grad_()
    c = m.field(xh, q)
    g, s = c[:, :-1], c[:, -1]
    ordinary = vjp(c, q, c, False, retain=True)
    ds = torch.autograd.grad(s.sum(), q, retain_graph=True)[0]
    probe = torch.empty_like(g).bernoulli_(0.5).mul_(2).sub_(1)
    dg = torch.autograd.grad(g, q, probe)[0]

    def norm(t):
        return t.flatten(1).norm(dim=1)

    with torch.no_grad():
        a = m.codes[y]
        pos = m.field(xh, a)
        delta = torch.nn.functional.normalize(torch.randn_like(a), dim=-1) * 0.1
        near = m.field(xh, a + delta)
        other = a.roll(1, 0)
        different = y.ne(y.roll(1, 0)).any(1)
        swapped_input = m.field(xh.roll(1, 0), a)[:, -1]
        swapped_answer = m.field(xh, other)[:, -1]
    return dict(
        current_correct=m.decode(q.detach()).argmax(-1).eq(y).all(1),
        residual=norm(c),
        geometry=norm(g),
        invalidity=s.detach() / math.sqrt(q.shape[1]),
        ordinary_force=norm(ordinary),
        score_derivative=norm(ds),
        geometry_vjp_probe=norm(dg),
        anchor_residual=norm(pos),
        local_geometry_error=norm(near[:, :-1] - delta.flatten(1)),
        anchor_invalidity=pos[:, -1] / math.sqrt(q.shape[1]),
        swapped_input_invalidity=swapped_input / math.sqrt(q.shape[1]),
        swapped_answer_invalidity=swapped_answer / math.sqrt(q.shape[1]),
        distinct_answer=different,
    )


def evaluate_arm(
    out, model_path, ds, cfg, device, price, *, bank_path=None, search_edits=0, seed=0
):
    bank = checkpoint(bank_path / "bank.pt") if bank_path else None
    if bank and bank["ids"] != ds.ids.tolist():
        raise ValueError("Reference bank and evaluation IDs differ")
    m, ck = from_checkpoint(model_path, device)
    if ck["dataset"] != ds.fingerprint:
        raise ValueError("Model and evaluation datasets differ")
    if bank and bank["reference"] != (ck["reference"] or digest(model_path)):
        raise ValueError("Controller and state bank do not share a reference")
    method = f"search{search_edits}" if search_edits else m.method
    spec = dict(
        dataset=ds.fingerprint,
        ids=fingerprint(ds.ids.tolist()),
        method=method,
        model=digest(model_path),
        bank=digest(bank_path / "bank.pt") if bank else None,
        config=cfg,
        budget=price,
        seed=seed,
        source=source(),
        hardware=hardware(device),
    )
    with exclusive(out):
        if checked_output(out, spec):
            return
        with (
            atomic(out / "examples.jsonl") as output,
            atomic(out / "diagnostics.jsonl") as probes,
        ):
            with output.open("w") as fp, probes.open("w") as dp:
                for f in cfg["forks"] if bank else [0]:
                    for start in range(0, len(ds.ids), cfg["batch"]):
                        stop = min(start + cfg["batch"], len(ds.ids))
                        idx = slice(start, stop)
                        x, y, editable = ds.batch(idx, device)

                        @torch.no_grad()
                        def initialize():
                            xh = m.encode(x)
                            if bank:
                                q = bank["q"][f][idx].to(device)
                                w = m.zero_memory(xh)
                            else:
                                q, w = m.start(xh)
                            return xh, q, w

                        (xh, q, w), setup = elapsed_call(device, initialize)
                        torch.manual_seed(seed + start + f * 7907)
                        _batch(
                            fp,
                            dp,
                            ds.ids[idx],
                            m,
                            xh,
                            q,
                            w,
                            y,
                            editable,
                            method,
                            f,
                            start,
                            cfg,
                            price["seconds"],
                            device,
                            search_edits,
                            setup,
                        )
        files = {p: digest(out / p) for p in ("examples.jsonl", "diagnostics.jsonl")}
        write(out / "complete.json", dict(spec=spec, files=files))


def _batch(
    fp,
    dp,
    ids,
    m,
    xh,
    q,
    w,
    y,
    editable,
    method,
    fork,
    start,
    cfg,
    budget,
    device,
    edits,
    setup,
):
    hset = sorted(set(cfg["horizons"]))
    if min(hset) != 0 or cfg["max_cost_steps"] < max(hset):
        raise ValueError("Horizons must contain 0 and fit the cost cap")
    with torch.no_grad():
        initial, readout_s = elapsed_call(device, lambda: m.decode(q).argmax(-1))
    setup += readout_s
    search = None
    seconds = 0.0

    def advance(t):
        nonlocal q, w
        if search is not None:
            return search.step(t)
        q, w = m.step(xh, q, w, train=False)
        return m.decode(q).argmax(-1)

    # Warm the exact arm without consuming its stochastic stream or altering its state.
    if start == 0:
        keep_q, keep_w = q, w
        with torch.random.fork_rng(
            devices=[torch.cuda.current_device()] if device.type == "cuda" else []
        ):
            if edits:
                with torch.no_grad():
                    scratch = Search(
                        m, xh, q, edits, cfg["uphill"], cfg["restart_every"]
                    )
                    scratch.step(1)
            else:
                advance(1)
        q, w = keep_q, keep_w
        sync(device)
    if edits:
        with torch.no_grad():
            search, seconds = elapsed_call(
                device,
                lambda: Search(m, xh, q, edits, cfg["uphill"], cfg["restart_every"]),
            )
    outcomes, token_counts = [], []

    def score(prediction, valid):
        correct = prediction.eq(y) & valid[:, None]
        outcomes.append(correct.all(1))
        token_counts.append(
            torch.stack((correct.sum(1), (correct & editable).sum(1)), dim=1)
        )

    finite = q.isfinite().flatten(1).all(1)
    score(initial, finite)
    failure = torch.where(finite, -1, 0)
    times = [seconds]
    prev = initial
    streak = torch.zeros(len(ids), dtype=torch.long, device=device)
    patient = torch.full_like(streak, -1)
    mechanism_at = min(2, max(hset))
    for t in range(1, cfg["max_cost_steps"] + 1):
        pred, dt = elapsed_call(device, lambda: advance(t))
        seconds += dt
        finite = finite & (
            search.finite
            if search is not None
            else q.isfinite().flatten(1).all(1) & w.isfinite().flatten(1).all(1)
        )
        failure = torch.where((failure < 0) & ~finite, t, failure)
        score(pred, finite)
        same = pred.eq(prev).all(1)
        streak = torch.where(same, streak + 1, 0)
        patient = torch.where((patient < 0) & (streak >= cfg["patience"]), t, patient)
        prev = pred
        times.append(seconds)
        if (
            t == mechanism_at
            and start < cfg["diagnostic_examples"]
            and m.method not in NO_FIELD
            and not edits
        ):
            n = min(len(ids), cfg["diagnostic_examples"] - start)
            with torch.random.fork_rng(
                devices=[torch.cuda.current_device()] if device.type == "cuda" else []
            ):
                _probe(
                    dp,
                    ids[:n],
                    m,
                    xh[:n],
                    q[:n],
                    w[:n],
                    y[:n],
                    fork,
                    t,
                    cfg.get("recompute_memory", False),
                )
        if t >= max(hset) and seconds > budget:
            break
    ok = torch.stack(outcomes, dim=1).cpu().tolist()
    counts = torch.stack(token_counts, dim=1).cpu().tolist()
    editable_totals = editable.sum(1).cpu().tolist()
    patient = patient.cpu().tolist()
    failure = failure.cpu().tolist()
    affordable = [i for i, v in enumerate(times) if v <= budget]
    # At zero affordable score calls, return the decoded starting candidate untouched.
    wall = affordable[-1] if affordable else 0
    for i, key in enumerate(ids):
        hit = patient[i] if 0 <= patient[i] <= max(hset) else max(hset)
        steps = {f"step_{t}": t for t in hset} | {"wall": wall, "patience": hit}
        rows = {}
        for name, t in steps.items():
            cost = ops(method, m.a.rank, t, cfg)
            observed = times[t]
            if name == "wall" and not affordable:
                cost["field"], observed = 0, 0.0
            tokens_correct, editable_tokens_correct = counts[i][t]
            rows[name] = dict(
                t=t,
                ok=ok[i][t],
                seconds=observed,
                tokens_correct=tokens_correct,
                tokens_total=y.shape[1],
                editable_tokens_correct=editable_tokens_correct,
                editable_tokens_total=editable_totals[i],
                **cost,
            )
        obj = dict(
            id=str(key),
            fork=fork,
            initial=ok[i][0],
            policy=rows,
            setup_seconds=setup,
            first_nonfinite_step=failure[i] if failure[i] >= 0 else None,
            correct_by_step=ok[i][: max(hset) + 1],
            patience_hit=0 <= patient[i] <= max(hset),
            wall_cap_hit=times[-1] <= budget,
            measured_steps=len(times) - 1,
            first_over_budget_s=times[wall + 1]
            if affordable and wall + 1 < len(times)
            else (times[0] if not affordable else None),
        )
        fp.write(json.dumps(obj, allow_nan=False) + "\n")


def _probe(dp, ids, m, xh, q, w, y, fork, t, recompute):
    values = {k: v.detach().cpu().tolist() for k, v in diagnostics(m, xh, q, y).items()}
    if m.method == "recurrent":
        # Fork identical continuous state and memory; run two charged steps per intervention.
        directions = {}
        modes = [None, "reset_memory", "random", "identity"]
        if recompute:
            modes.append("recompute_memory")
        for mode in modes:
            state, memory = q, w
            for j in range(2):
                state, memory = m.step(
                    xh, state, memory, train=False, intervention=mode
                )
                if j == 0:
                    directions[mode] = state
            name = mode or "native"
            values[name + "_endpoint"] = (
                m.decode(state).argmax(-1).eq(y).all(1).tolist()
            )
            values[name + "_delta"] = (
                (directions[mode] - directions[None]).flatten(1).norm(dim=1).tolist()
            )
    for i, key in enumerate(ids):
        dp.write(
            json.dumps(
                dict(
                    id=str(key),
                    fork=fork,
                    step=t,
                    **{
                        k: v[i] if math.isfinite(v[i]) else None
                        for k, v in values.items()
                    },
                ),
                allow_nan=False,
            )
            + "\n"
        )
