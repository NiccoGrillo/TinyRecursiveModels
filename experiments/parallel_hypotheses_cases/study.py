"""End-to-end conditional table study. Invoke with python -m experiments.parallel_hypotheses_cases.study."""
import argparse
import csv
import hashlib
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

SEEDS = (31, 47, 83)
PERMS = list(itertools.permutations(range(4)))
HERE = Path(__file__).resolve().parent


def split_groups():
    groups = list(itertools.combinations_with_replacement(range(24), 2))
    indices = np.random.default_rng(9701).permutation(len(groups))
    return [[groups[i] for i in part] for part in np.split(indices, [200, 250])]


def materialize(groups):
    rows = []
    for group, (a, b) in enumerate(groups):
        for left, right in sorted({(a, b), (b, a)}):
            for q, r0, r1 in itertools.product(range(4), repeat=3):
                equal = PERMS[left][q] % 2 == PERMS[right][q] % 2
                rows.append((group, left, right, q, r0, r1, r1 if equal else r0))
    return np.asarray(rows, dtype=np.int64)


class Cases(nn.Module):
    def __init__(self, selection):
        super().__init__()
        self.address = nn.Parameter(torch.randn(4, 4) * .1)
        self.values = nn.Parameter(torch.randn(4, 8) * .2)
        self.join = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
        self.selection = selection
        if selection:
            self.proposals = nn.Linear(8, 1, bias=False)
            self.roles = nn.Parameter(torch.tensor([-.1, .1]))

    def read(self, tables, query):
        rows = self.address[query].softmax(-1)
        symbols = (tables * rows[:, None, :, None]).sum(dim=2)
        return symbols @ self.values

    def combine(self, memories, outputs):
        a, b = memories.unbind(1)
        weight = self.join(torch.cat([a+b, a*b], -1)).sigmoid()
        proposals = None
        if self.selection:
            proposals = (self.proposals(memories).squeeze(-1) + self.roles).sigmoid()
            weight = ((1-weight) * proposals[:, :1] + weight * proposals[:, 1:])
        return (1-weight)*outputs[:, 0] + weight*outputs[:, 1], proposals

    def forward(self, tables, query, outputs, serial=False):
        if serial:
            memories = torch.cat([self.read(tables[:, i:i+1], query) for i in range(2)], 1)
        else:
            memories = self.read(tables, query)
        return self.combine(memories, outputs)[0]


def inputs(rows, device):
    data = torch.as_tensor(rows.copy(), device=device)
    eye = torch.eye(4, device=device)
    constants = eye[torch.as_tensor(PERMS, device=device)]
    # Expand supplied table contents; IDs and grouping never enter the network.
    return constants[data[:, 1:3]], data[:, 3], eye[data[:, 4:6]], data[:, 6]


def donor_indices(rows):
    by_input = {tuple(row[1:6]): i for i, row in enumerate(rows)}
    recipient, active, placebo = [], [], []
    for i, (_, a, b, q, r0, r1, _) in enumerate(rows):
        if r0 == r1:
            continue
        others = [(d, by_input[(a,d,q,r0,r1)]) for d in range(24)
                  if d != b and (a,d,q,r0,r1) in by_input]
        changed = [idx for d,idx in others if PERMS[d][q]%2 != PERMS[b][q]%2]
        stable = [idx for d,idx in others if PERMS[d][q]%2 == PERMS[b][q]%2]
        if changed and stable:
            recipient.append(i)
            active.append(changed[0])
            placebo.append(stable[0])
    return [np.array(v, dtype=int) for v in (recipient, active, placebo)]


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def train(model, optimizer, rng, train_inputs, first, last, curve, device):
    t, q, r, y = train_inputs
    synchronize(device)
    start = time.perf_counter()
    for step in range(first, last+1):
        idx = rng.integers(len(y), size=128)
        prob = model(t[idx], q[idx], r[idx]).gather(1,y[idx,None])
        loss = -prob.clamp_min(1e-12).log().mean()
        optimizer.zero_grad()
        loss.backward()
        if step == 1:
            assert all(p.grad.isfinite().all() and p.grad.norm() > 0 for p in model.parameters())
        optimizer.step()
        if step % 400 == 0:
            with torch.no_grad():
                correct = model(t,q,r).argmax(1) == y
                curve.append(dict(step=step, train_exact=correct.float().mean().item(), loss=loss.item()))
    synchronize(device)
    return time.perf_counter()-start


@torch.inference_mode()
def evaluate(model, fold, donors, device):
    t,q,r,y = fold
    mem = model.read(t,q)
    prob, proposals = model.combine(mem,r)
    pred = prob.argmax(1)
    serial = model(t,q,r,True)
    assert torch.equal(pred, serial.argmax(1)) and torch.allclose(prob, serial, atol=1e-6)
    orig, active, placebo = donors
    meta = {"serial_max_abs": (serial-prob).abs().max().item(), "timings_us": {}}
    for donor, name in [(active,"active"), (placebo,"placebo")]:
        mixed = torch.stack([mem[orig,0], mem[donor,1]], dim=1)
        out = model.combine(mixed,r[orig])[0].argmax(1)
        intact = (pred[orig]==y[orig]) & (pred[donor]==y[donor])
        meta[name] = dict(n=len(orig), both_intact=int(intact.sum()),
                          correct_donor=int((out==y[donor]).sum()),
                          conditional_correct=int(((out==y[donor]) & intact).sum()),
                          changed=int((out!=pred[orig]).sum()))
    candidate = None
    if proposals is not None:
        candidate = r.argmax(2).gather(1,(proposals > .5).long()).cpu().numpy()
    for n,serial_schedule in itertools.product((1,128),(False,True)):
        args = t[:n],q[:n],r[:n],serial_schedule
        for _ in range(5):
            model(*args)
        ticks=[]
        for _ in range(7):
            synchronize(device)
            tick=time.perf_counter_ns()
            for _ in range(30):
                model(*args)
            synchronize(device)
            ticks.append((time.perf_counter_ns()-tick)/30000)
        meta["timings_us"][f"{n}/{'serial' if serial_schedule else 'parallel'}"]=float(np.median(ticks))
    return pred.cpu().numpy(), candidate, meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    groups=split_groups()
    train_rows, dev_rows = (materialize(part) for part in groups[:2])
    train_data, dev_data = (inputs(rows,args.device) for rows in (train_rows,dev_rows))
    donors=donor_indices(dev_rows)
    actors=[]
    for seed,selector in itertools.product(SEEDS,(False,True)):
        torch.manual_seed(seed)
        net=Cases(selector).to(args.device)
        opt=torch.optim.Adam(net.parameters(), lr=.01)
        rng=np.random.default_rng(seed+9701)
        curve=[]
        elapsed=train(net,opt,rng,train_data,1,1200,curve,args.device)
        actors.append((seed,net,opt,rng,curve,elapsed))
    extend=any(c[-1]["train_exact"]-c[-2]["train_exact"]>=.02 for _,_,_,_,c,_ in actors)
    groups_primary=dev_rows[:,4]!=dev_rows[:,5]
    target=dev_rows[:,6]
    scores=[]
    observations=[]
    with (args.out/"predictions.csv").open("w") as stream:
        writer=csv.writer(stream)
        writer.writerow(["seed","arm","group","A","B","q","R0","R1","target","pred","candidate_a","candidate_b"])
        for seed,net,opt,rng,curve,elapsed in actors:
            if extend:
                elapsed+=train(net,opt,rng,train_data,1201,2400,curve,args.device)
            pred,candidates,meta=evaluate(net,dev_data,donors,args.device)
            arm="selection" if net.selection else "join"
            correct=pred==target
            group_scores=[correct[(dev_rows[:,0]==i)&groups_primary].mean() for i in range(50)]
            scores.append(group_scores)
            oracle=None
            if candidates is not None:
                oracle=int((candidates==target[:,None]).any(1)[groups_primary].sum())
            for i,row in enumerate(dev_rows):
                writer.writerow([seed,arm,*row,pred[i],*(candidates[i] if candidates is not None else ("",""))])
            observations.append(dict(seed=seed,arm=arm,curve=curve,train_seconds=elapsed,
                primary_correct=int(correct[groups_primary].sum()), group_exact=float(np.mean(group_scores)),
                constant_correct=int(correct[~groups_primary].sum()), candidate_oracle=oracle,
                params=sum(v.numel() for v in net.parameters()), **meta))
            # Tiny models; raw parameter arrays permit exact recovery without pickle.
            (args.out/f"{arm}-{seed}.json").write_text(json.dumps({k:v.detach().cpu().tolist() for k,v in net.named_parameters()}))
    deltas=np.asarray(scores)[::2]-np.asarray(scores)[1::2]
    boot=deltas.mean(0)[np.random.default_rng(9703).integers(50,size=(2000,50))].mean(1)
    summary=dict(seeds=SEEDS,extended=extend,mean_gain=deltas.mean(),seed_gains=deltas.mean(1).tolist(),
        interval=np.quantile(boot,[.025,.975]).tolist(), groups=groups,primary_n=int(groups_primary.sum()),
        negative_n=int((~groups_primary).sum()),train_n=len(train_rows),torch=str(torch.__version__),numpy=np.__version__,
        device=args.device,machine=platform.platform(),base=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in (train_rows,dev_rows)], observations=observations)
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    main()
