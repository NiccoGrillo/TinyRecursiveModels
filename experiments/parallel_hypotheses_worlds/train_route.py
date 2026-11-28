"""Train tokenwise retention heads through one frozen, full-gradient TRM macro."""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from native import State, batch_for, duplicate, example, load_adapter, score, sha, steps

HERE = Path(__file__).resolve().parent
SEEDS = (17, 43, 107)
ARMS = ("content", "global")


class Router(nn.Module):
    def __init__(self, width, content):
        super().__init__()
        self.content = content
        self.net = nn.Sequential(nn.Linear(3 * width + 2, 32), nn.GELU(), nn.Linear(32, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(.88/.12))

    def forward(self, features, warm, initial):
        # Both rivals execute the same MLP. Zero input makes the global gate constant.
        gates = self.net(features if self.content else torch.zeros_like(features)).sigmoid()
        high = torch.lerp(initial.z_H.float(), warm.z_H.float(), gates[..., :1]).to(warm.z_H.dtype)
        low = torch.lerp(initial.z_L.float(), warm.z_L.float(), gates[..., 1:]).to(warm.z_L.dtype)
        return State(high, low), gates


def features(adapter, warm, logits, inputs, pivot):
    row = torch.arange(len(inputs), device=inputs.device)
    digit = inputs[row, pivot]
    token = adapter.inner.embed_tokens(digit).float()
    probability = logits[row, pivot].float().softmax(-1).gather(1, digit[:,None])
    n = warm.z_H.shape[1]
    position = pivot + n - inputs.shape[1]
    flag = F.one_hot(position, n).float()[...,None]
    def norm(x):
        return F.layer_norm(x.float(), (x.shape[-1],))
    return torch.cat((norm(warm.z_H), norm(warm.z_L), norm(token)[:,None].expand(-1,n,-1),
                      probability[:,None].expand(-1,n,-1), flag), -1)


def supervision(y, other, conditioned, pivots):
    off = torch.arange(y.shape[1],device=y.device)[None] != pivots[:,None]
    blank = conditioned.eq(1) & off
    changed = y.ne(other) & blank
    stable = y.eq(other) & blank
    return changed, stable


def task_loss(logits, y, changed, stable):
    ce = F.cross_entropy(logits.float().transpose(1,2), y, reduction="none")
    losses = []
    for mask in (changed, stable):
        n = mask.sum(1)
        if not bool(n.gt(0).all()):
            raise ValueError("Unrepresented loss class in a world")
        losses.append((ce * mask).sum(1) / n)
    return .5 * (losses[0]+losses[1]).mean()


@torch.no_grad()
def cache(adapter, records, device):
    banks = [[] for _ in range(6)]
    for row in records:
        base, inputs, y, pivot = example(row, device)
        b = batch_for(base)
        warm, logits = steps(adapter, adapter.initial(b), b, 8)
        for group, tensor in zip(banks, (inputs,y,torch.tensor([pivot],device=device),
                                        warm.z_H,warm.z_L,logits)):
            group.append(tensor.cpu())
    return tuple(torch.stack(group)[:, :, ...] for group in banks)


def gather(bank, idx, device):
    parent, world = idx//2, idx%2
    inputs, y, pivot, h, l, logit = bank
    return (inputs[parent,world].to(device), y[parent,world].to(device),
            y[parent,1-world].to(device), pivot[parent,0].to(device),
            State(h[parent,0].to(device),l[parent,0].to(device)),logit[parent,0].to(device))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def diagnose(warm, initial, routed, gates):
    metrics = {}
    for j,(name,out,old,init) in enumerate(zip(("H","L"),(routed.z_H,routed.z_L),
                                             (warm.z_H,warm.z_L),(initial.z_H,initial.z_L))):
        gate = gates[...,j].detach().float()
        metrics[name] = dict(gate_mean=gate.mean().item(), gate_std=gate.std(unbiased=False).item(),
                             gate_min=gate.min().item(), gate_max=gate.max().item(),
                             changed_from_warm=float((out!=old).float().mean()),
                             changed_from_initial=float((out!=init).float().mean()),
                             relative_warm_delta=float((out.float()-old.float()).norm()/old.float().norm().clamp_min(1e-8)))
    return metrics


def train_stage(adapter, bank, state, stop, device):
    net, opt = state["net"], state["opt"]
    sync(device)
    start = time.perf_counter()
    for i in range(len(state["losses"]),stop):
        idx = state["orders"][i]
        x,y,other,pivot,warm,prior = gather(bank,idx,device)
        batch = batch_for(x)
        init = adapter.initial(batch)
        feat = features(adapter,warm,prior,x,pivot)
        routed,gates = net(feat,warm,init)
        _,logits = adapter.step(routed,batch,differentiable=True)
        if i == 0:
            with torch.no_grad():
                _,reference = adapter.step(routed,batch)
                if not torch.equal(logits.detach(),reference):
                    raise RuntimeError("Full-gradient macro differs from native forward")
        changed, stable = supervision(y,other,x,pivot)
        loss = task_loss(logits,y,changed,stable)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norms={k:float(v.grad.norm()) for k,v in net.named_parameters() if v.grad is not None}
        grad=math.sqrt(sum(v*v for v in norms.values()))
        if not math.isfinite(grad) or (i==0 and grad==0):
            raise RuntimeError(f"Invalid routed gradient {grad}")
        opt.step()
        state["losses"].append(float(loss.detach()))
        if i==0 or (i+1)%32==0:
            state["checks"].append(dict(step=i+1,loss=float(loss.detach()),gradient_norms=norms,
                                        activity=diagnose(warm,init,routed,gates)))
    sync(device)
    state["seconds"]+=time.perf_counter()-start
    print(state["seed"],state["arm"],stop,np.mean(state["losses"][-32:]),flush=True)


@torch.no_grad()
def evaluate(adapter, records, states, device):
    rows=[]
    start=time.perf_counter()
    for row in records:
        base,x,y,pivot=example(row,device)
        b0,b=batch_for(base),batch_for(x)
        warm,prior=steps(adapter,adapter.initial(b0),b0,8)
        pair=duplicate(warm)
        init=adapter.initial(b)
        feat=features(adapter,pair,prior.expand(2,-1,-1),x,
                      torch.full((2,),pivot,device=device,dtype=torch.long))
        keep=(prior[0,pivot].argmax()==x[:,pivot])[:,None,None]
        result={}
        def evaluate_state(name,s,depth):
            first, logits=steps(adapter,s,b,1)
            result[name]=dict(first=score(logits,y,base,pivot))
            _,logits=steps(adapter,first,b,depth-1)
            result[name]["final"]=score(logits,y,base,pivot)
        # The blinded branch receives the new pivot digit only at answer construction.
        _,blind=steps(adapter,warm,b0,8)
        pred=blind.argmax(-1).expand(2,-1).clone()
        pred[:,pivot]=x[:,pivot]
        result["blind"]=dict(final=score(pred,y,base,pivot))
        for name,s,depth in (("cold12",init,12),("warm",pair,8),
                             ("reset_answer",State(init.z_H,pair.z_L),8),
                             ("reset_workspace",State(pair.z_H,init.z_L),8),
                             ("hard_reset",State(torch.where(keep,pair.z_H,init.z_H),
                                                 torch.where(keep,pair.z_L,init.z_L)),8)):
            evaluate_state(name,s,depth)
        for state in states:
            key=f'{state["arm"]}-{state["seed"]}'
            routed,gates=state["net"](feat,pair,init)
            evaluate_state(key,routed,8)
            result[key]["activity"]=diagnose(pair,init,routed,gates)
        rows.append(dict(id=row["id"],pivot=pivot,keep=keep.flatten().tolist(),scores=result))
        if len(rows)%32==0:
            print("evaluated",len(rows),flush=True)
    sync(device)
    return rows,time.perf_counter()-start


def compare(rows):
    rng=np.random.default_rng(713)
    boot=rng.integers(len(rows),size=(2000,len(rows)))
    summary={}
    for rival in ("global","hard_reset","cold12","warm","reset_answer","reset_workspace","blind"):
        differences=[]
        paired=[]
        for seed in SEEDS:
            rkey=f"global-{seed}" if rival=="global" else rival
            a=np.array([r["scores"][f"content-{seed}"]["final"]["both"] for r in rows])
            b=np.array([r["scores"][rkey]["final"]["both"] for r in rows])
            differences.append(a.astype(float)-b)
            paired.append(dict(seed=seed,content=int(a.sum()),rival=int(b.sum()),gains=int((a&~b).sum()),losses=int((b&~a).sum())))
        group=np.mean(differences,axis=0)
        summary[rival]=dict(by_seed=paired,delta=float(group.mean()),paired_parent_95=np.quantile(group[boot].mean(1),[.025,.975]).tolist())
    return summary


def main(args):
    if args.seed!="all":
        raise ValueError("Use --seed all so every model shares the extension decision")
    device=torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter()
    args.run.mkdir(parents=True,exist_ok=True)
    assert sha(args.weights)=="533999676a1d44bdb84205461b55ecf1719f238cad2c2af699de78128e3dbf7f"
    def read(path,manifest,n):
        metadata=json.loads(manifest.read_text())
        assert sha(path)==metadata["pairs_sha256"]
        records=json.loads(path.read_text())
        assert len(records)==n==len(set(r["id"] for r in records))
        assert [r["id"] for r in records]==[r["id"] for r in metadata["rows"]]
        return records
    train=read(args.training,HERE/"training-manifest.json",256)
    dev=read(args.data,args.manifest,158)
    assert not {r["id"] for r in train}.intersection(r["id"] for r in dev)
    adapter=load_adapter(args.weights,args.config,device)
    adapter.model.eval()
    bank=cache(adapter,train,device)
    states=[]
    for seed in SEEDS:
        orders=torch.randint(2*len(train),(256,8),generator=torch.Generator().manual_seed(seed+1729))
        for arm in ARMS:
            torch.manual_seed(seed)
            net=Router(adapter.config.hidden_size,arm=="content").to(device)
            states.append(dict(net=net,opt=torch.optim.AdamW(net.parameters(),lr=.001),seed=seed,
                               arm=arm,orders=orders,losses=[],checks=[],seconds=0.0))
    for s in states:
        train_stage(adapter,bank,s,128,device)
    extend=any(np.mean(s["losses"][96:128])<=.95*np.mean(s["losses"][32:64]) for s in states)
    if extend:
        for s in states:
            train_stage(adapter,bank,s,256,device)
    assert all(p.grad is None for p in adapter.model.parameters())
    training=[]
    for s in states:
        file=args.run/f'{s["arm"]}-{s["seed"]}.pt'
        torch.save(s["net"].state_dict(),file)
        training.append(dict(seed=s["seed"],arm=s["arm"],losses=s["losses"],checks=s["checks"],seconds=s["seconds"],
                             order_sha256=hashlib.sha256(s["orders"][:len(s["losses"])].numpy().tobytes()).hexdigest(),
                             parameters=sum(p.numel() for p in s["net"].parameters()),checkpoint_sha256=sha(file)))
    (args.run/"training.json").write_text(json.dumps(dict(extended=extend,training=training),indent=2)+"\n")
    rows,eval_seconds=evaluate(adapter,dev,states,device)
    comparisons=compare(rows)
    advance=all(comparisons[k]["delta"]>=.02 and comparisons[k]["paired_parent_95"][0]>0 for k in ("global","hard_reset")) and comparisons["cold12"]["delta"]>0
    out=dict(rows=rows,comparisons=comparisons,advance=advance,extended=extend,evaluation_seconds=eval_seconds,
             wall_seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else 0,
             torch=str(torch.__version__),gpu=torch.cuda.get_device_name() if device.type=="cuda" else str(device),
             platform=platform.platform(),weights_sha256=sha(args.weights),config_sha256=sha(args.config),
             training_sha256=sha(args.training),data_sha256=sha(args.data),source_sha256=sha(Path(__file__)),
             protocol_sha256=sha(HERE/"route-protocol.md"),training_results_sha256=sha(args.run/"training.json"))
    (args.run/"result.json").write_text(json.dumps(out,separators=(",",":"))+"\n")
    print("registered gate",advance,"comparisons",comparisons,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("weights","config","data","manifest","training","run"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--seed",default="all")
    main(p.parse_args())
