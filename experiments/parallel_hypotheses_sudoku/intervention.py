"""Fresh frozen-trunk intervention. Use PYTHONPATH=.:scripts from the checkout."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluate_matched_sudoku import read_boards
from run_trm_hypothesis_pilot import (
    Adapter, State, WorkspaceEdit, batch_for, hypothesis_choices, model_from_checkpoint, rollout, sha,
)


def write_states(anchor, peer, sites, identifier):
    if len(sites) < 2:
        return anchor, anchor, (0., 0.), "too_few"
    offset = 1 + int(identifier[:16], 16) % (len(sites) - 1)
    original = anchor.z_L[:, sites].float()
    delta = (peer.z_L[:, sites].float() - original) * .5
    states, norms = [], []
    for correction in (delta, delta.roll(offset, dims=1)):
        low = anchor.z_L.clone()
        low[:, sites] = (original + correction).to(low.dtype)
        norms.append((low[:, sites].float()-original).norm().item())
        states.append(State(anchor.z_H, low))
    if min(norms) == 0 or max(norms)/min(norms) > 1.05:
        return anchor, anchor, norms, "dose"
    return *states, norms, "active"


def apply(adapter, edit, x, identifier):
    batch = batch_for(x)
    start, prefix = rollout(adapter, adapter.initial(batch), batch, 12)
    cell,digits = hypothesis_choices(x,prefix[-1])
    ends, scores = [], []
    for digit in digits.unbind(1):
        state, out = rollout(adapter, edit(start,prefix[-1],cell,digit), batch, 2)
        ends.append(state)
        scores.append(out[-1].float())
    blank = x[0].eq(1)
    confidence = [s.log_softmax(-1).amax(-1)[0,blank].mean().item() for s in scores]
    anchor = int(confidence[1]>confidence[0])
    a,b = [s.argmax(-1) for s in scores]
    indices = torch.where(blank & a[0].ne(b[0]))[0]
    sites = indices + ends[0].z_L.shape[1] - x.shape[1]
    aligned, shuffled, norms, status = write_states(ends[anchor],ends[1-anchor],sites,identifier)
    arms = {"aligned":aligned, "shuffled":shuffled, "untouched":ends[anchor]}
    first, last = {}, {}
    times = {}
    def sync():
        if x.is_cuda:
            torch.cuda.synchronize()
    for name,state in arms.items():
        assert torch.equal(adapter.readout(state).float(),scores[anchor])
        sync()
        tick=time.perf_counter()
        _, out = rollout(adapter,state,batch,2)
        first[name], last[name] = [s.argmax(-1) for s in out]
        sync()
        times[name]=time.perf_counter()-tick
    _, serial = rollout(adapter,start,batch,6)
    last.update(serial=serial[-1].argmax(-1),pool=(scores[0]+scores[1]).argmax(-1),a=a,b=b)
    return last, dict(id=identifier,sites=len(sites),status=status,norms=norms,anchor=anchor,
        first_changes={k:int((v!=first["untouched"]).sum()) for k,v in first.items()},continuation_seconds=times)


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    device=torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    manifest=json.loads(args.cohort.read_text())
    excluded=set().union(*(manifest[k] for k in ("original_ids","dev_ids","heldout_ids")))
    x,y,ids,data_sha=read_boards(args.data,"train","trm",slice(None),"cpu")
    assert len(set(ids)) == len(ids) == 8192 and len(manifest["original_ids"])==1000
    rank=lambda i: hashlib.sha256(("ph3-v1/"+ids[i]).encode()).digest()
    order=sorted((i for i,g in enumerate(ids) if g not in excluded),key=rank)[:256]
    assert len(order)==256
    audit=json.loads(args.reference.read_text())
    assert sha(args.weights)==audit["native_sha256"] and sha(args.edit)==audit["edit_sha256"]
    calibration=json.loads(args.calibration.read_text())
    assert sha(args.config)==calibration["config_sha256"]
    if device.type == "cuda":
        assert torch.cuda.get_device_name()==calibration["device"] and str(torch.__version__)==calibration["torch"]
    native,_=model_from_checkpoint(SimpleNamespace(weights=args.weights,config=args.config,family="trm"), device)
    adapter=Adapter(native)
    edit=WorkspaceEdit(native.config.hidden_size).eval()
    edit.load_state_dict(torch.load(args.edit,map_location="cpu",weights_only=True))
    edit.to(device)
    rows=[]
    start=time.perf_counter()
    for j,i in enumerate(order):
        predictions,row=apply(adapter,edit,x[i:i+1].to(device),ids[i])
        label=y[i:i+1].to(device)
        row["exact"]={k:bool(v.eq(label).all()) for k,v in predictions.items()}
        rows.append(row)
        if (j+1)%32==0:
            print(j+1,flush=True)
    summary=dict(rows=rows,train_file_sha256=data_sha,cohort_manifest_sha256=sha(args.cohort),
        split_sha256=hashlib.sha256("\n".join(ids[i] for i in order).encode()).hexdigest(),
        elapsed=time.perf_counter()-start,device=str(device),torch=str(torch.__version__),
        native_sha256=sha(args.weights),edit_sha256=sha(args.edit),source_sha256=sha(Path(__file__)),
        serial_or_branch_blocks_after_prefix=6*adapter.block_calls_per_macro)
    args.out.write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data","cohort","weights","edit","config","reference","calibration","out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    run(p.parse_args())
