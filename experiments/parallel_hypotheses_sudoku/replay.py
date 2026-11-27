"""Run from repository root with PYTHONPATH=.:scripts. Inputs are development archives."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from run_trm_hypothesis_pilot import (Adapter, State, WorkspaceEdit, batch_for, hypothesis_choices,
                                      model_from_checkpoint, rollout, sha)


@torch.inference_mode()
def run(args):
    torch.set_num_threads(args.threads)
    report = json.loads(args.report.read_text())
    calibration = json.loads(args.calibration.read_text())
    assert sha(args.weights) == report["native_sha256"] == calibration["checkpoint_sha256"]
    assert sha(args.config) == calibration["config_sha256"]
    assert sha(args.edit) == report["checkpoint_sha256"]
    assert sha(args.bank) == calibration["bank_sha256"]
    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    assert bank["ids"] == [r["id"] for r in report["rows"]]
    model, _ = model_from_checkpoint(SimpleNamespace(family="trm",weights=args.weights,config=args.config),
                                     torch.device("cpu"))
    adapter = Adapter(model)
    edit = WorkspaceEdit(model.config.hidden_size).eval()
    edit.load_state_dict(torch.load(args.edit, map_location="cpu", weights_only=True))
    rows = []
    start = time.monotonic()
    for i, archive in enumerate(report["rows"]):
        flags = archive["arms"]["two_hypotheses"]["exact"]
        if (flags[2] or flags[4] or not archive["branches_disagree_at_end"]
                or any(archive["arms"]["serial"]["exact"])):
            continue
        x = bank["inputs"][i:i+1]
        y = bank["labels"][i:i+1]
        batch = batch_for(x)
        state = State(bank["z_H"][i:i+1],bank["z_L"][i:i+1])
        state, prefix = rollout(adapter,state,batch,12-bank["fork"])
        logits = prefix[-1]
        cell,digits = hypothesis_choices(x,logits)
        scores = []
        for digit in digits.unbind(1):
            branch = edit(state,logits,cell,digit)
            _, outputs = rollout(adapter,branch,batch,2)
            scores.append(outputs[-1].float())
        _, serial = rollout(adapter,state,batch,4)
        a,b = (s.argmax(-1) for s in scores)
        ca,cb = a==y,b==y
        avg=(scores[0]+scores[1]).argmax(-1)
        serial_exact=any(s.argmax(-1).eq(y).all().item() for s in [logits,*serial])
        rows.append(dict(id=archive["id"],site_digits_match=cell.item()==archive["cell"] and
                         digits[0].tolist()==archive["proposed"],a_only=int((ca & ~cb).sum()),
                         b_only=int((cb & ~ca).sum()),both_wrong=int((~ca & ~cb).sum()),
                         a_exact=bool(ca.all()),b_exact=bool(cb.all()),agree=bool(torch.equal(a,b)),
                         pooled_exact=bool(avg.eq(y).all()),serial_exact=serial_exact))
        if len(rows)%10==0:
            print(len(rows), round(time.monotonic()-start,1),flush=True)
    args.out.write_text(json.dumps(dict(rows=rows,torch=str(torch.__version__),threads=args.threads,
        wall_seconds=time.monotonic()-start,report_sha256=sha(args.report),bank_sha256=sha(args.bank),
        native_sha256=sha(args.weights),edit_sha256=sha(args.edit)),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("report","calibration","bank","weights","edit","config","out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--threads", type=int, default=4)
    run(p.parse_args())
