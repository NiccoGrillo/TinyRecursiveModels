import argparse
from pathlib import Path
import json
import time
from types import SimpleNamespace
import torch
from run_trm_hypothesis_pilot import Adapter, State, model_from_checkpoint, batch_for, sha

def load_adapter(weights,config,device):
    native,_=model_from_checkpoint(SimpleNamespace(weights=weights,config=config,family="trm"),device)
    native.requires_grad_(False)
    return Adapter(native)

def steps(adapter,state,batch,n):
    for _ in range(n):
        state,logits=adapter.step(state,batch)
    return state,logits

def example(row,device):
    base=torch.tensor([row["board"]],device=device)+1
    pivot=row["pivot"]
    target=torch.tensor(row["solutions"],device=device)+1
    conditioned=base.expand(2,-1).clone()
    conditioned[:,pivot]=target[:,pivot]
    return base,conditioned,target,pivot

def duplicate(state):
    return State(state.z_H.expand(2,-1,-1).contiguous(),state.z_L.expand(2,-1,-1).contiguous())

def score(logits,target,base,pivot):
    pred=logits.argmax(-1) if logits.ndim==3 else logits
    changed=target[0]!=target[1]
    changed[pivot]=False
    same=(target[0]==target[1]) & (base[0]==1)
    idx=torch.arange(81,device=target.device)
    far=(idx//9!=pivot//9) & (idx%9!=pivot%9) & ((idx//27!=pivot//27) | (idx%9//3!=pivot%9//3))
    ok=pred==target
    return dict(exact=ok.all(1).tolist(),both=bool(ok.all()),changed_correct=int(ok[:,changed].sum()),
        changed_n=2*int(changed.sum()),changed_both=bool(ok[:,changed].all()),
        far_correct=int(ok[:,far & changed].sum()),far_n=2*int((far & changed).sum()),
        stable_correct=int(ok[:,same].sum()),stable_n=2*int(same.sum()))

@torch.inference_mode()
def run(args):
    device=torch.device(args.device); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    adapter=load_adapter(args.weights,args.config,device)
    assert sha(args.weights)=="533999676a1d44bdb84205461b55ecf1719f238cad2c2af699de78128e3dbf7f"
    assert sha(args.data)==json.loads(args.manifest.read_text())['pairs_sha256']
    rows=[]; start=time.perf_counter()
    for row in json.loads(args.data.read_text()):
        base,x,y,pivot=example(row,device)
        b0=batch_for(base); b=batch_for(x)
        init=adapter.initial(b)
        parent,parent_score=steps(adapter,adapter.initial(b0),b0,8)
        parent2=duplicate(parent)
        out={}
        def record(name,logits):out[name]=score(logits,y,base,pivot)
        _,uncond=steps(adapter,parent,b0,8)
        blind=uncond.argmax(-1).expand(2,-1).clone()
        blind[:,pivot]=x[:,pivot]
        record("blind",blind)
        cold,z=steps(adapter,init,b,8);record("cold8",z)
        cold,z=steps(adapter,cold,b,4);record("cold12",z)
        _,z=steps(adapter,cold,b,4);record("cold16",z)
        keep=(parent_score[0,pivot].argmax()==x[:,pivot])[:,None,None]
        hard=State(torch.where(keep,parent2.z_H,init.z_H),torch.where(keep,parent2.z_L,init.z_L))
        for name,s in (("warm",parent2),("reset_answer",State(init.z_H,parent2.z_L)),
                       ("reset_workspace",State(parent2.z_H,init.z_L)),("hard_reset",hard)):
            _,z=steps(adapter,s,b,8)
            record(name,z)
        diff=y[0]!=y[1]
        rows.append(dict(id=row["id"],pivot=pivot,affected_digits=len(set(y[:,diff].flatten().tolist())),
            keep=keep.flatten().tolist(),scores=out))
        if len(rows)%32==0:print(len(rows),flush=True)
    args.out.write_text(json.dumps(dict(rows=rows,wall_seconds=time.perf_counter()-start,torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name() if device.type=="cuda" else str(device),sha=sha(args.weights),
        data_sha=sha(args.data),source_sha=sha(Path(__file__))),separators=(',',':'))+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('weights','config','data','manifest','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='cuda')
    run(p.parse_args())
