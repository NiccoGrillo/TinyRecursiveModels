"""Construct and certify paired worlds without querying a learned model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import numpy as np

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()

def valid(sol, givens):
    v=np.asarray(sol).reshape(9,9)
    return (all(c==0 or s==c for c,s in zip(givens,sol)) and
            all(sorted(g)==list(range(1,10)) for g in [*v,*v.T,*v.reshape(3,3,3,3).transpose(0,2,1,3).reshape(9,9)]))

def generate(args):
    m=json.loads(args.cohort.read_text())
    assert sha(args.data.parent/"dataset.json")==m["source_manifest"]
    assert sha(args.data)==json.loads((args.data.parent/"dataset.json").read_text())["splits"]["train"]["sha256"]
    prior=json.loads(args.intervention.read_text())["rows"]
    excluded={r["id"] for r in prior}.union(*(m[k] for k in ("original_ids","dev_ids","heldout_ids")))
    z=np.load(args.data,allow_pickle=False); ids=z['ids'].tolist(); inputs=z['x']; targets=z['y']+1
    eligible=sorted((i for i,g in enumerate(ids) if g not in excluded),key=lambda i:digest('ph4-master/'+ids[i]))
    pool=eligible[:-2048] if args.training else eligible[-2048:]
    solver=subprocess.Popen([args.solver],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    counts=Counter(); rows=[]; meta=[]; seen=set()
    def solve(puzzle,reverse=False):
        start=time.perf_counter_ns()
        solver.stdin.write(''.join(map(str,puzzle))+f' {int(reverse)} 300000\n');solver.stdin.flush()
        result=solver.stdout.readline().split()
        elapsed=(time.perf_counter_ns()-start)*1e-6
        capped,nodes,n=map(int,result[:3])
        sols=[list(map(int,s)) for s in result[3:]]
        assert len(sols)==n and all(valid(s,puzzle) for s in sols)
        return capped,sols,nodes,elapsed
    for i in pool:
        if len(rows)==256: break
        counts['parents_considered']+=1
        original=inputs[i].tolist(); rng=random.Random(digest("viable/"+ids[i]))
        for attempt in range(4):
            board=original.copy(); order=np.flatnonzero(original).tolist();rng.shuffle(order)
            accepted=False
            for pivot in order:
                original_digit=board[pivot];board[pivot]=0
                capped,sols,_,_=solve(board)
                if capped: counts['capped_attempts']+=1;break
                if len(sols)==1: continue
                if len(sols)!=2: counts['three_or_more_attempts']+=1;break
                assert sols[0][pivot]!=sols[1][pivot]
                sols.sort(key=lambda s:s[pivot]!=original_digit)
                assert sols[0]==targets[i].tolist()
                key=tuple(board)
                if key in seen: break
                nodes=[]; times=[]
                for s in sols:
                    case=board.copy();case[pivot]=s[pivot]
                    capped,confirmed,n,ms=solve(case,True)
                    if capped or confirmed!=[s]: break
                    nodes.append(n);times.append(ms)
                if len(nodes)!=2: counts['failed_confirmation']+=1;break
                seen.add(key)
                rows.append(dict(id=ids[i],board=board,pivot=pivot,solutions=sols))
                meta.append(dict(id=ids[i],pivot=pivot,removed=sum(a!=b for a,b in zip(board,original)),
                    different=sum(a!=b for a,b in zip(*sols)),conditional_search_nodes=nodes,conditional_certification_ms=times))
                accepted=True;break
            if accepted: break
    solver.stdin.close(); assert solver.wait()==0 and len(rows)>=1
    args.out.write_text(json.dumps(rows,separators=(',',':')))
    args.manifest.write_text(json.dumps(dict(n=len(rows),counts=counts,source_sha256=sha(args.data),
        pairs_sha256=sha(args.out),enumerator_sha256=sha(Path(__file__).with_name('enumerate.cpp')),rows=meta),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('solver','data','cohort','intervention','out','manifest'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--training',action='store_true')
    generate(p.parse_args())
