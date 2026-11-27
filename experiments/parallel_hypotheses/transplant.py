"""Evaluate the secondary PH1 matched transplant from saved checkpoints."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from study import D, SEEDS, forward, make_data


def choose_donors(keys):
    perms = np.array(list(itertools.permutations(range(4))))
    selected = []
    for i, (f, g, q) in enumerate(keys):
        if g == 0:
            continue
        other = np.where((keys[:, 1] == g) & (keys[:, 2] == q) & (keys[:, 0] != f))[0]
        distances = np.sum(perms[keys[other, 0]] != perms[f], axis=1)
        changed = perms[keys[other, 0], q] != perms[f, q]
        matches = [(keys[a, 0], keys[b, 0], a, b)
                   for a in other[changed] for b in other[~changed]
                   if distances[np.where(other == a)[0][0]] == distances[np.where(other == b)[0][0]]]
        if matches:
            _, _, active, placebo = min(matches)
            selected.append((i, active, placebo))
    return np.array(selected).T


def scores(p, states, recipients, donors):
    h = states[1][recipients]
    peer = states[1][donors, ::-1]
    mix = np.tanh(h.reshape(-1,D) @ p["U"] + peer.reshape(-1,D) @ p["V"] + p["b"])
    return (mix.reshape(-1,2*D) @ p["R"] + p["c"]).argmax(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    _, data = make_data()
    x, y, keys = data["dev"]
    r, a, b = choose_donors(keys)
    rows, counts = [], []
    for seed in SEEDS:
        with np.load(args.weights / f"dynamic-{seed}.npz") as file:
            p = dict(file)
        logits, states = forward(p, x, True)
        pred = logits.argmax(1)
        assert np.array_equal(pred[r], scores(p, states, r, r))
        assert np.all(y[a] != y[r]) and np.all(y[b] == y[r])
        active = scores(p, states, r, a)
        placebo = scores(p, states, r, b)
        both = (pred[r] == y[r]) & (pred[a] == y[a])
        counts.append(dict(seed=seed, n=len(r), recipient_intact=int((pred[r]==y[r]).sum()),
                           placebo_at_recipient=int((placebo==y[r]).sum()), active_at_recipient=int((active==y[r]).sum()),
                           active_at_donor=int((active==y[a]).sum()), recipient_and_active_donor_correct=int(both.sum()),
                           placebo_stable_and_active_adopts_among_both_correct=int((both & (placebo==y[r]) & (active==y[a])).sum())))
        rows.extend([seed,*keys[i],keys[j,0],keys[k,0],y[i],y[j],pred[i],pred[j],pred[k],v,w]
                    for i,j,k,v,w in zip(r,a,b,active,placebo))
    with (args.out / "transplants.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["seed","F","G","q","active_F","placebo_F","target","active_target","intact","active_intact",
                    "placebo_intact","active_transplant","placebo_transplant"])
        w.writerows(rows)
    (args.out/"transplants.json").write_text(json.dumps(counts,indent=2)+"\n")
    print(counts)


if __name__ == "__main__":
    main()
