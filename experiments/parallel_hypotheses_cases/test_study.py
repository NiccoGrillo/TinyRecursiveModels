import unittest

import numpy as np
import torch

from experiments.parallel_hypotheses_cases.study import Cases, inputs, materialize, split_groups


class ConditionalWorldTests(unittest.TestCase):
    def test_split_keeps_each_unordered_pair_together(self):
        folds = split_groups()
        self.assertEqual(list(map(len, folds)), [200, 50, 50])
        self.assertEqual(len(set().union(*folds)), 300)
        self.assertTrue(all(a <= b for fold in folds for a,b in fold))

    def test_labels_do_not_enter_inputs(self):
        rows = materialize([(0, 1)])
        changed = rows.copy()
        changed[:, 6] = (rows[:, 6]+1) % 4
        a = inputs(rows,"cpu")
        b = inputs(changed,"cpu")
        self.assertTrue(all(torch.equal(v,w) for v,w in zip(a[:3], b[:3])))
        self.assertFalse(torch.equal(a[-1],b[-1]))

    def test_only_selected_world_changes_at_read(self):
        rows = materialize([(0,1),(0,2)])
        a = rows[[0]].copy()
        b = a.copy()
        b[:,2] = 23
        net = Cases(False)
        t,q,r,_ = inputs(np.vstack([a,b]),"cpu")
        mem = net.read(t,q)
        self.assertTrue(torch.equal(mem[0,0], mem[1,0]))
        self.assertTrue(torch.equal(net(t,q,r), net(t,q,r,True)))
        self.assertTrue(torch.allclose(net.combine(torch.stack([mem[:1,0],mem[1:,1]],1),r[:1])[0],
                                       net(t[1:],q[1:],r[1:])))

    def test_shared_initial_weights_and_constant_control(self):
        torch.manual_seed(31)
        a = Cases(False)
        torch.manual_seed(31)
        b = Cases(True)
        for key,v in a.state_dict().items():
            self.assertTrue(torch.equal(v,b.state_dict()[key]))
        rows = materialize([(0,13)])
        rows = rows[rows[:,4]==rows[:,5]]
        t,q,r,y = inputs(rows,"cpu")
        self.assertTrue(torch.equal(b(t,q,r).argmax(1),y))
