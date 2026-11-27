import unittest
import torch

from experiments.decision_stability.model import State
from intervention import write_states


class MatchedWrite(unittest.TestCase):
    def test_aligned_and_rotated(self):
        base=torch.zeros(1,5,2,dtype=torch.bfloat16)
        high=torch.ones_like(base)
        peer=base.clone()
        peer[0,1,0]=2
        peer[0,3,1]=4
        anchor=State(high,base)
        aligned,shuffled,norms,status=write_states(anchor,State(high,peer),torch.tensor([1,3]),"0"*64)
        self.assertEqual(status,"active")
        self.assertEqual(norms[0],norms[1])
        self.assertTrue(torch.equal(aligned.z_L[:,[1,3]],shuffled.z_L[:,[3,1]]))
        self.assertTrue(torch.equal(aligned.z_L[:,[1,3]],peer[:,[1,3]]*.5))
        self.assertTrue(torch.equal(high,shuffled.z_H))
        self.assertEqual(int(base.count_nonzero()),0)
        self.assertEqual(int(shuffled.z_L[:,[0,2,4]].count_nonzero()),0)

    def test_no_single_site_shuffle(self):
        s=State(torch.zeros(1,2,1),torch.ones(1,2,1))
        a,b,_,status=write_states(s,s,torch.tensor([1]),"1"*64)
        self.assertIs(a,s)
        self.assertIs(b,s)
        self.assertEqual(status,"too_few")
