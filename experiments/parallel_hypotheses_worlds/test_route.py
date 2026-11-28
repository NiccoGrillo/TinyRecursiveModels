import unittest

import torch

from experiments.decision_stability.model import load_model, smoke_config
from train_route import Router, cache, diagnose, features, gather, supervision, task_loss, train_stage
from native import State, batch_for


class RoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(1)
        config=smoke_config()
        config.update(hidden_size=16,puzzle_emb_ndim=16,forward_dtype="bfloat16")
        self.adapter=load_model(config)
        self.device=torch.device("cpu")
        a=[(i%9)+1 for i in range(81)]
        b=a.copy()
        b[0],b[1]=b[1],b[0]
        self.bank=cache(self.adapter,[dict(board=[0]*81,solutions=[a,b],pivot=0)],self.device)

    def test_parity_gradient_cast_and_cached_state_integrity(self):
        original=[x.clone() for x in self.bank]
        x,y,other,pivot,warm,logits=gather(self.bank,torch.tensor([0,1]),self.device)
        initial=self.adapter.initial(batch_for(x))
        feat=features(self.adapter,warm,logits,x,pivot)
        content=Router(16,True)
        global_gate=Router(16,False)
        global_gate.load_state_dict(content.state_dict())
        a,g=content(feat,warm,initial)
        b,h=global_gate(feat,warm,initial)
        self.assertTrue(torch.equal(g,h) and torch.allclose(g,torch.full_like(g,.88)))
        self.assertTrue(torch.equal(a.z_H,b.z_H) and torch.equal(a.z_L,b.z_L))
        same,_=content(feat,warm,warm)
        self.assertTrue(torch.equal(same.z_H,warm.z_H) and torch.equal(same.z_L,warm.z_L))
        changed,stable=supervision(y,other,x,pivot)
        self.assertEqual(changed.sum().item(),2)
        self.assertEqual(stable.sum().item(),158)
        self.assertTrue(torch.equal(feat[..., -1].sum(1),torch.ones(2)))
        for net in (content,global_gate):
            s=dict(net=net,opt=torch.optim.AdamW(net.parameters(),lr=.001),seed=17,
                   arm="test",orders=torch.tensor([[0,1],[1,0]]),losses=[],checks=[],seconds=0.)
            train_stage(self.adapter,self.bank,s,1,self.device)
            train_stage(self.adapter,self.bank,s,2,self.device)
            d=s["checks"][0]["activity"]
            self.assertGreater(d["H"]["changed_from_warm"],0)
            self.assertGreater(d["L"]["changed_from_warm"],0)
            self.assertGreater(s["checks"][0]["gradient_norms"]["net.2.bias"],0)
        self.assertTrue(all(torch.equal(saved,current) for saved,current in zip(original,self.bank)))
        self.assertTrue(all(p.grad is None for p in self.adapter.model.parameters()))


if __name__=="__main__":
    unittest.main()
