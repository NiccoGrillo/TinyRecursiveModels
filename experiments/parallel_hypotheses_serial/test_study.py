import unittest

import torch

from experiments.parallel_hypotheses.study import make_data
from experiments.parallel_hypotheses_serial.study import SerialLookup, nll


class TestLookup(unittest.TestCase):
    def test_reads_and_order(self):
        _, data = make_data()
        x, y, _ = data["train"]
        inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
        model = SerialLookup()
        with torch.no_grad():
            model.address.copy_(60 * torch.eye(4))
            model.selector.copy_(60 * torch.eye(2))
            pred = model(inputs)
            self.assertTrue(torch.equal(pred.argmax(1), targets))
            self.assertTrue(torch.allclose(pred.sum(1), torch.ones(len(inputs))))
            model.selector.copy_(model.selector.flip([0]))
            self.assertFalse(torch.equal(model(inputs).argmax(1), targets))

    def test_useful_gradients(self):
        torch.manual_seed(31)
        _, data = make_data()
        x, y, _ = data["train"]
        model = SerialLookup()
        loss = nll(model(torch.from_numpy(x[:64])), torch.from_numpy(y[:64]))
        loss.backward()
        for p in model.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.norm().item(), 0)
