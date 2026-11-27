from copy import deepcopy
import unittest

from experiments.parallel_hypotheses_arc.bridge import choose, evaluate, fit, geometry


class TestBridge(unittest.TestCase):
    def test_transforms_and_preference(self):
        grid = [[1, 2, 3], [4, 5, 6]]
        target = [[4, 1], [5, 2], [6, 3]]
        self.assertEqual(len({str(geometry(grid, i)) for i in range(8)}), 8)
        full, _, _ = choose([dict(input=grid, output=target)])
        self.assertEqual(full[:3], (0, 1, list(range(10))))

    def test_abstain_on_shape_or_color_conflict(self):
        self.assertIsNone(fit([dict(input=[[1, 1]], output=[[2, 3]])], 0))
        self.assertEqual(choose([dict(input=[[1, 2]], output=[[2, 2], [3, 4]])]), (None, None, []))

    def test_test_labels_only_affect_outer_score(self):
        sample = dict(input=[[1, 2, 3], [4, 5, 6]], output=[[4, 1], [5, 2], [6, 3]])
        task = dict(train=[sample], test=[deepcopy(sample)])
        observed = evaluate("fixture", task)
        task["test"][0]["output"] = [[0]]
        mutated = evaluate("fixture", task)
        for policy in ("full", "loo"):
            self.assertTrue(observed[policy]["exact"])
            self.assertFalse(mutated[policy]["exact"])
            for key in ("prediction_sha256", "geometry", "table"):
                self.assertEqual(observed[policy][key], mutated[policy][key])
