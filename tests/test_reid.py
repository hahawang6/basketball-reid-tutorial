"""身份编号、身份采样和检索指标的基本约定。"""
import ast
from collections import defaultdict
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def function(filename, name, namespace):
    tree = ast.parse((ROOT/'src'/filename).read_text())
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


class ReIDContracts(unittest.TestCase):
    def test_query_and_gallery_keep_original_identity(self):
        parse = function('dataset.py', 'parse_data_dir', {'os':os})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for sub, ids in [('q',[2,8]),('g',[1,2,8])]:
                (root/sub).mkdir()
                for pid in ids:
                    (root/sub/f'{pid:04}_c1s1_000001_00.jpg').touch()
            q, _ = parse(root/'q', relabel=False)
            g, _ = parse(root/'g', relabel=False)
            self.assertEqual([item[1] for item in q], [2,8])
            self.assertEqual([item[1] for item in g], [1,2,8])
            training, classes = parse(root/'q')
            self.assertEqual([item[1] for item in training], [0,1])
            self.assertEqual(classes, 2)

    def test_sampler_cannot_make_a_single_identity_batch(self):
        sampler = function('dataset.py','RandomIdentitySampler',{'Sampler':object,'defaultdict':defaultdict,'random':random})
        data = [('a',0,0),('b',1,0)]
        with self.assertRaises(ValueError):
            sampler(data,4,4)
        result = list(sampler(data,4,2))
        self.assertEqual(sorted(data[i][1] for i in result), [0,0,1,1])

    def test_retrieval_metrics_use_shared_pid_and_actual_rank(self):
        import numpy as np
        evaluate = function('evaluate.py', 'evaluate', {'np': np})
        cmc, mean_ap = evaluate(np.array([[.1,.2,.3],[.3,.2,.1]]),
                                np.array([2,8]),np.array([1,2,8]),
                                np.zeros(2),np.zeros(3))
        self.assertAlmostEqual(float(cmc[0]), .5)
        self.assertAlmostEqual(mean_ap, .75)
        with self.assertRaises(ValueError):
            evaluate(np.array([[.1]]),np.array([2]),np.array([1]),
                     np.zeros(1),np.zeros(1))


if __name__ == "__main__":
    unittest.main()
