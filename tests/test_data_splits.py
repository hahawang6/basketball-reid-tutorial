"""独立测试保留、身份隔离和内容变化的检查。"""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_splits import create_reid_split,validate_reid_split,identity

class SplitTests(unittest.TestCase):
    def fixture(self,root):
        for directory in ('bounding_box_train','query','bounding_box_test'):
            (root/directory).mkdir()
        for pid in range(1,7):
            for frame in range(3):
                p=root/'bounding_box_train'/f'{pid:04d}_c1s1_{frame:06d}_00.jpg'
                p.write_bytes(f'{pid}-{frame}'.encode())
        for pid in (100,101):
            for directory,frame in [('query',0),('bounding_box_test',1)]:
                p=root/directory/f'{pid:04d}_c1s1_{frame:06d}_00.jpg'
                p.write_bytes(f'{pid}-{frame}'.encode())
    def test_identity_holdout_and_unchanged_official_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            manifest=create_reid_split(root)
            parts=validate_reid_split(manifest,root)
            train={identity(p) for p in parts['train']}
            val={identity(p) for p in parts['validation_query']}
            self.assertFalse(train&val)
            self.assertEqual(len(parts['test_query']),2)
            self.assertEqual(manifest,create_reid_split(root))
            parts['train'][0].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'content changed'):
                validate_reid_split(manifest,root)
    def test_cross_split_identical_content_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            q=next((root/'query').glob('*.jpg'))
            g=next((root/'bounding_box_test').glob('*.jpg'))
            g.write_bytes(q.read_bytes())
            with self.assertRaisesRegex(ValueError,'Exact image duplicates'):
                create_reid_split(root)

if __name__=='__main__':
    unittest.main()
