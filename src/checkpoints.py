"""受限读取模型检查点。"""
from pathlib import Path
import numpy as np
import torch


def load_checkpoint(path, mmap=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Model weight not found: {path}')
    from numpy._core.multiarray import scalar
    allowed = {'numpy._core.multiarray.scalar', 'numpy.dtype'}
    unknown = set(torch.serialization.get_unsafe_globals_in_checkpoint(str(path))) - allowed
    if unknown:
        raise ValueError(f'Unreviewed checkpoint globals: {sorted(unknown)}')
    types = [type(np.dtype(t)) for t in (np.float32, np.float64, np.int32, np.int64, np.bool_)]
    with torch.serialization.safe_globals([scalar, np.dtype] + types):
        return torch.load(str(path), map_location='cpu', weights_only=True, mmap=mmap)
