"""检查点写入、随机数状态和输入清单签名。"""
import hashlib
import json
import os
from pathlib import Path
import random


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    import numpy as np
    import torch
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('GPU count differs from the saved run')
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(state, path):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_list_signature(paths, root):
    root = Path(root).resolve()
    records = [(Path(p).resolve().relative_to(root).as_posix(), Path(p).stat().st_size) for p in paths]
    return hashlib.sha256(json.dumps(sorted(records), ensure_ascii=False).encode('utf-8')).hexdigest()


def append_metrics(path, record):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
