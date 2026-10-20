"""训练配置、设备与输出目录。"""
import argparse
import json
from pathlib import Path


def parse_training_args(task):
    parser = argparse.ArgumentParser(description=f'Train {task}')
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--allow-pretrained-download', action='store_true')
    parser.add_argument('--resume', help='Resume a complete checkpoint from the same run')
    return parser.parse_args()


def load_training_config(path, task):
    path = Path(path).resolve()
    if task != 'swin':
        raise ValueError('Only swin is supported')
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))[task]
    cfg.setdefault('seed', 42)
    cfg.setdefault('eval_batch_size', 8)
    cfg.setdefault('split_file', 'data/splits/reid.json')
    keys = ('data_root', 'output_dir', 'split_file')
    for key in keys:
        value = Path(cfg[key]).expanduser()
        cfg[key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    for key in ('epochs', 'batch_size', 'eval_batch_size'):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if task == 'swin':
        if cfg['eval_every'] < 1 or cfg['epochs'] < cfg['eval_every']:
            raise ValueError('epochs must reach eval_every to produce a best checkpoint')
        k = cfg['num_instances']
        if k < 2 or cfg['batch_size'] % k or cfg['batch_size'] // k < 2:
            raise ValueError('A batch needs at least two identities and two images per identity')
    if cfg['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive')
    return cfg


def prepare_output(cfg, device, task, resume=None):
    out = Path(cfg['output_dir'])
    if resume:
        expected = out / 'latest_checkpoint.pth'
        if Path(resume).resolve() != expected.resolve() or not expected.is_file():
            raise ValueError(f'Resume must point to the complete checkpoint in output_dir: {expected}')
        saved = json.loads((out/'configuration.json').read_text(encoding='utf-8'))
        path_keys = {'output_dir', 'data_root', 'split_file', 'device'}
        for key, value in cfg.items():
            if key not in path_keys and saved.get(key) != value:
                raise ValueError(f'Keep {key} unchanged when resuming; use the original configuration')
    else:
        out.mkdir(parents=True, exist_ok=False)
        with (out/'configuration.json').open('x', encoding='utf-8') as handle:
            json.dump(dict(cfg, device=device), handle, ensure_ascii=False, indent=2)


def prepare_run(cfg, device, task, resume=None):
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; check the installed PyTorch build or choose --device cpu')
    if task == 'swin':
        if not Path(cfg['split_file']).is_file():
            raise FileNotFoundError('Run tools/prepare_reid_split.py first: '+cfg['split_file'])
        required = [Path(cfg['data_root'])/name for name in ('bounding_box_train','query','bounding_box_test')]
        if any(not p.is_dir() or not any(p.glob('*.jpg')) for p in required):
            raise FileNotFoundError('Training, Query and Gallery directories must contain JPG images')
    prepare_output(cfg, device, task, resume)
