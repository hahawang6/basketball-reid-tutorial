"""按身份隔离的 ReID 清单；保留官方 Query/Gallery 作独立测试。"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

PARTS = ('train', 'validation_query', 'validation_gallery', 'test_query', 'test_gallery')

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    return int(Path(path).name.split('_')[0])


def create_reid_split(root, seed=42, validation_fraction=.2):
    root = Path(root).resolve()
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    files = {name: sorted((root/name).glob('*.jpg')) for name in
             ('bounding_box_train','query','bounding_box_test')}
    if any(not paths for paths in files.values()):
        raise ValueError('All three original image directories must contain JPG images')
    by_pid = defaultdict(list)
    for path in files['bounding_box_train']:
        by_pid[identity(path)].append(path)
    hashes = {p:sha256(p) for paths in files.values() for p in paths}
    eligible = sorted(pid for pid, paths in by_pid.items() if len({hashes[p] for p in paths}) >= 2)
    if len(eligible)<3:
        raise ValueError('Need at least three training identities with distinct image contents')
    rng = random.Random(seed)
    rng.shuffle(eligible)
    n = max(1,min(len(eligible)-2,round(len(eligible)*validation_fraction)))
    validation_ids = set(eligible[:n])
    parts = {'train':[], 'validation_query':[], 'validation_gallery':[],
             'test_query':files['query'], 'test_gallery':files['bounding_box_test']}
    for pid, paths in sorted(by_pid.items()):
        if pid not in validation_ids:
            parts['train'].extend(paths)
            continue
        unique = {}
        for p in paths:
            unique.setdefault(hashes[p], p)
        selected = list(unique.values())
        rng.shuffle(selected)
        parts['validation_query'].append(selected[0])
        parts['validation_gallery'].extend(selected[1:])
    manifest = {'version':1,'task':'reid','seed':seed,'validation_fraction':validation_fraction,
                'protocol':'Identity-disjoint train/validation/test; official query/gallery reserved for test; validation exact duplicates collapsed',
                'parts':{name:[{'path':p.relative_to(root).as_posix(),'sha256':hashes[p]} for p in parts[name]] for name in PARTS}}
    validate_reid_split(manifest,root)
    return manifest


def validate_reid_split(manifest, root):
    root = Path(root).resolve()
    if manifest.get('version')!=1 or manifest.get('task')!='reid':
        raise ValueError('Unsupported ReID split format')
    result, ids, hashes = {}, {}, {}
    for name in PARTS:
        rows = manifest['parts'][name]
        if not rows:
            raise ValueError(f'Empty split: {name}')
        paths = []
        seen = set()
        for row in rows:
            relative = Path(row['path'])
            path = (root/relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root):
                raise ValueError('Split path escapes the dataset root')
            if path in seen:
                raise ValueError(f'Duplicate path in {name}: {relative}')
            seen.add(path)
            if sha256(path)!=row['sha256']:
                raise ValueError(f'Image content changed: {relative}')
            paths.append(path)
        result[name] = paths
        ids[name] = {identity(p) for p in paths}
        hashes[name] = {row['sha256'] for row in rows}
    groups = [ids['train'], ids['validation_query']|ids['validation_gallery'], ids['test_query']|ids['test_gallery']]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Identity leakage across training, validation and test')
    if len(ids['train'])<2:
        raise ValueError('Training requires at least two identities')
    for query,gallery in [('validation_query','validation_gallery'),('test_query','test_gallery')]:
        if ids[query]-ids[gallery]:
            raise ValueError(f'Query identity without a positive in {gallery}')
    for i,name in enumerate(PARTS):
        for other in PARTS[i+1:]:
            if hashes[name]&hashes[other]:
                raise ValueError(f'Exact image duplicates across {name} and {other}')
    return result


def load_reid_split(path, root):
    manifest = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    return validate_reid_split(manifest, root)
