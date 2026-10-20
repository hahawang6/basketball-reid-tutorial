"""用原项目的余弦距离与身份评价规则评价完整 Query/Gallery。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weight', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--split', choices=('validation','test'), default='test')
    parser.add_argument('--tta', action='store_true', help='Average original and horizontal-flip features as in original training evaluation')
    args = parser.parse_args()
    from training_config import load_training_config
    from data_splits import load_reid_split, identity, sha256
    from checkpoints import load_checkpoint
    from dataset import parse_data_dir, BallShowDataset, build_transforms
    from evaluate import evaluate, compute_dist_matrix
    from model import SwinReID
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    config = load_training_config(args.config, 'swin')
    split = load_reid_split(config['split_file'],config['data_root'])
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; choose a compatible environment or --device cpu')
    ckpt = load_checkpoint(args.weight)
    model = SwinReID(ckpt['num_classes'], ckpt.get('num_stripes', 6), ckpt.get('reduce_dim', 512), freeze_stages=0, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(args.device).eval()
    def extract(paths):
        data = [(str(p),identity(p),0) for p in paths]
        if not data:
            raise ValueError('Empty image split')
        loader = DataLoader(BallShowDataset(data, build_transforms(False)), batch_size=8, shuffle=False, num_workers=0)
        features, pids, cameras = [], [], []
        with torch.inference_mode():
            for images, pid, camera in loader:
                images = images.to(args.device)
                feature = model(images)
                if args.tta:
                    feature = F.normalize(feature + model(torch.flip(images, dims=[3])), dim=1)
                features.append(feature.cpu().numpy())
                pids.extend(pid.tolist()); cameras.extend(camera.tolist())
        return np.concatenate(features), np.asarray(pids), np.asarray(cameras)
    prefix = 'validation' if args.split == 'validation' else 'test'
    if ckpt.get('split_signature') != sha256(config['split_file']):
        raise ValueError('Evaluation requires the training split manifest recorded in the checkpoint')
    qf, qp, qc = extract(split[prefix+'_query'])
    gf, gp, gc = extract(split[prefix+'_gallery'])
    cmc, mean_ap = evaluate(compute_dist_matrix(qf, gf), qp, gp, qc, gc)
    report = {'split': args.split, 'query_count':len(qp), 'gallery_count':len(gp),
              'queries_with_gallery_match':int(np.isin(qp, gp).sum()),
              'rank1':float(cmc[0]), 'mAP':float(mean_ap),
              'tta':args.tta, 'reranking':False,
              'protocol':'PID equality; no same-camera exclusion; queries without positives are skipped, matching recovered evaluate.py'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
