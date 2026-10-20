"""使用项目训练权重进行球员图片检索。"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from training_config import load_training_config

def retrieve(config, query, limit, device="cpu"):
    if not Path(query).is_file():
        raise FileNotFoundError(query)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as T
    from model import SwinReID
    from checkpoints import load_checkpoint
    torch.set_num_threads(2)
    checkpoint = load_checkpoint(config['swin_weight'])
    model = SwinReID(checkpoint['num_classes'], checkpoint.get('num_stripes', 6),
                     checkpoint.get('reduce_dim', 512), freeze_stages=0, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    model.to(device).eval()
    norm = T.Normalize([.485,.456,.406], [.229,.224,.225])
    tf = T.Compose([T.Resize((224,224), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), norm])
    transforms = [tf, T.Compose([T.Resize((224,224), interpolation=T.InterpolationMode.BICUBIC),
                    T.RandomHorizontalFlip(p=1), T.ToTensor(), norm]),
                  T.Compose([T.Resize((240,240), interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop((224,224)), T.ToTensor(), norm])]
    paths = sorted(Path(config['gallery']).glob('*.jpg'))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError('No gallery images')
    with torch.inference_mode():
        with Image.open(query) as image:
            q = model(tf(image.convert('RGB')).unsqueeze(0).to(device))
        features = []
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                feature = F.normalize(sum(model(t(rgb).unsqueeze(0).to(device)) for t in transforms), dim=1)
            features.append(feature)
            if (index+1) % 25 == 0:
                print('Gallery:', index+1, '/', len(paths), flush=True)
        sims = (q @ torch.cat(features).T)[0]
        if not torch.isfinite(sims).all():
            raise ValueError('Nonfinite retrieval score')
        indices = torch.argsort(sims, descending=True)[:5].tolist()
    return {'query': Path(query).name, 'gallery_count': len(paths),
            'scope': 'limited_gallery' if limit else 'full_gallery_retrieval',
            'top5': [{'file': paths[i].name, 'cosine_similarity': float(sims[i].item())} for i in indices],
            'feature_protocol': 'single-view query; original, flip and center-crop gallery; cosine similarity'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--weight',required=True)
    parser.add_argument('--query',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--gallery-limit',type=int,default=0)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args()
    if args.gallery_limit<0:
        parser.error('gallery-limit must be nonnegative; zero uses the complete gallery')
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    cfg=load_training_config(args.config,'swin')
    settings={'swin_weight':str(Path(args.weight).resolve()),
              'gallery':str(Path(cfg['data_root'])/'bounding_box_test')}
    result=retrieve(settings,args.query,args.gallery_limit,args.device)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('Output:',output)

if __name__=='__main__':
    main()
