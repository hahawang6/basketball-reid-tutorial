"""从原始训练身份中留出验证身份，不移动或修改图片。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from training_config import load_training_config
from data_splits import create_reid_split

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--validation-fraction',type=float,default=.2)
    args=parser.parse_args()
    cfg=load_training_config(args.config,'swin')
    output=Path(cfg['split_file'])
    if output.exists():
        raise FileExistsError(output)
    manifest=create_reid_split(cfg['data_root'],cfg['seed'],args.validation_fraction)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as handle:
        json.dump(manifest,handle,ensure_ascii=False,indent=2)
    print('Split:',output)
    print(json.dumps({k:len(v) for k,v in manifest['parts'].items()}))

if __name__=='__main__':
    main()
