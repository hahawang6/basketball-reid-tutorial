"""SwinReID 训练与检查点恢复。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from training_config import load_training_config, parse_training_args, prepare_run


def train(cfg, device, resume=None):
    from training_state import seed_everything, capture_rng, restore_rng, atomic_save, append_metrics
    seed_everything(cfg["seed"])
    resume_state = None
    from checkpoints import load_checkpoint
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    import numpy as np
    from tqdm import tqdm

    from dataset import parse_data_dir, BallShowDataset, build_transforms, RandomIdentitySampler
    from model import SwinReID
    from loss import ReIDLoss
    from evaluate import compute_dist_matrix, evaluate

    DATA_ROOT = cfg["data_root"]
    TRAIN_DIR   = os.path.join(DATA_ROOT, "bounding_box_train")
    SAVE_DIR = cfg["output_dir"]
    os.makedirs(SAVE_DIR, exist_ok=True)

    HEIGHT        = 224
    WIDTH         = 224
    BATCH_SIZE = int(cfg["batch_size"])
    NUM_INSTANCES = int(cfg["num_instances"])
    NUM_EPOCHS = int(cfg["epochs"])
    BASE_LR = float(cfg["learning_rate"])
    WEIGHT_DECAY  = 5e-4
    WARMUP_EPOCHS = 10
    NUM_WORKERS = 0
    NUM_STRIPES = 6
    REDUCE_DIM    = 512
    FREEZE_STAGES = 2           # 初始冻结 stage0 + stage1
    DROPOUT       = 0.1
    EVAL_EVERY = int(cfg["eval_every"])

    # 渐进解冻计划：{epoch: 冻结到第几个 stage（不含）}
    # freeze_until=2 → 冻结 stage0,1；freeze_until=1 → 只冻结 stage0；0 → 全解冻
    UNFREEZE_SCHEDULE = {
        20: 1,   # epoch 20 起：只冻结 stage0（patch_embed）
        40: 0,   # epoch 40 起：全部解冻
    }

    LATEST_CKPT = os.path.join(SAVE_DIR, "latest_checkpoint.pth")
    BEST_CKPT   = os.path.join(SAVE_DIR, "best_model.pth")

    DEVICE = torch.device(device)
    print(f"设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"显卡: {torch.cuda.get_device_name(0)}")

    print("\n加载数据集...")
    from data_splits import load_reid_split, identity, sha256
    split = load_reid_split(cfg['split_file'], DATA_ROOT)
    original, _ = parse_data_dir(TRAIN_DIR, relabel=False)
    allowed = {str(p) for p in split['train']}
    train_data = [(p,pid,cam) for p,pid,cam in original if str(Path(p).resolve()) in allowed]
    mapping = {pid:i for i,pid in enumerate(sorted({pid for _,pid,_ in train_data}))}
    train_data = [(p,mapping[pid],cam) for p,pid,cam in train_data]
    num_classes = len(mapping)
    query_data = [(str(p),identity(p),0) for p in split['validation_query']]
    gallery_data = [(str(p),identity(p),0) for p in split['validation_gallery']]
    print('Model selection uses held-out validation identities; official test is not evaluated during training.')
    print(f"训练: {len(train_data)} 张 / {num_classes} 个 ID")
    print(f"Query: {len(query_data)} 张  Gallery: {len(gallery_data)} 张")

    train_tf = build_transforms(is_train=True,  height=HEIGHT, width=WIDTH)
    test_tf  = build_transforms(is_train=False, height=HEIGHT, width=WIDTH)

    train_loader = DataLoader(
        BallShowDataset(train_data, train_tf),
        batch_size=BATCH_SIZE,
        sampler=RandomIdentitySampler(train_data, BATCH_SIZE, NUM_INSTANCES),
        num_workers=NUM_WORKERS, drop_last=True, pin_memory=True
    )
    query_loader   = DataLoader(BallShowDataset(query_data,   test_tf),
                                batch_size=cfg["eval_batch_size"], shuffle=False, num_workers=NUM_WORKERS)
    gallery_loader = DataLoader(BallShowDataset(gallery_data, test_tf),
                                batch_size=cfg["eval_batch_size"], shuffle=False, num_workers=NUM_WORKERS)
    print(f"训练 batch 数: {len(train_loader)}")

    dataset_signature = sha256(cfg["split_file"])
    if resume:
        resume_state = load_checkpoint(resume, mmap=False)
        if resume_state.get("format_version") != 1 or resume_state.get("task") != "swin":
            raise ValueError("Resume requires a complete checkpoint produced by this training entry")
        if resume_state.get("dataset_signature") != dataset_signature:
            raise ValueError("Dataset filenames or sizes changed since the saved run")

    # 模型
    print(f"\n初始化 Swin-Base ReID (stripes={NUM_STRIPES}, reduce_dim={REDUCE_DIM})...")
    print(f"初始冻结前 {FREEZE_STAGES} 个 stage")
    model = SwinReID(
        num_classes   = num_classes,
        num_stripes   = NUM_STRIPES,
        reduce_dim    = REDUCE_DIM,
        freeze_stages = FREEZE_STAGES,
        dropout       = DROPOUT,
        pretrained    = resume is None,
    ).to(DEVICE)

    criterion = ReIDLoss(
        num_classes = num_classes,
        feat_dim    = REDUCE_DIM,
        margin      = 0.3,
        w_circle    = 0.3,
        w_tri       = 0.5,
        arc_scale   = 64.0,
        arc_margin  = 0.35,
    ).to(DEVICE)

    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.gem.parameters()) +
                   list(model.global_reduce.parameters()) +
                   list(model.stripe_reduces.parameters()) +
                   list(model.global_neck.parameters()) +
                   list(model.stripe_necks.parameters()) +
                   list(model.occlusion_unit.parameters()) +
                   list(criterion.arcface.parameters()))

    optimizer = AdamW([
        {'params': backbone_params, 'lr': BASE_LR * 0.1},
        {'params': head_params,     'lr': BASE_LR},
    ], weight_decay=WEIGHT_DECAY)


    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))


    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')

    start_epoch = 0
    best_rank1  = -1.0
    best_mAP    = 0.0
    no_improve  = 0



    def apply_unfreeze(model, epoch):
        """根据 epoch 动态调整冻结层数"""
        freeze_until = FREEZE_STAGES
        for trigger in sorted(UNFREEZE_SCHEDULE.keys()):
            if epoch >= trigger:
                freeze_until = UNFREEZE_SCHEDULE[trigger]
        model.unfreeze_stages(freeze_until)
        return freeze_until


    def extract_features_tta(model, loader, device):
        """原图 + 水平翻转 TTA，用于评估（不用于推理计时）"""
        model.eval()
        feats, pids, camids = [], [], []
        with torch.no_grad():
            for imgs, pid, camid in tqdm(loader, desc='TTA 提取', leave=False):
                imgs = imgs.to(device)
                f1   = model(imgs)
                f2   = model(torch.flip(imgs, dims=[3]))
                feat = F.normalize(f1 + f2, p=2, dim=1)
                feats.append(feat.cpu())
                pids.extend(pid.numpy())
                camids.extend(camid.numpy())
        return torch.cat(feats, 0).numpy(), np.array(pids), np.array(camids)


    if resume_state is not None:
        model.load_state_dict(resume_state['model_state_dict'], strict=True)
        criterion.load_state_dict(resume_state['criterion_state_dict'], strict=True)
        optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        scheduler.load_state_dict(resume_state['scheduler_state_dict'])
        scaler.load_state_dict(resume_state['scaler_state_dict'])
        start_epoch = int(resume_state['epoch'])
        best_rank1 = float(resume_state['best_rank1'])
        best_mAP = float(resume_state['best_mAP'])
        no_improve = int(resume_state['no_improve'])
        restore_rng(resume_state['rng_state'])
        del resume_state
        if start_epoch >= NUM_EPOCHS:
            raise ValueError('The configured training run is already complete')

    # 训练循环
    print(f"\n开始训练 (共 {NUM_EPOCHS} epochs，当前从 {start_epoch+1} 开始)...")


    for epoch in range(start_epoch, NUM_EPOCHS):

        # 渐进解冻
        cur_freeze = apply_unfreeze(model, epoch)

        model.train()
        total_loss = correct = total = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        for imgs, pids, _ in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            pids = pids.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                outputs = model(imgs)
                loss, loss_arc, loss_circle = criterion(outputs, pids)

            if not torch.isfinite(loss):
                raise ValueError("Nonfinite ReID training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.arcface.parameters()),
                max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            with torch.no_grad():
                feat_f32   = outputs['global_feat'].float()
                weight_f32 = criterion.arcface.weight.float()
                cosine     = F.linear(F.normalize(feat_f32, p=2, dim=1),
                                      F.normalize(weight_f32, p=2, dim=1))
                pred     = cosine.argmax(dim=1)
                correct += pred.eq(pids).sum().item()
                total   += pids.size(0)

            pbar.set_postfix(
                loss=f'{loss.item():.3f}',
                arc=f'{loss_arc.item():.3f}',
                circle=f'{loss_circle.item():.3f}',
                acc=f'{100.*correct/total:.1f}%',
                lr=f'{optimizer.param_groups[1]["lr"]:.1e}',
                frz=cur_freeze,
            )

        scheduler.step()
        avg_loss  = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f"  → avg_loss={avg_loss:.3f}  train_acc={train_acc:.1f}%  "
              f"lr={optimizer.param_groups[1]['lr']:.1e}  freeze={cur_freeze}")

        if (epoch + 1) % EVAL_EVERY == 0:
            print(f"\n[Epoch {epoch+1}] 评估（TTA）...")
            torch.cuda.empty_cache()

            qf, q_pids, q_camids = extract_features_tta(model, query_loader,   DEVICE)
            gf, g_pids, g_camids = extract_features_tta(model, gallery_loader, DEVICE)

            distmat  = compute_dist_matrix(qf, gf)
            cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
            std_r1   = cmc[0] * 100
            std_map  = mAP * 100
            print(f"  [标准+TTA]  Rank-1={std_r1:.2f}%  mAP={std_map:.2f}%")

            if std_r1 > best_rank1:
                best_rank1 = std_r1
                best_mAP   = std_map
                no_improve = 0
                atomic_save({
                    'epoch':                epoch + 1,
                    'model_state_dict':     model.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    'rank1':                best_rank1,
                    'mAP':                  best_mAP,
                    'num_classes':          num_classes,
                    'training_ids': sorted(mapping),
                    'split_signature': dataset_signature,
                    'selection_split': 'validation',
                    'num_stripes':          NUM_STRIPES,
                    'reduce_dim':           REDUCE_DIM,
                }, BEST_CKPT)
                print(f"  保存最佳! Rank-1={best_rank1:.2f}%  mAP={best_mAP:.2f}%")
            else:
                no_improve += 1
                print(f"  历史最佳: Rank-1={best_rank1:.2f}%  mAP={best_mAP:.2f}%"
                      f"  (已 {no_improve} 次未提升)")
                # 连续 3 次未提升 → LR * 0.5
                if no_improve % 3 == 0:
                    for pg in optimizer.param_groups:
                        pg['lr'] = max(pg['lr'] * 0.5, BASE_LR * 0.05)
                    print(f"  LR 衰减 → head_lr={optimizer.param_groups[1]['lr']:.2e}")

            torch.cuda.empty_cache()
            model.train()

        atomic_save({
            'format_version': 1, 'task': 'swin', 'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'best_rank1': float(best_rank1), 'best_mAP': float(best_mAP),
            'no_improve': no_improve, 'num_classes': num_classes,
            'num_stripes': NUM_STRIPES, 'reduce_dim': REDUCE_DIM,
            'dataset_signature': dataset_signature, 'rng_state': capture_rng(),
        }, LATEST_CKPT)
        append_metrics(Path(SAVE_DIR)/'metrics.jsonl', {
            'selection_split': 'validation',
            'epoch': epoch+1, 'loss': float(avg_loss), 'train_accuracy_percent': float(train_acc),
            'best_rank1_percent': float(best_rank1), 'best_mAP_percent': float(best_mAP),
            'head_lr': float(optimizer.param_groups[1]['lr']),
        })

    print(f"Training complete. Best observed Rank-1={best_rank1:.2f}%, mAP={best_mAP:.2f}%")


def main():
    args = parse_training_args('swin')
    cfg = load_training_config(args.config, 'swin')
    if not args.resume and not args.allow_pretrained_download:
        raise SystemExit('Training needs general pretrained weights. Review the download requirements, then add --allow-pretrained-download.')
    prepare_run(cfg, args.device, 'swin', args.resume)
    train(cfg, args.device, args.resume)


if __name__ == '__main__':
    main()
