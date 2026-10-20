import os, random
from collections import defaultdict
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T


def read_image(path):
    return Image.open(path).convert('RGB')


def parse_data_dir(data_dir, relabel=True):
    paths = []
    pids  = set()
    for name in sorted(os.listdir(data_dir)):
        if not name.lower().endswith('.jpg'):
            continue
        parts = name.split('_')
        if len(parts) < 2:
            continue
        try:
            pid   = int(parts[0])
            camid = int(parts[1][1]) if len(parts[1]) > 1 else 0
        except:
            continue
        paths.append((os.path.join(data_dir, name), pid, camid))
        pids.add(pid)
    pid2label = {p: l for l, p in enumerate(sorted(pids))}
    dataset   = [(path, pid2label[pid] if relabel else pid, camid) for path, pid, camid in paths]
    return dataset, len(pids)


# 自定义增强
class MotionBlur:
    def __init__(self, p=0.3):
        self.p = p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5)))


class RandomPatch:
    def __init__(self, p=0.5, max_patches=2):
        self.p = p
        self.n = max_patches
    def __call__(self, img):
        if random.random() > self.p:
            return img
        img = img.copy()
        W, H = img.size
        for _ in range(random.randint(1, self.n)):
            w = random.randint(W // 8, W // 3)
            h = random.randint(H // 8, H // 3)
            x = random.randint(0, W - w)
            y = random.randint(0, H - h)
            col = tuple(random.randint(0, 255) for _ in range(3))
            img.paste(Image.new('RGB', (w, h), col), (x, y))
        return img


class NumberRegionErasing:
    """胸口/背部号码区域专项擦除（Tensor 操作）"""
    def __init__(self, p=0.6, chest_ratio=(0.25, 0.65)):
        self.p    = p
        self.y_lo = chest_ratio[0]
        self.y_hi = chest_ratio[1]
    def __call__(self, img):
        if random.random() > self.p:
            return img
        _, H, W = img.shape
        y1 = int(H * self.y_lo)
        y2 = int(H * self.y_hi)
        h  = random.randint((y2 - y1) // 4, (y2 - y1) // 2)
        w  = random.randint(W // 4, W * 2 // 3)
        y  = random.randint(y1, max(y1, y2 - h))
        x  = random.randint(0, max(0, W - w))
        img = img.clone()
        img[:, y:y+h, x:x+w] = 0
        return img


# Dataset & Sampler
class BallShowDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data      = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        path, pid, camid = self.data[idx]
        img = read_image(path)
        if self.transform:
            img = self.transform(img)
        return img, pid, camid


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances=4):
        if num_instances < 2 or batch_size % num_instances or batch_size // num_instances < 2:
            raise ValueError("batch_size must contain at least two identities and two images per identity")
        self.batch_size         = batch_size
        self.num_instances      = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.index_dic = defaultdict(list)
        for idx, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(idx)
        self.pids = list(self.index_dic.keys())
        if len(self.pids) < self.num_pids_per_batch:
            raise ValueError("Not enough identities for one training batch")

    def __iter__(self):
        pid_idxs = {}
        for pid in self.pids:
            idxs = self.index_dic[pid][:]
            while len(idxs) < self.num_instances:
                idxs += self.index_dic[pid][:]
            random.shuffle(idxs)
            pid_idxs[pid] = idxs
        pids_shuffled = self.pids[:]
        random.shuffle(pids_shuffled)
        final = []
        for pid in pids_shuffled:
            final.extend(pid_idxs[pid][:self.num_instances])
        n = (len(final) // self.batch_size) * self.batch_size
        return iter(final[:n])

    def __len__(self):
        return (len(self.pids) * self.num_instances // self.batch_size) * self.batch_size


# Transforms
def build_transforms(is_train=True, height=224, width=224):
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        return T.Compose([
            T.Resize((height + 16, width + 16), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            T.RandomGrayscale(p=0.1),
            MotionBlur(p=0.3),
            RandomPatch(p=0.5, max_patches=2),
            T.ToTensor(),
            norm,
            NumberRegionErasing(p=0.6),
            T.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=0),
        ])
    else:
        return T.Compose([
            T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            norm,
        ])