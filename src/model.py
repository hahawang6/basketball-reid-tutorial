"""Swin ReID：全局与条带特征拼接。推理默认不下载权重。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """Generalized Mean Pooling — 对判别性局部区域赋予更高权重"""
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p_init)
        self.eps = eps

    def forward(self, x):
        # x: [B, N, D]
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)


class BNNeck(nn.Module):
    """训练返回 (原始feat, logits)，推理返回 BN后feat"""
    def __init__(self, in_feat, num_classes, dropout=0.0):
        super().__init__()
        self.bn  = nn.BatchNorm1d(in_feat)
        self.bn.bias.requires_grad_(False)
        self.cls = nn.Linear(in_feat, num_classes, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.cls.weight, std=0.001)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, feat):
        feat_bn = self.bn(feat)
        if self.training:
            return feat, self.cls(self.drop(feat_bn))
        return feat_bn


class OcclusionUnit(nn.Module):
    """轻量遮挡可见性加权：预测每个条带的可见性权重"""
    def __init__(self, in_dim, num_parts, hidden=128):
        super().__init__()
        self.reduce  = nn.Linear(in_dim, hidden)
        self.attn    = nn.Linear(hidden * num_parts, num_parts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, part_feats):
        reduced = [F.relu(self.reduce(f)) for f in part_feats]
        weights = self.sigmoid(self.attn(torch.cat(reduced, dim=1)))
        return [f * weights[:, i:i+1] for i, f in enumerate(part_feats)], weights


class SwinReID(nn.Module):
    """
    Swin-Base ReID 模型 v4
    骨干：swin_base_patch4_window7_224
    全局特征：Swin 输出的 patch tokens 做 GeM pooling
    条带特征：将 2D feature map 按高度切成 num_stripes 份
    遮挡加权：OcclusionUnit 预测每条带可见性

    Swin 输出尺寸：
      - 输入 224x224，输出 feature map 7x7x1024（stage4）
      - forward_features 返回 [B, 7, 7, 1024]，随后展平为49个token
    """
    def __init__(self, num_classes, num_stripes=6, reduce_dim=512,
                 freeze_stages=1, dropout=0.1, pretrained=False):
        super().__init__()
        self.num_stripes = num_stripes
        self.reduce_dim  = reduce_dim

        # 与完整项目权重一起使用；推理默认不额外下载骨干权重。
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0,
            global_pool='',          # 不做全局池化，我们自己处理
        )
        self.feat_dim = self.backbone.num_features  # 1024

        # 冻结前 freeze_stages 个 stage
        self._freeze_stages(freeze_stages)

        # GeM pooling
        self.gem = GeM(p_init=3.0)

        # 全局降维
        self.global_reduce = nn.Sequential(
            nn.Linear(self.feat_dim, reduce_dim),
            nn.BatchNorm1d(reduce_dim),
            nn.ReLU(inplace=True),
        )

        # 条带降维
        self.stripe_reduces = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feat_dim, reduce_dim),
                nn.BatchNorm1d(reduce_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(num_stripes)
        ])

        # BNNeck 分类头
        self.global_neck  = BNNeck(reduce_dim, num_classes, dropout)
        self.stripe_necks = nn.ModuleList([
            BNNeck(reduce_dim, num_classes, dropout)
            for _ in range(num_stripes)
        ])

        # 遮挡加权
        self.occlusion_unit = OcclusionUnit(self.feat_dim, num_stripes,
                                            hidden=max(64, 128 * 6 // num_stripes))

    def _freeze_stages(self, num_stages):
        """冻结 Swin 的前 num_stages 个 stage"""
        # stage 0: patch_embed
        # stage 1-4: layers[0..3]
        if num_stages >= 1:
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad_(False)
        for i in range(min(num_stages, len(self.backbone.layers))):
            for p in self.backbone.layers[i].parameters():
                p.requires_grad_(False)

    def unfreeze_stages(self, freeze_until):
        """动态解冻，freeze_until=0 表示全部解冻"""
        # patch_embed
        req = (freeze_until >= 1)
        for p in self.backbone.patch_embed.parameters():
            p.requires_grad_(not req)
        # layers
        for i, layer in enumerate(self.backbone.layers):
            req = (i < freeze_until - 1)  # layers[0] 对应 stage1
            for p in layer.parameters():
                p.requires_grad_(not req)

    def get_stripe_feats(self, tokens_2d):
        """
        tokens_2d: [B, H, W, C]
        沿高度方向切成 num_stripes 份，每份做空间均值
        """
        B, H, W, C = tokens_2d.shape
        stripe_h = H // self.num_stripes
        parts = []
        for i in range(self.num_stripes):
            s = i * stripe_h
            e = s + stripe_h if i < self.num_stripes - 1 else H
            part = tokens_2d[:, s:e, :, :].reshape(B, -1, C).mean(dim=1)
            parts.append(part)
        return parts

    def forward(self, x):
        # Swin forward_features 返回 [B, H, W, C]（当前实现要求该布局）
        feat_map = self.backbone.forward_features(x)   # [B, 7, 7, 1024]

        if feat_map.ndim != 4 or feat_map.shape[-1] != self.feat_dim:
            raise ValueError("Unexpected timm feature layout; expected BHWC")
        B, H, W, C = feat_map.shape
        tokens = feat_map.reshape(B, H * W, C)         # [B, 49, 1024]

        # 全局特征：GeM over all patch tokens
        global_raw = self.gem(tokens)                  # [B, 1024]

        # 条带特征 + 遮挡加权
        stripe_feats_raw = self.get_stripe_feats(feat_map)
        weighted_stripes, _ = self.occlusion_unit(stripe_feats_raw)

        # 降维
        g_red = self.global_reduce(global_raw)
        s_red = [self.stripe_reduces[i](weighted_stripes[i])
                 for i in range(self.num_stripes)]

        if self.training:
            g_feat, g_cls = self.global_neck(g_red)
            s_feats, s_cls = [], []
            for feat, neck in zip(s_red, self.stripe_necks):
                f, c = neck(feat)
                s_feats.append(f)
                s_cls.append(c)
            return {
                'global_feat':  g_feat,
                'global_cls':   g_cls,
                'stripe_feats': s_feats,
                'stripe_cls':   s_cls,
            }
        else:
            g_bn = self.global_neck(g_red)
            s_bn = [neck(f) for f, neck in zip(s_red, self.stripe_necks)]
            feat_cat = torch.cat([g_bn] + s_bn, dim=1)
            return F.normalize(feat_cat, p=2, dim=1)
