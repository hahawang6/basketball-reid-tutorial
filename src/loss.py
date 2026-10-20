import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, num_classes, scale=64.0, margin=0.35):
        super().__init__()
        self.scale  = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

    def forward(self, features, labels):
        cosine = F.linear(F.normalize(features, p=2, dim=1),
                          F.normalize(self.weight, p=2, dim=1))
        sine   = torch.sqrt((1.0 - cosine.pow(2)).clamp(1e-9, 1.0))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
        return F.cross_entropy(output, labels)


class CircleLoss(nn.Module):
    """Circle Loss，限制 logit 范围以控制数值幅度"""
    def __init__(self, m=0.25, gamma=80):
        super().__init__()
        self.m         = m
        self.gamma     = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, feats, labels):
        feats    = F.normalize(feats, p=2, dim=1)
        sim      = torch.matmul(feats, feats.T)
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1))
        mask_pos.fill_diagonal_(False)
        mask_neg = ~mask_pos
        mask_neg.fill_diagonal_(False)

        if not mask_pos.any() or not mask_neg.any():
            return sim.sum() * 0.0

        ap = torch.clamp_min(-sim.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min( sim.detach()     + self.m, min=0.)

        logit_p = (-ap * (sim - (1 - self.m)) * self.gamma).clamp(-100, 100)
        logit_n = ( an * (sim - self.m)        * self.gamma).clamp(-100, 100)

        loss = self.soft_plus(
            torch.logsumexp(logit_n * mask_neg.float() + (~mask_neg).float() * (-1e9), dim=1) +
            torch.logsumexp(logit_p * mask_pos.float() + (~mask_pos).float() * (-1e9), dim=1)
        ).mean()
        return loss


class TripletLoss(nn.Module):
    """Batch Hard Triplet Loss"""
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        dist     = torch.cdist(inputs, inputs, p=2)
        mask_pos = targets.unsqueeze(0).eq(targets.unsqueeze(1))
        dist_ap  = (dist * mask_pos.float()).max(dim=1)[0]
        dist_an  = dist.masked_fill(mask_pos, float('inf')).min(dim=1)[0]
        return F.relu(dist_ap - dist_an + self.margin).mean()


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon     = epsilon

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        smooth    = self.epsilon / self.num_classes
        pos_val   = 1.0 - self.epsilon + smooth
        t = torch.full_like(log_probs, smooth)
        t.scatter_(1, targets.unsqueeze(1), pos_val)
        return (-t * log_probs).sum(dim=1).mean()


class ReIDLoss(nn.Module):
    """
    组合全局分类、条带分类和特征距离损失：
        L = ArcFace
          + 0.4 * CE_stripe（每条带的分类损失均值）
          + w_circle * Circle
          + w_tri * Triplet（全局特征）
          + 0.3 * Triplet（条带特征拼接）
    """
    def __init__(self, num_classes, feat_dim=512, margin=0.3,
                 epsilon=0.1, w_circle=0.3, w_tri=0.5,
                 arc_scale=64.0, arc_margin=0.35):
        super().__init__()
        self.arcface  = ArcFaceLoss(feat_dim, num_classes, arc_scale, arc_margin)
        self.ce       = CrossEntropyLabelSmooth(num_classes, epsilon)
        self.circle   = CircleLoss(m=0.25, gamma=80)
        self.triplet  = TripletLoss(margin)
        self.w_circle = w_circle
        self.w_tri    = w_tri

    def forward(self, outputs, targets):
        global_feat     = outputs['global_feat']
        stripe_cls_list = outputs['stripe_cls']

        # 全局和条带损失
        loss_arc    = self.arcface(global_feat, targets)

        loss_stripe = sum(self.ce(sc, targets) for sc in stripe_cls_list)
        loss_stripe = loss_stripe / max(len(stripe_cls_list), 1)

        loss_circle = self.circle(global_feat, targets)

        loss_tri    = self.triplet(F.normalize(global_feat, p=2, dim=1), targets)

        total = (loss_arc
                 + 0.4  * loss_stripe
                 + self.w_circle * loss_circle
                 + self.w_tri    * loss_tri)

        # 条带特征拼接 Triplet
        stripe_feats = outputs.get('stripe_feats', [])
        if stripe_feats:
            stripe_cat      = torch.cat(stripe_feats, dim=1)   # [B, num_stripes * reduce_dim]
            loss_stripe_tri = self.triplet(
                F.normalize(stripe_cat, p=2, dim=1), targets)
            total = total + 0.3 * loss_stripe_tri

        return total, loss_arc, loss_circle
