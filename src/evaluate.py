"""余弦距离与按原始身份编号计算的检索指标。"""
import numpy as np


def compute_dist_matrix(qf, gf):
    """余弦距离（值越小越相似）"""
    return 1.0 - np.dot(qf, gf.T)


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    if distmat.ndim != 2 or min(distmat.shape) == 0 or not np.isfinite(distmat).all():
        raise ValueError("Expected a finite nonempty distance matrix")
    num_q   = distmat.shape[0]
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_AP, num_valid = [], [], 0
    for q_idx in range(num_q):
        cmc = matches[q_idx]
        if not np.any(cmc):
            continue
        num_valid += 1
        curve = cmc.cumsum()
        curve[curve > 1] = 1
        all_cmc.append(curve[:max_rank])
        num_rel = cmc.sum()
        tmp     = cmc.cumsum()
        prec    = [t / (i + 1.) for i, t in enumerate(tmp)]
        all_AP.append((np.array(prec) * cmc).sum() / num_rel)

    if not num_valid:
        raise ValueError("No query identity has a gallery match")
    all_cmc = np.array(all_cmc, dtype=np.float32).sum(0) / num_valid
    return all_cmc, float(np.mean(all_AP))
