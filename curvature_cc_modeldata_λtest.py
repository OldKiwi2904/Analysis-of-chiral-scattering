# -*- coding: utf-8 -*-
"""
curvature_cc_ktilde_argzero_grid.py

功能概要
--------
1) 特征检测：κ̃(x) 的“全部局部极大值（含边界）”，不使用 AMPD，不丢弃边界峰；
2) 等长化补点：严格按 PDF 式(4)，在“同一曲率分区（argzero 划分）”内找最近左右邻 a,b，
   用中点 x~=(a+b)/2，z~ 在 corr 上插值；
3) 连接：配置→配置（Hungarian），并支持 GAP_AUG_PENALTY=λ 的“补点惩罚”；
4) 网格搜索 λ：对一组 λ 跑全流程，统计指标并画曲线；
5) 选取流程：基于 L-curve（base_loss vs augmented_frac）的“拐点”，
   在拐点附近选平滑性更优者作为 λ*；
6) 可视化：对每个 λ 都输出 3 组“原本的连接图”（corr-轨迹、κ&κ̃、2D 检查）。

所有输出统一保存在脚本所在目录 ROOT_DIR。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
from scipy.optimize import linear_sum_assignment

# ========== 统一根目录（脚本所在目录） ==========
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 绘图设置 ==========
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 可调参数（匹配/可视化） ==========
TRACK_X_JUMP = 0.15     # 真实-真实配对允许的最大 |Δx|；超过禁配
EPS = 1e-8              # 防除零
BIG_M = 1e6             # 禁配大代价
NUM_DETAIL = 16         # 2D 小图数量

# ========== 可调参数（网格搜索 λ） ==========
# 你可以直接改成等距：np.linspace(0, 0.05, 9)；这里给出一组常用点
LAMBDA_GRID = np.linspace(0, 1, 30)
# 是否为每个 λ 都绘制“原本的连接图”（3 张）；如数据很大可设为 False 仅画 λ*=best
PLOT_CONNECTION_FOR_EACH_LAMBDA = True

# ========== 数据读取与基础函数 ==========
def try_read(basename: str) -> str:
    """
    优先读脚本同根目录；其次当前工作目录；最后 /mnt/data 兜底。
    """
    p1 = os.path.join(ROOT_DIR, basename)
    if os.path.exists(p1):
        return p1
    if os.path.exists(basename):
        return basename
    return os.path.join("/mnt/data", basename)

def read_data_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    angles = np.array([float(x) for x in lines[0].strip().split(', ')])
    wavelengths, measurements = [], []
    for line in lines[1:]:
        data = [float(x) for x in line.strip().split(', ')]
        wavelengths.append(data[0])
        measurements.append(data[1:])
    return angles, np.array(wavelengths), np.array(measurements)

def correlation_2(x, y):
    """标准互相关归一化。"""
    num = np.correlate(x, y, mode='full')
    xx = np.correlate(x, x, mode='full')
    yy = np.correlate(y, y, mode='full')
    denom = np.sqrt(xx * yy)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(denom > 0, num / denom, 0.0)
    return out

def first_derivative(y, x):
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    if y.size < 2 or x.size < 2:
        return np.zeros_like(y)
    return np.gradient(y, x, edge_order=2)

def second_derivative(y, x):
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    if y.size < 3 or x.size < 3:
        return np.zeros_like(y)
    d1 = np.gradient(y, x, edge_order=2)
    d2 = np.gradient(d1, x, edge_order=2)
    return d2

def curvature_signed(fy, x):
    """带符号曲率 κ(x)=f''/(1+f'^2)^(3/2)。"""
    d1 = np.nan_to_num(first_derivative(fy, x), nan=0.0, posinf=0.0, neginf=0.0)
    d2 = np.nan_to_num(second_derivative(fy, x), nan=0.0, posinf=0.0, neginf=0.0)
    den = np.power(1.0 + d1 * d1, 1.5)
    kappa = d2 / np.where(den > 1e-12, den, np.inf)
    return np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)

def kappa_tilde_from_signed(kappa):
    """κ̃(x)：仅保留“峰”（κ<0）且取模。"""
    kappa = np.asarray(kappa)
    return np.where(kappa < 0.0, -kappa, 0.0)

def all_local_maxima_positive(arr):
    """
    取 κ̃(x) 的全部局部极大值（含边界；不使用 AMPD；不丢弃边界峰）。
    仅收集 arr[i] > 0 的点，避免 0 平台被当成极大值。
    判定：arr[i] >= 左邻 且 arr[i] >= 右邻。
    """
    a = np.asarray(arr, dtype=float)
    n = a.size
    if n == 0:
        return np.array([], dtype=int)
    idx = []
    for i in range(n):
        left = a[i-1] if i-1 >= 0 else -np.inf
        right = a[i+1] if i+1 < n else -np.inf
        if a[i] > 0.0 and a[i] >= left and a[i] >= right:
            idx.append(i)
    return np.array(idx, dtype=int)

def zero_crossings(xg, kappa):
    """返回 κ(x) 的零点位置（线性插值）。"""
    xg = np.asarray(xg); kappa = np.asarray(kappa)
    zc = []
    for i in range(len(kappa)-1):
        if kappa[i] == 0.0:
            zc.append(xg[i]); continue
        if kappa[i] * kappa[i+1] < 0:  # 异号，必有过零
            t = abs(kappa[i]) / (abs(kappa[i]) + abs(kappa[i+1]) + 1e-12)
            zc_x = xg[i] * (1 - t) + xg[i+1] * t
            zc.append(zc_x)
    return np.array(sorted(zc), dtype=float)

def segment_of(x, zc, xmin, xmax):
    """给定 x，返回它所在的曲率分区 (L, R)。"""
    if zc.size == 0:
        return (xmin, xmax)
    pos = np.searchsorted(zc, x)
    L = xmin if pos == 0 else zc[pos-1]
    R = xmax if pos == len(zc) else zc[pos]
    return (L, R)

def nearest_neighbors_in_segment(x_target, xs, segL, segR):
    """只在 (segL, segR) 内找 xs 的最近左/右邻。"""
    mask = (xs > segL) & (xs < segR)
    xs_seg = xs[mask]
    if xs_seg.size == 0:
        return None, None
    left_candidates  = xs_seg[xs_seg < x_target]
    right_candidates = xs_seg[xs_seg > x_target]
    a = left_candidates.max()  if left_candidates.size  > 0 else None
    b = right_candidates.min() if right_candidates.size > 0 else None
    return a, b

def midpoint_aug_for_prev_ARGZERO(prev_x, prev_z, curr_x_sorted, curr_z_sorted,
                                  xgrid, curr_corr, curr_kappa):
    """
    按 PDF 式(4)：补点仅在与 prev_x 同一曲率分区内选左右邻；
    若缺一侧邻点，用 prev_x 与另一侧邻点取中点；若分区内无点，退化为“同 x 插值”。
    """
    xmin, xmax = xgrid[0], xgrid[-1]
    zc = zero_crossings(xgrid, curr_kappa)  # κ 的零点（argzero）
    segL, segR = segment_of(prev_x, zc, xmin, xmax)

    a, b = nearest_neighbors_in_segment(prev_x, curr_x_sorted, segL, segR)

    if a is None and b is None:
        xt = float(np.clip(prev_x, xmin, xmax))
    elif a is None:
        xt = float(0.5 * (prev_x + b))
    elif b is None:
        xt = float(0.5 * (a + prev_x))
    else:
        xt = float(0.5 * (a + b))

    # z̃ 在 corr 上插值
    xt_clip = np.clip(xt, xmin, xmax)
    zt = float(np.interp(xt_clip, xgrid, curr_corr))
    return xt, zt

# ========== 预计算：每个角度的 corr/xgrid/特征 ==========
def build_per_angle():
    """
    读取数据并为每个角度预计算：
      xgrid, corr(x), κ(x), κ̃(x), 以及 κ̃-极大值特征 (feats_x, feats_z)。
    """
    LCP_TXT = try_read('Left_or_right_left_Rotation helix_R_LCP_.txt')
    RCP_TXT = try_read('Left_or_right_left_Rotation helix_R_RCP_.txt')

    lcp_angles, lcp_wavelengths, lcp_data = read_data_file(LCP_TXT)
    rcp_angles, rcp_wavelengths, rcp_data = read_data_file(RCP_TXT)

    assert np.allclose(lcp_angles, rcp_angles), "Angle data inconsistent"
    assert np.allclose(lcp_wavelengths, rcp_wavelengths), "Wavelength data inconsistent"

    angles = lcp_angles
    angle_indices = list(range(len(angles)))

    angle_ints = np.sort(angles.astype(int))
    per_angle: Dict[int, Dict[str, np.ndarray]] = {}

    for i, angle_idx in enumerate(angle_indices):
        angle = int(angles[angle_idx])
        s1 = rcp_data[:, angle_idx]; s2 = lcp_data[:, angle_idx]
        corr = correlation_2(s1, s2)
        xgrid = np.array([2 * k / len(corr) - 1 for k in range(len(corr))])

        kappa = curvature_signed(corr, xgrid)
        kappa_tilde = kappa_tilde_from_signed(kappa)

        feat_idx = all_local_maxima_positive(kappa_tilde)  # 全部极大值（含边界）
        feats_x = xgrid[feat_idx] if feat_idx.size > 0 else np.array([], dtype=float)
        feats_z = corr[feat_idx]  if feat_idx.size > 0 else np.array([], dtype=float)

        per_angle[angle] = dict(
            xgrid=xgrid, corr=corr, kappa=kappa, kappa_tilde=kappa_tilde,
            feats_x=feats_x, feats_z=feats_z, feat_idx=feat_idx
        )
    return angle_ints, per_angle

# ========== 代价矩阵与一次评估 ==========
def real_cost(prev_x, prev_z, curr_x, curr_z):
    dx = abs(curr_x - prev_x)
    dz = abs(curr_z - prev_z)
    norm = max(abs(curr_z), abs(prev_z), EPS)
    return dx * (1.0 + dz / norm)

def build_augmented_cost(prev_x, prev_z, curr_x, curr_z, xgrid, curr_corr, curr_kappa, lambda_gap):
    """
    返回用于 Hungarian 的代价矩阵 C 及元信息 meta。
    模式 A: n_prev >= n_curr → (n_prev, n_curr + n_prev)
      - 前 n_curr 列是真实列；
      - 后 n_prev 列为“每个 prev 的专属补点列”（只允许该行），代价 = real_cost(prev, x~, z~) + lambda_gap。
    模式 B: n_curr > n_prev → (n_prev + n_curr, n_curr)
      - 前 n_prev 行是真实行；
      - 后 n_curr 行为“每个 curr 的专属补点行”（只允许该列），代价 = lambda_gap（开新轨迹的 gap 罚）。
    """
    order_curr = np.argsort(curr_x)
    curr_x_sorted = curr_x[order_curr]
    curr_z_sorted = curr_z[order_curr]

    n_prev = len(prev_x)
    n_curr = len(curr_x)

    if n_prev >= n_curr:
        C = np.full((n_prev, n_curr + n_prev), BIG_M, dtype=float)

        # 真实-真实代价（前 n_curr 列）
        for r in range(n_prev):
            for cj in range(n_curr):
                dx = abs(curr_x_sorted[cj] - prev_x[r])
                if dx > TRACK_X_JUMP:
                    C[r, cj] = BIG_M
                else:
                    C[r, cj] = real_cost(prev_x[r], prev_z[r], curr_x_sorted[cj], curr_z_sorted[cj])

        # 专属补点列
        aug_cols = {}
        for r in range(n_prev):
            xt, zt = midpoint_aug_for_prev_ARGZERO(prev_x[r], prev_z[r],
                                                   curr_x_sorted, curr_z_sorted,
                                                   xgrid, curr_corr, curr_kappa)
            base = real_cost(prev_x[r], prev_z[r], xt, zt)
            C[r, n_curr + r] = base + float(lambda_gap)
            aug_cols[n_curr + r] = (r, xt, zt, base)  # 用于统计 base/gap
        meta = dict(mode='prev_ge_curr', order_curr=order_curr, n_curr=n_curr, n_prev=n_prev,
                    aug_cols_meta=aug_cols)
        return C, meta

    else:
        C = np.full((n_prev + n_curr, n_curr), BIG_M, dtype=float)

        # 真实-真实代价（前 n_prev 行）
        for r in range(n_prev):
            for cj in range(n_curr):
                dx = abs(curr_x_sorted[cj] - prev_x[r])
                if dx > TRACK_X_JUMP:
                    C[r, cj] = BIG_M
                else:
                    C[r, cj] = real_cost(prev_x[r], prev_z[r], curr_x_sorted[cj], curr_z_sorted[cj])

        # 专属补点行（仅对角允许）
        for cj in range(n_curr):
            C[n_prev + cj, cj] = float(lambda_gap)
        meta = dict(mode='curr_gt_prev', order_curr=order_curr, n_curr=n_curr, n_prev=n_prev)
        return C, meta

@dataclass
class EvalResult:
    lambda_gap: float
    base_loss: float
    gap_term: float
    total_loss: float
    augmented_pairs: int
    total_pairs: int
    augmented_frac: float
    mean_dx_realreal: float
    mean_abs_second_diff_x: float
    num_tracks: int

def mean_abs_second_diff(xs: List[float]) -> float:
    xs = np.array(xs, dtype=float)
    if xs.size < 3:
        return np.nan
    return float(np.mean(np.abs(xs[2:] - 2*xs[1:-1] + xs[:-2])))

def evaluate_lambda(angle_ints: np.ndarray,
                    per_angle: Dict[int, Dict[str, np.ndarray]],
                    lambda_gap: float,
                    collect_tracks: bool = False) -> Tuple[EvalResult, List[Dict[str, Any]]]:
    """
    在给定 λ 下执行整条追踪：
      - 返回 EvalResult（各类度量）
      - 若 collect_tracks=True，同步返回轨迹列表以便绘图
    """
    tracks: List[Dict[str, Any]] = []

    # 初始化：第一角度的特征即开启轨迹
    first_angle = int(angle_ints[0])
    fx = per_angle[first_angle]['feats_x']
    fz = per_angle[first_angle]['feats_z']
    if fx.size > 0:
        for j in range(len(fx)):
            tracks.append(dict(
                angles=[first_angle],
                x=[float(fx[j])],
                z=[float(fz[j])],
                last_x=float(fx[j]),
                last_z=float(fz[j]),
            ))

    # 统计量
    total_base_loss = 0.0
    total_gap_term  = 0.0
    total_pairs     = 0
    augmented_pairs = 0
    dx_realreal     = []

    # 逐角连接
    for a_idx in range(1, len(angle_ints)):
        prev_angle = int(angle_ints[a_idx - 1])
        curr_angle = int(angle_ints[a_idx])

        prev_x = np.array([t['last_x'] for t in tracks], dtype=float)
        prev_z = np.array([t['last_z'] for t in tracks], dtype=float)

        curr_x = per_angle[curr_angle]['feats_x']
        curr_z = per_angle[curr_angle]['feats_z']

        xg  = per_angle[curr_angle]['xgrid']
        cg  = per_angle[curr_angle]['corr']
        kap = per_angle[curr_angle]['kappa']

        if len(prev_x) == 0 and len(curr_x) > 0:
            # 无 prev：全新轨迹
            for j in range(len(curr_x)):
                tracks.append(dict(
                    angles=[curr_angle], x=[float(curr_x[j])], z=[float(curr_z[j])],
                    last_x=float(curr_x[j]), last_z=float(curr_z[j]),
                ))
            continue

        if len(curr_x) == 0:
            # 无 curr：所有既有轨迹“同 x 续连”
            for t in tracks:
                j = int(np.argmin(np.abs(xg - t['last_x'])))
                t['angles'].append(curr_angle)
                t['x'].append(float(xg[j]))
                t['z'].append(float(cg[j]))
                t['last_x'] = float(xg[j])
                t['last_z'] = float(cg[j])
            continue

        # 构造代价并分配
        C, meta = build_augmented_cost(prev_x, prev_z, curr_x, curr_z, xg, cg, kap, lambda_gap)
        row_ind, col_ind = linear_sum_assignment(C)

        order_curr = meta['order_curr']
        curr_x_sorted = curr_x[order_curr]
        curr_z_sorted = curr_z[order_curr]

        mode = meta['mode']

        if mode == 'prev_ge_curr':
            n_curr = meta['n_curr']; n_prev = meta['n_prev']
            chosen_real_cols = set([c for c in col_ind if c < n_curr])

            for r, c in zip(row_ind, col_ind):
                total_pairs += 1
                if c < n_curr:
                    # 真实-真实
                    x_new = float(curr_x_sorted[c]); z_new = float(curr_z_sorted[c])
                    base = real_cost(prev_x[r], prev_z[r], x_new, z_new)
                    total_base_loss += base
                    dx_realreal.append(abs(x_new - prev_x[r]))

                    tracks[r]['angles'].append(curr_angle)
                    tracks[r]['x'].append(x_new); tracks[r]['z'].append(z_new)
                    tracks[r]['last_x'] = x_new;   tracks[r]['last_z'] = z_new
                else:
                    # 命中“专属补点列”
                    augmented_pairs += 1
                    total_gap_term  += float(lambda_gap)
                    # 统计 base：使用专属补点 (prev, x~, z~)
                    aug_meta = meta.get('aug_cols_meta', {}).get(c, None)
                    if aug_meta is not None:
                        _, xt, zt, base = aug_meta
                        total_base_loss += base
                    # 表征轨迹：同 x 续连
                    j = int(np.argmin(np.abs(xg - prev_x[r])))
                    x_cont = float(xg[j]); z_cont = float(cg[j])
                    tracks[r]['angles'].append(curr_angle)
                    tracks[r]['x'].append(x_cont); tracks[r]['z'].append(z_cont)
                    tracks[r]['last_x'] = x_cont;   tracks[r]['last_z'] = z_cont

            # 未被选择的真实 curr 列 ⇒ 新轨迹
            for c_sorted in range(n_curr):
                if c_sorted not in chosen_real_cols:
                    x_new = float(curr_x_sorted[c_sorted]); z_new = float(curr_z_sorted[c_sorted])
                    tracks.append(dict(angles=[curr_angle], x=[x_new], z=[z_new],
                                       last_x=x_new, last_z=z_new))

        else:  # mode == 'curr_gt_prev'
            n_curr = meta['n_curr']; n_prev = meta['n_prev']

            for r, c in zip(row_ind, col_ind):
                total_pairs += 1
                if r < n_prev:
                    # 真实-真实
                    x_new = float(curr_x_sorted[c]); z_new = float(curr_z_sorted[c])
                    base = real_cost(prev_x[r], prev_z[r], x_new, z_new)
                    total_base_loss += base
                    dx_realreal.append(abs(x_new - prev_x[r]))

                    tracks[r]['angles'].append(curr_angle)
                    tracks[r]['x'].append(x_new); tracks[r]['z'].append(z_new)
                    tracks[r]['last_x'] = x_new;   tracks[r]['last_z'] = z_new
                else:
                    # “专属补点行” ⇒ 新轨迹
                    augmented_pairs += 1
                    total_gap_term  += float(lambda_gap)
                    x_new = float(curr_x_sorted[c]); z_new = float(curr_z_sorted[c])
                    tracks.append(dict(angles=[curr_angle], x=[x_new], z=[z_new],
                                       last_x=x_new, last_z=z_new))

            # 兜底：若有 prev 行未被选中（理论不会发生），同 x 续连
            chosen_rows = set(row_ind.tolist())
            for r in range(n_prev):
                if r not in chosen_rows:
                    j = int(np.argmin(np.abs(xg - prev_x[r])))
                    x_cont = float(xg[j]); z_cont = float(cg[j])
                    tracks[r]['angles'].append(curr_angle)
                    tracks[r]['x'].append(x_cont); tracks[r]['z'].append(z_cont)
                    tracks[r]['last_x'] = x_cont;   tracks[r]['last_z'] = z_cont

    # 平滑性
    smooth_vals = [mean_abs_second_diff(t['x']) for t in tracks]
    mean_smooth = float(np.nanmean(smooth_vals)) if np.isfinite(smooth_vals).any() else np.nan
    aug_frac = augmented_pairs / max(1, total_pairs)
    mean_dx = float(np.mean(dx_realreal)) if len(dx_realreal) > 0 else np.nan

    res = EvalResult(lambda_gap=lambda_gap,
                     base_loss=float(total_base_loss),
                     gap_term=float(total_gap_term),
                     total_loss=float(total_base_loss + total_gap_term),
                     augmented_pairs=int(augmented_pairs),
                     total_pairs=int(total_pairs),
                     augmented_frac=float(aug_frac),
                     mean_dx_realreal=mean_dx,
                     mean_abs_second_diff_x=mean_smooth,
                     num_tracks=len(tracks))

    return (res, tracks if collect_tracks else [])

# ========== 指标与可视化 ==========
def plot_for_lambda(angle_ints, per_angle, tracks, tag: str, outdir=ROOT_DIR):
    """
    为指定 λ 的结果绘制：
      1) 3D corr + 轨迹 + κ̃-极大值
      2) 3D κ 与 κ̃ + κ 上标记
      3) 2D 多角度检查图
    文件名包含 tag（例如 'lambda_0.010'）。
    """
    os.makedirs(outdir, exist_ok=True)
    discr_colors = cm.hsv(np.delete(np.linspace(0, 1, len(angle_ints) + 1), -1))

    # 1) 3D corr
    fig_corr = plt.figure(figsize=(8, 9))
    ax_corr = fig_corr.add_subplot(projection='3d')
    ax_corr.set_box_aspect([2.0, 4.0, 2.0])

    for i, angle in enumerate(angle_ints[::-1]):
        cg = per_angle[int(angle)]['corr']
        xg = per_angle[int(angle)]['xgrid']
        ax_corr.plot(
            xg, np.full(len(xg), int(angle)), cg,
            color=discr_colors[i], linewidth=2.0,
            path_effects=[pe.Stroke(linewidth=3.2, foreground='black'), pe.Normal()]
        )
        fx = per_angle[int(angle)]['feats_x']
        fz = per_angle[int(angle)]['feats_z']
        if fx.size > 0:
            ax_corr.scatter(fx, np.full_like(fx, int(angle)), fz,
                            color='magenta', s=22, depthshade=False, alpha=0.95, marker='o',
                            label='κ̃ maxima on corr' if i == 0 else None)

    cmap = plt.get_cmap('tab20')
    for idx, t in enumerate(tracks):
        color = cmap(idx % 20)
        order = np.argsort(t['angles'])
        ang = np.asarray(t['angles'])[order]
        xs  = np.asarray(t['x'])[order]
        zs  = np.asarray(t['z'])[order]
        ax_corr.plot(xs, ang, zs, linewidth=2.0, color=color, alpha=0.95)
        ax_corr.scatter(xs, ang, zs, s=16, color=color, depthshade=False, alpha=0.95)

    ax_corr.set_xlabel('Normalized spectral lag (x)', labelpad=2)
    ax_corr.set_ylabel('Rotation angle (deg)', labelpad=10)
    ax_corr.set_xlim(-1.0, 1.0); ax_corr.set_ylim(-15, 370)
    ax_corr.set_title(f'Correlation surface & tracks ({tag})')
    ax_corr.view_init(30, -130, 0)
    ax_corr.legend(loc='upper left')
    fig_corr.tight_layout()
    fig_corr.savefig(os.path.join(outdir, f"corr_tracks_{tag}.png"), dpi=160)
    plt.close(fig_corr)

    # 2) 3D κ
    fig_k = plt.figure(figsize=(8, 9))
    ax_k = fig_k.add_subplot(projection='3d')
    ax_k.set_box_aspect([2.0, 4.0, 2.0])
    for i, angle in enumerate(angle_ints[::-1]):
        xg   = per_angle[int(angle)]['xgrid']
        kap  = per_angle[int(angle)]['kappa']
        kapt = per_angle[int(angle)]['kappa_tilde']
        fidx = per_angle[int(angle)]['feat_idx']
        ax_k.plot(xg, np.full(len(xg), int(angle)), kap,
                  color=discr_colors[i], linewidth=2.0,
                  path_effects=[pe.Stroke(linewidth=3.2, foreground='black'), pe.Normal()],
                  label='κ(x)' if i == 0 else None)
        ax_k.plot(xg, np.full(len(xg), int(angle)), kapt,
                  color='magenta', linewidth=1.2, alpha=0.7,
                  label='κ̃(x)' if i == 0 else None)
        if fidx.size > 0:
            ax_k.scatter(xg[fidx], np.full_like(xg[fidx], int(angle)), kap[fidx],
                         color='red', s=20, depthshade=False, alpha=0.95, marker='^',
                         label='κ̃ maxima on κ' if i == 0 else None)

    ax_k.set_xlabel('Normalized spectral lag (x)', labelpad=2)
    ax_k.set_ylabel('Rotation angle (deg)', labelpad=10)
    ax_k.set_xlim(-1.0, 1.0); ax_k.set_ylim(-15, 370)
    ax_k.set_title(f'Curvature κ & κ̃ ({tag})')
    ax_k.view_init(30, -130, 0)
    ax_k.legend(loc='upper left')
    fig_k.tight_layout()
    fig_k.savefig(os.path.join(outdir, f"kappa_{tag}.png"), dpi=160)
    plt.close(fig_k)

    # 3) 2D 小图
    fig2d, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    show_angles = angle_ints[:min(NUM_DETAIL, len(angle_ints))]
    for k, angle in enumerate(show_angles):
        xg = per_angle[int(angle)]['xgrid']
        cg = per_angle[int(angle)]['corr']
        kap= per_angle[int(angle)]['kappa']
        kapt=per_angle[int(angle)]['kappa_tilde']
        fidx = per_angle[int(angle)]['feat_idx']
        ax = axes[k]
        ax.plot(xg, cg, lw=2, label='corr', color='C0')
        ax.plot(xg, kap, lw=1.2, label='κ(x)', color='C1')
        ax.plot(xg, kapt, lw=1.2, label='κ̃(x)', color='C3', alpha=0.7)
        if fidx.size > 0:
            ax.scatter(xg[fidx], cg[fidx], s=22, color='magenta', label='κ̃ maxima on corr')
            ax.scatter(xg[fidx], kap[fidx], s=22, color='red', marker='^', label='κ̃ maxima on κ')
        ax.set_title(f'{int(angle)}°'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig2d.tight_layout()
    fig2d.savefig(os.path.join(outdir, f"check2d_{tag}.png"), dpi=160)
    plt.close(fig2d)

def discrete_curvature(xs, ys):
    """
    计算折线在对数-对数坐标下的“离散曲率”，用于 L-curve 拐点检测。
    输入：xs, ys 为一维数组（建议先正数）。
    返回：与 xs 同长度的数组（端点为 NaN）。
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    xs = np.log(np.maximum(xs, 1e-12))
    ys = np.log(np.maximum(ys, 1e-12))
    kappa = np.full_like(xs, np.nan)
    for i in range(1, len(xs)-1):
        v1 = np.array([xs[i]-xs[i-1], ys[i]-ys[i-1]])
        v2 = np.array([xs[i+1]-xs[i], ys[i+1]-ys[i]])
        den = np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-12
        kappa[i] = abs(np.cross(v1, v2)) / den
    return kappa

def save_records_and_plots(records: List[Dict[str, Any]], outdir=ROOT_DIR):
    """
    保存 CSV 并绘制多张 “λ 对指标” 曲线（统一写到 ROOT_DIR）。
    """
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(records).sort_values("lambda_gap").reset_index(drop=True)
    csv_path = os.path.join(outdir, "lambda_grid_metrics.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[Saved] {csv_path}")

    # ---- L-curve：base_loss vs augmented_frac ----
    plt.figure(figsize=(6,4))
    plt.plot(df['base_loss'], df['augmented_frac'], '-o')
    for x,y,lam in zip(df['base_loss'], df['augmented_frac'], df['lambda_gap']):
        plt.annotate(f"{lam:.3f}", (x,y), textcoords="offset points", xytext=(4,4), fontsize=7)
    plt.xlabel("Base CC loss (sum over pairs)")
    plt.ylabel("Augmented-pair fraction")
    plt.title("L-curve: data-fit vs augmentation usage")
    plt.tight_layout()
    p1 = os.path.join(outdir, "grid_lcurve.png")
    plt.savefig(p1, dpi=160); plt.close()
    print(f"[Saved] {p1}")

    # ---- Smoothness vs λ ----
    plt.figure(figsize=(6,4))
    plt.plot(df['lambda_gap'], df['mean_abs_second_diff_x'], '-o')
    plt.xlabel("lambda_gap")
    plt.ylabel("Mean |second diff of x| (↓ smoother)")
    plt.title("Track smoothness vs lambda")
    plt.tight_layout()
    p2 = os.path.join(outdir, "grid_smoothness_vs_lambda.png")
    plt.savefig(p2, dpi=160); plt.close()
    print(f"[Saved] {p2}")

    # ---- Aug fraction vs λ ----
    plt.figure(figsize=(6,4))
    plt.plot(df['lambda_gap'], df['augmented_frac'], '-o')
    plt.xlabel("lambda_gap")
    plt.ylabel("Augmented-pair fraction (↓ better)")
    plt.title("Augmentation usage vs lambda")
    plt.tight_layout()
    p3 = os.path.join(outdir, "grid_augfrac_vs_lambda.png")
    plt.savefig(p3, dpi=160); plt.close()
    print(f"[Saved] {p3}")

    # ---- Total (base+gap) vs λ ----
    plt.figure(figsize=(6,4))
    plt.plot(df['lambda_gap'], df['total_loss'], '-o')
    plt.xlabel("lambda_gap")
    plt.ylabel("Total loss (base + gap term)")
    plt.title("Total loss vs lambda")
    plt.tight_layout()
    p4 = os.path.join(outdir, "grid_total_vs_lambda.png")
    plt.savefig(p4, dpi=160); plt.close()
    print(f"[Saved] {p4}")

    # ---- 归一化对比（相对 λ=0 的比例） ----
    base_row = df.iloc[(df['lambda_gap']-0).abs().idxmin()]
    eps = 1e-12
    r_fit = df['base_loss'] / max(base_row['base_loss'], eps)
    r_aug = df['augmented_frac'] / max(base_row['augmented_frac'], eps)
    r_smt = df['mean_abs_second_diff_x'] / (base_row['mean_abs_second_diff_x'] if np.isfinite(base_row['mean_abs_second_diff_x']) else 1.0 + eps)

    plt.figure(figsize=(7.5,4.5))
    plt.plot(df['lambda_gap'], r_fit, '-o', label='fit ratio (base_loss/base@0)')
    plt.plot(df['lambda_gap'], r_aug, '-o', label='aug ratio (aug_frac/aug@0)')
    plt.plot(df['lambda_gap'], r_smt, '-o', label='smooth ratio (smooth/smooth@0)')
    plt.axhline(1.0, color='gray', ls='--', lw=1)
    plt.xlabel("lambda_gap"); plt.ylabel("ratio to λ=0 (↓ better except fit)")
    plt.title("Normalized metrics vs lambda (relative to λ=0)")
    plt.legend()
    plt.tight_layout()
    p5 = os.path.join(outdir, "grid_normalized_metrics.png")
    plt.savefig(p5, dpi=160); plt.close()
    print(f"[Saved] {p5}")

    return df

# ========== 选取流程（L-curve 拐点 + 平滑性优先） ==========
def select_lambda_from_grid(df: pd.DataFrame) -> float:
    """
    λ 选择流程（可复现）：
      Step-1: 计算 L-curve（base_loss vs augmented_frac）的离散曲率 κ，取 κ 最大的 λ 作为“拐点候选”；
      Step-2: 在该候选的 ±1 个网格邻域内，选择“平滑性 mean_abs_second_diff_x 最小”的 λ；
              如有平手，再选 base_loss 较小者。
    返回 λ*。
    """
    xs = df['base_loss'].values
    ys = df['augmented_frac'].values
    kapp = discrete_curvature(xs, ys)
    # 若全部 NaN（比如点很少），退回到“最小归一化复合分数”（fit/aug/smooth 等权）
    if not np.isfinite(kapp[1:-1]).any():
        eps = 1e-12
        base0 = df.loc[(df['lambda_gap']-0).abs().idxmin()]
        r_fit = df['base_loss'] / max(base0['base_loss'], eps)
        r_aug = df['augmented_frac'] / max(base0['augmented_frac'], eps)
        r_smt = df['mean_abs_second_diff_x'] / (base0['mean_abs_second_diff_x'] if np.isfinite(base0['mean_abs_second_diff_x']) else 1.0 + eps)
        score = r_fit + r_aug + r_smt
        i_best = int(np.nanargmin(score.values))
        return float(df.loc[i_best, 'lambda_gap'])

    i_corner = int(np.nanargmax(kapp))  # 拐点索引
    # 邻域窗口
    cand_idx = [i for i in range(max(0, i_corner-1), min(len(df), i_corner+2))]
    sub = df.loc[cand_idx].copy()
    # 首选平滑性最小
    sm_min = sub['mean_abs_second_diff_x'].min()
    sub = sub.loc[sub['mean_abs_second_diff_x'] <= sm_min + 1e-12]
    # 再次按 base_loss 最小挑选
    i_best = int(sub['base_loss'].idxmin())
    return float(df.loc[i_best, 'lambda_gap'])

# ========== 主流程 ==========
def main():
    # 1) 预计算每个角度的数据
    angle_ints, per_angle = build_per_angle()
    print(f"[Info] Angles: {len(angle_ints)}; First={angle_ints[0]}, Last={angle_ints[-1]}")
    print(f"[Info] ROOT_DIR (outputs here): {ROOT_DIR}")

    # 2) 网格搜索 λ：评估、记录与可视化
    records = []
    for lam in LAMBDA_GRID:
        res, tracks = evaluate_lambda(angle_ints, per_angle, lambda_gap=float(lam),
                                      collect_tracks=PLOT_CONNECTION_FOR_EACH_LAMBDA)
        records.append(res.__dict__)
        if PLOT_CONNECTION_FOR_EACH_LAMBDA:
            tag = f"lambda_{lam:.3f}"
            plot_for_lambda(angle_ints, per_angle, tracks, tag=tag, outdir=ROOT_DIR)
            print(f"[Plot] {tag} | base={res.base_loss:.3e}, aug_frac={res.augmented_frac:.3f}, smooth={res.mean_abs_second_diff_x:.3e}")

    # 3) 指标曲线与 CSV
    df = save_records_and_plots(records, outdir=ROOT_DIR)

    # 4) 选取流程（L-curve 拐点 + 平滑性最优）
    lam_star = select_lambda_from_grid(df)
    print(f"[Select] λ* = {lam_star:.6f}  (L-curve 拐点邻域 + 平滑性优先)")

    # 5) 若没有为每个 λ 作连接图，这里至少为 λ* 输出一次
    if not PLOT_CONNECTION_FOR_EACH_LAMBDA:
        res, tracks = evaluate_lambda(angle_ints, per_angle, lambda_gap=lam_star, collect_tracks=True)
        tag = f"lambda_{lam_star:.3f}"
        plot_for_lambda(angle_ints, per_angle, tracks, tag=tag, outdir=ROOT_DIR)
        print(f"[Plot] {tag} | base={res.base_loss:.3e}, aug_frac={res.augmented_frac:.3f}, smooth={res.mean_abs_second_diff_x:.3e}")

if __name__ == "__main__":
    main()
