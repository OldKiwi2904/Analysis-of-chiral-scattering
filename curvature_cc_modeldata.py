# -*- coding: utf-8 -*-
"""
curvature_cc_ktilde_maxima_argzero_aug.py

核心特性（对齐 PDF 式(4)(5) + 你的全部要求）：
1) 特征检测：从 κ̃(x) 的“全部局部极大值（含边界）”取点；不使用 AMPD。
2) 标注：在 corr(x) 与 κ(x) 的 3D/2D 图上，全部标出这些 κ̃-极大值。
3) 等长化（重点改动）：
   - 当上一角度点数 n_prev >= 当前角度点数 n_curr 时：
     为“每个上一角度点”在**当前角度**按式(4)在“相同曲率分区（由 κ 的零点 argzero 划分）”
     内找最近左右邻 a_k、b_k，取中点 x~ = (a_k + b_k)/2，z~ 用当前 corr(x~) 插值；
     形成“专属补点列”（仅该行可选），与真实列一起进入 Hungarian。
     若匹到补点列 ⇒ 该轨迹在本角度“同 x 续连”。
   - 当 n_curr > n_prev 时（新生特征较多）：
     继续采用“专属补点行”（仅该列可选）的结构；匹到补点行 ⇒ 该点开新轨迹。
     （如需完全对称地也用 argzero 生成上一角度的 x~，可再扩展，但对“新生轨迹”语义非必需。）
4) 匹配代价：|Δx| * (1 + |Δz| / max(|z_prev|,|z_curr|,eps))；|Δx|>TRACK_X_JUMP 禁配。
5) 兜底：未配到的老轨迹“同 x 续连”；未被认领的当前真实点“开新轨迹”。

数据文件（ left-rotation 数据）：
  Left_or_right_left_Rotation helix_R_LCP_.txt
  Left_or_right_left_Rotation helix_R_RCP_.txt
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
from scipy.optimize import linear_sum_assignment

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# --------------------------- 可调参数 ---------------------------
TRACK_X_JUMP = 0.150      # 真实-真实配对允许的最大 |Δx|；超过禁配
EPS = 1e-8               # 防除零
GAP_AUG_PENALTY = 0.330   # 选择“补点列/行”的小额惩罚（越大越偏向真实-真实）
NUM_DETAIL = 16          # 小图数量
BIG_M = 1e6              # 禁配大代价

# --------------------------- 基础函数 ---------------------------
def adjust_lightness(color, amount):
    import matplotlib.colors as mc, colorsys
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(h, max(0, min(1, amount * l)), s)

def correlation_2(x, y):
    num = np.correlate(x, y, mode='full')
    xx = np.correlate(x, x, mode='full')
    yy = np.correlate(y, y, mode='full')
    denom = np.sqrt(xx * yy)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(denom > 0, num / denom, 0.0)
    return out

def read_data_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    angles = np.array([float(x) for x in lines[0].strip().split(', ')])
    wavelengths, measurements = [], []
    for line in lines[1:]:
        data = [float(x) for x in line.strip().split(', ')]
        wavelengths.append(data[0])
        measurements.append(data[1:])
    return angles, np.array(wavelengths), np.array(measurements)

# -------- 导数/曲率 --------
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
    d1 = np.nan_to_num(first_derivative(fy, x), nan=0.0, posinf=0.0, neginf=0.0)
    d2 = np.nan_to_num(second_derivative(fy, x), nan=0.0, posinf=0.0, neginf=0.0)
    den = np.power(1.0 + d1 * d1, 1.5)
    kappa = d2 / np.where(den > 1e-12, den, np.inf)
    return np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)

def kappa_tilde_from_signed(kappa):
    # 保留峰（κ<0），取模；其他置0
    kappa = np.asarray(kappa)
    return np.where(kappa < 0.0, -kappa, 0.0)

def all_local_maxima_positive(arr):
    """
    取 κ̃(x) 的全部局部极大值（包含边界；不使用 AMPD；不丢弃边界峰）。
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

# -------- 曲率零点与分区（argzero 实现） --------
def zero_crossings(xg, kappa):
    """返回 κ(x) 的零点位置（线性插值）。"""
    xg = np.asarray(xg); kappa = np.asarray(kappa)
    zc = []
    for i in range(len(kappa)-1):
        if kappa[i] == 0.0:
            zc.append(xg[i]); continue
        if kappa[i] * kappa[i+1] < 0:  # 异号，必有过零
            # 线性内插
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
    严格按 PDF 式(4)：补点仅在与 prev_x 同一曲率分区内选左右邻；
    若缺一侧邻点，退化为与另一侧/prev_x 取中点；若分区内无点，退化为“同 x 插值”。
    """
    xmin, xmax = xgrid[0], xgrid[-1]
    zc = zero_crossings(xgrid, curr_kappa)  # κ 的零点（argzero）
    segL, segR = segment_of(prev_x, zc, xmin, xmax)

    a, b = nearest_neighbors_in_segment(prev_x, curr_x_sorted, segL, segR)

    if a is None and b is None:
        # 分区内完全没有当前特征 → 退化为“同 x 插值”
        xt = float(np.clip(prev_x, xmin, xmax))
    elif a is None:
        # 只有右邻 → 与 prev_x 取中点
        xt = float(0.5 * (prev_x + b))
    elif b is None:
        # 只有左邻 → 与 prev_x 取中点
        xt = float(0.5 * (a + prev_x))
    else:
        xt = float(0.5 * (a + b))

    # z̃ 在 corr 上插值
    xt_clip = np.clip(xt, xmin, xmax)
    zt = float(np.interp(xt_clip, xgrid, curr_corr))
    return xt, zt

# -------- 读取数据 --------
def try_read(basename):
    local = basename
    alt = os.path.join('/mnt/data', basename)
    return local if os.path.exists(local) else alt

LCP_TXT = try_read('Left_or_right_left_Rotation helix_R_LCP_.txt')
RCP_TXT = try_read('Left_or_right_left_Rotation helix_R_RCP_.txt')

lcp_angles, lcp_wavelengths, lcp_data = read_data_file(LCP_TXT)
rcp_angles, rcp_wavelengths, rcp_data = read_data_file(RCP_TXT)

assert np.allclose(lcp_angles, rcp_angles), "Angle data inconsistent"
assert np.allclose(lcp_wavelengths, rcp_wavelengths), "Wavelength data inconsistent"

angles = lcp_angles
wavelengths = lcp_wavelengths
angle_indices = list(range(len(angles)))
discr_colors = cm.hsv(np.delete(np.linspace(0, 1, len(angle_indices) + 1), -1))

# --------------------------- 预计算每个角度的 corr/xgrid/特征 ---------------------------
angle_ints = np.sort(angles.astype(int))
per_angle = {}  # angle(int) -> dict(xgrid, corr, kappa, kappa_tilde, feats_x, feats_z, feats_k, feat_idx)

for i, angle_idx in enumerate(angle_indices):
    angle = int(angles[angle_idx])
    s1 = rcp_data[:, angle_idx]; s2 = lcp_data[:, angle_idx]
    corr = correlation_2(s1, s2)
    xgrid = np.array([2 * k / len(corr) - 1 for k in range(len(corr))])

    kappa = curvature_signed(corr, xgrid)
    kappa_tilde = kappa_tilde_from_signed(kappa)

    feat_idx = all_local_maxima_positive(kappa_tilde)  # 关键：全部极大值，含边界
    feats_x = xgrid[feat_idx] if feat_idx.size > 0 else np.array([], dtype=float)
    feats_z = corr[feat_idx]  if feat_idx.size > 0 else np.array([], dtype=float)
    feats_k = kappa[feat_idx] if feat_idx.size > 0 else np.array([], dtype=float)

    per_angle[angle] = dict(
        xgrid=xgrid, corr=corr, kappa=kappa, kappa_tilde=kappa_tilde,
        feats_x=feats_x, feats_z=feats_z, feats_k=feats_k, feat_idx=feat_idx
    )

# --------------------------- CC 代价与“补点”等长化 ---------------------------
def real_cost(prev_x, prev_z, curr_x, curr_z):
    dx = abs(curr_x - prev_x)
    dz = abs(curr_z - prev_z)
    norm = max(abs(curr_z), abs(prev_z), EPS)
    return dx * (1.0 + dz / norm)

def build_augmented_cost(prev_x, prev_z, curr_x, curr_z, xgrid, curr_corr, curr_kappa):
    """
    返回用于 Hungarian 的代价矩阵和模式元信息。
    模式 A: n_prev >= n_curr → (n_prev, n_curr + n_prev)
            前 n_curr 列是真实列；后 n_prev 列为“每个 prev 的专属补点列”（只允许该行）。
    模式 B: n_curr >  n_prev → (n_prev + n_curr, n_curr)
            前 n_prev 行是真实行；后 n_curr 行为“每个 curr 的专属补点行”（只允许该列）。
    """
    order_curr = np.argsort(curr_x)
    curr_x_sorted = curr_x[order_curr]
    curr_z_sorted = curr_z[order_curr]

    n_prev = len(prev_x)
    n_curr = len(curr_x)
    if n_prev == 0 and n_curr == 0:
        return np.zeros((0, 0)), dict(mode='empty', order_curr=order_curr)

    if n_prev >= n_curr:
        # ----- 模式 A：列多加“专属补点列”（每个 prev 一列，按 argzero 生成 x~, z~） -----
        C = np.full((n_prev, n_curr + n_prev), BIG_M, dtype=float)

        # 真实-真实代价（前 n_curr 列）
        for r in range(n_prev):
            for cj in range(n_curr):
                dx = abs(curr_x_sorted[cj] - prev_x[r])
                if dx > TRACK_X_JUMP:
                    C[r, cj] = BIG_M
                else:
                    C[r, cj] = real_cost(prev_x[r], prev_z[r], curr_x_sorted[cj], curr_z_sorted[cj])

        # 专属补点列（第 n_curr ~ n_curr+n_prev-1 列），只开放对角元素
        for r in range(n_prev):
            xt, zt = midpoint_aug_for_prev_ARGZERO(prev_x[r], prev_z[r],
                                                   curr_x_sorted, curr_z_sorted,
                                                   xgrid, curr_corr, curr_kappa)
            aug_cost = real_cost(prev_x[r], prev_z[r], xt, zt) + GAP_AUG_PENALTY
            C[r, n_curr + r] = aug_cost  # 仅该行开放，其它行保持 BIG_M

        meta = dict(mode='prev_ge_curr', order_curr=order_curr, n_curr=n_curr, n_prev=n_prev)
        return C, meta

    else:
        # ----- 模式 B：行为多加“专属补点行”（每个 curr 一行；用于新轨迹） -----
        C = np.full((n_prev + n_curr, n_curr), BIG_M, dtype=float)

        # 真实-真实代价（前 n_prev 行）
        for r in range(n_prev):
            for cj in range(n_curr):
                dx = abs(curr_x_sorted[cj] - prev_x[r])
                if dx > TRACK_X_JUMP:
                    C[r, cj] = BIG_M
                else:
                    C[r, cj] = real_cost(prev_x[r], prev_z[r], curr_x_sorted[cj], curr_z_sorted[cj])

        # 专属补点行（第 n_prev ~ n_prev+n_curr-1 行），只开放对角元素
        for cj in range(n_curr):
            C[n_prev + cj, cj] = GAP_AUG_PENALTY

        meta = dict(mode='curr_gt_prev', order_curr=order_curr, n_curr=n_curr, n_prev=n_prev)
        return C, meta

# --------------------------- 轨迹连接 ---------------------------
tracks = []  # 每条轨迹：dict(angles, x, z, forced_flags, last_x, last_z)

# 用第一角度初始化轨迹
first_angle = int(angle_ints[0])
fx = per_angle[first_angle]['feats_x']
fz = per_angle[first_angle]['feats_z']
if fx.size > 0:
    for j in range(len(fx)):
        tracks.append(dict(
            angles=[first_angle],
            x=[float(fx[j])],
            z=[float(fz[j])],
            forced=[False],
            last_x=float(fx[j]),
            last_z=float(fz[j]),
        ))

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

    # 情况1：无 prev 或无 curr
    if len(prev_x) == 0:
        for j in range(len(curr_x)):
            tracks.append(dict(
                angles=[curr_angle],
                x=[float(curr_x[j])],
                z=[float(curr_z[j])],
                forced=[False],
                last_x=float(curr_x[j]),
                last_z=float(curr_z[j]),
            ))
        continue

    if len(curr_x) == 0:
        for t in tracks:
            j = int(np.argmin(np.abs(xg - t['last_x'])))
            t['angles'].append(curr_angle)
            t['x'].append(float(xg[j]))
            t['z'].append(float(cg[j]))
            t['forced'].append(True)
            t['last_x'] = float(xg[j])
            t['last_z'] = float(cg[j])
        continue

    # 常规：构造“等长化 + 专属补点（argzero 分区）”的代价矩阵并求解
    C, meta = build_augmented_cost(prev_x, prev_z, curr_x, curr_z, xg, cg, kap)
    row_ind, col_ind = linear_sum_assignment(C)

    order_curr = meta['order_curr']
    curr_x_sorted = curr_x[order_curr]
    curr_z_sorted = curr_z[order_curr]

    mode = meta['mode']
    if mode == 'prev_ge_curr':
        n_curr = meta['n_curr']; n_prev = meta['n_prev']

        # 被选中的真实 curr 列（0..n_curr-1）
        chosen_real_cols = set([c for c in col_ind if c < n_curr])

        # 逐“prev 行”处理（每一行对应一条现有轨迹）
        for r, c in zip(row_ind, col_ind):
            if r >= len(tracks):
                continue
            if c < n_curr:
                # 真实-真实匹配
                j_sorted = c
                tracks[r]['angles'].append(curr_angle)
                tracks[r]['x'].append(float(curr_x_sorted[j_sorted]))
                tracks[r]['z'].append(float(curr_z_sorted[j_sorted]))
                tracks[r]['forced'].append(False)
                tracks[r]['last_x'] = float(curr_x_sorted[j_sorted])
                tracks[r]['last_z'] = float(curr_z_sorted[j_sorted])
            else:
                # 命中“专属补点列”：同 x 续连
                j = int(np.argmin(np.abs(xg - prev_x[r])))
                tracks[r]['angles'].append(curr_angle)
                tracks[r]['x'].append(float(xg[j]))
                tracks[r]['z'].append(float(cg[j]))
                tracks[r]['forced'].append(True)
                tracks[r]['last_x'] = float(xg[j])
                tracks[r]['last_z'] = float(cg[j])

        # 未被选择的真实 curr 列 ⇒ 新轨迹
        for j_sorted in range(n_curr):
            if j_sorted not in chosen_real_cols:
                tracks.append(dict(
                    angles=[curr_angle],
                    x=[float(curr_x_sorted[j_sorted])],
                    z=[float(curr_z_sorted[j_sorted])],
                    forced=[False],
                    last_x=float(curr_x_sorted[j_sorted]),
                    last_z=float(curr_z_sorted[j_sorted]),
                ))

    elif mode == 'curr_gt_prev':
        n_curr = meta['n_curr']; n_prev = meta['n_prev']

        # 专属补点行的索引集合：{ n_prev + j_sorted }
        claimed_real_cols = set()

        for r, c in zip(row_ind, col_ind):
            if c >= n_curr:
                continue  # 不会发生
            j_sorted = c
            if r < n_prev:
                # 真实-真实匹配：更新已有轨迹 r
                claimed_real_cols.add(j_sorted)
                tracks[r]['angles'].append(curr_angle)
                tracks[r]['x'].append(float(curr_x_sorted[j_sorted]))
                tracks[r]['z'].append(float(curr_z_sorted[j_sorted]))
                tracks[r]['forced'].append(False)
                tracks[r]['last_x'] = float(curr_x_sorted[j_sorted])
                tracks[r]['last_z'] = float(curr_z_sorted[j_sorted])
            else:
                # 命中“专属补点行” ⇒ 新轨迹
                tracks.append(dict(
                    angles=[curr_angle],
                    x=[float(curr_x_sorted[j_sorted])],
                    z=[float(curr_z_sorted[j_sorted])],
                    forced=[False],
                    last_x=float(curr_x_sorted[j_sorted]),
                    last_z=float(curr_z_sorted[j_sorted]),
                ))
                claimed_real_cols.add(j_sorted)

        # 额外兜底：如有 prev 行未选中，做同 x 续连
        chosen_rows = set(row_ind.tolist())
        for r in range(n_prev):
            if r not in chosen_rows:
                j = int(np.argmin(np.abs(xg - prev_x[r])))
                tracks[r]['angles'].append(curr_angle)
                tracks[r]['x'].append(float(xg[j]))
                tracks[r]['z'].append(float(cg[j]))
                tracks[r]['forced'].append(True)
                tracks[r]['last_x'] = float(xg[j])
                tracks[r]['last_z'] = float(cg[j])

# --------------------------- 可视化 ---------------------------

# 1) 3D corr：曲面 + κ̃-极大值标记 + 轨迹
fig_corr = plt.figure(figsize=(8, 9))
ax_corr = fig_corr.add_subplot(projection='3d')
ax_corr.set_box_aspect([2.0, 4.0, 2.0])

zmins, zmaxs = [], []
for angle in angle_ints:
    cg = per_angle[int(angle)]['corr']
    zmins.append(np.nanmin(cg)); zmaxs.append(np.nanmax(cg))
bottom_z_lim = float(np.nanmin(zmins)); top_z_lim = float(np.nanmax(zmaxs))

for i, angle in enumerate(angle_ints[::-1]):
    cg = per_angle[int(angle)]['corr']
    xg = per_angle[int(angle)]['xgrid']
    ax_corr.plot(
        xg, np.full(len(xg), int(angle)), cg,
        color=adjust_lightness(discr_colors[i], 1.0),
        linewidth=2.2,
        path_effects=[pe.Stroke(linewidth=3.6, foreground='black'), pe.Normal()]
    )
    fx = per_angle[int(angle)]['feats_x']
    fz = per_angle[int(angle)]['feats_z']
    if fx.size > 0:
        ax_corr.scatter(fx, np.full_like(fx, int(angle)), fz,
                        color='magenta', s=22, depthshade=False, alpha=0.95, marker='o',
                        label='κ~ maxima on corr' if i == 0 else None)

ax_corr.set_xlabel('Normalized spectral lag (x)', labelpad=2)
ax_corr.set_ylabel('Rotation angle (deg)', labelpad=10)
ax_corr.set_xlim(-1.0, 1.0); ax_corr.set_ylim(-15, 370)
ax_corr.set_title('Polarization correlation (Left rotation)')
ax_corr.zaxis.set_pane_color((1, 1, 1, 0))
ax_corr.yaxis.set_pane_color((1, 1, 1, 0))
ax_corr.xaxis.set_pane_color((1, 1, 1, 0))
ax_corr.view_init(30, -130, 0)
ax_corr.legend(loc='upper left')

# 叠加轨迹
cmap = plt.get_cmap('tab20')
for idx, t in enumerate(tracks):
    color = cmap(idx % 20)
    order = np.argsort(t['angles'])
    ang = np.asarray(t['angles'])[order]
    xs  = np.asarray(t['x'])[order]
    zs  = np.asarray(t['z'])[order]
    ax_corr.plot(xs, ang, zs, linewidth=2.0, color=color, alpha=0.95)
    ax_corr.scatter(xs, ang, zs, s=16, color=color, depthshade=False, alpha=0.95)

# 2) 3D κ：κ(x)、κ̃(x) + 在 κ 上标注 κ̃-极大值
fig_k = plt.figure(figsize=(8, 9))
ax_k = fig_k.add_subplot(projection='3d')
ax_k.set_box_aspect([2.0, 4.0, 2.0])

for i, angle in enumerate(angle_ints[::-1]):
    xg   = per_angle[int(angle)]['xgrid']
    kap  = per_angle[int(angle)]['kappa']
    kapt = per_angle[int(angle)]['kappa_tilde']
    fidx = per_angle[int(angle)]['feat_idx']
    ax_k.plot(xg, np.full(len(xg), int(angle)), kap,
              color=adjust_lightness(discr_colors[i], 1.0),
              linewidth=2.0,
              path_effects=[pe.Stroke(linewidth=3.2, foreground='black'), pe.Normal()],
              label='κ(x)' if i == 0 else None)
    ax_k.plot(xg, np.full(len(xg), int(angle)), kapt,
              color='magenta', linewidth=1.2, alpha=0.7,
              label='κ~(x)' if i == 0 else None)
    if fidx.size > 0:
        ax_k.scatter(xg[fidx], np.full_like(xg[fidx], int(angle)), kap[fidx],
                     color='red', s=20, depthshade=False, alpha=0.95, marker='^',
                     label='κ~ maxima on κ' if i == 0 else None)

ax_k.set_xlabel('Normalized spectral lag (x)', labelpad=2)
ax_k.set_ylabel('Rotation angle (deg)', labelpad=10)
ax_k.set_xlim(-1.0, 1.0); ax_k.set_ylim(-15, 370)
ax_k.set_title('Signed curvature κ(x) and κ~(x)')
ax_k.zaxis.set_pane_color((1, 1, 1, 0))
ax_k.yaxis.set_pane_color((1, 1, 1, 0))
ax_k.xaxis.set_pane_color((1, 1, 1, 0))
ax_k.view_init(30, -130, 0)
ax_k.legend(loc='upper left')

# 3) 2D 检查图
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
    ax.plot(xg, kapt, lw=1.2, label='κ~(x)', color='C3', alpha=0.7)
    if fidx.size > 0:
        ax.scatter(xg[fidx], cg[fidx], s=22, color='magenta', label='κ~ maxima on corr')
        ax.scatter(xg[fidx], kap[fidx], s=22, color='red', marker='^', label='κ~ maxima on κ')
    ax.set_title(f'{int(angle)}°')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 控制台输出
print("\n[track counts]")
for k, t in enumerate(tracks, 1):
    print(f"  track#{k:02d}: counts={len(t['angles'])} "
          f"angles range=[{min(t['angles'])}, {max(t['angles'])}] "
          f"x范围=[{min(t['x']):+.3f}, {max(t['x']):+.3f}] "
          f"（其中 forced 连线 {sum(t['forced'])} 次）")
