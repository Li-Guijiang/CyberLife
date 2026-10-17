# 磁异常检测
# 思路：地磁场本身很稳，地下有铁磁物体就会在局部把场强抬高或压低。
# 用滑动中位数估一条基线，测值和基线一减就是异常量 ΔT。
# ΔT 超过噪声的几倍就认为有目标，再用偶极子模型估深度和磁矩。
import numpy as np

# 磁偶极子场强换算系数：B(nT) = COEF * m / r^3
# 推导：B = (μ0/4π)*m/r^3 = 1e-7*m/r^3 特斯拉，乘 1e9 得 nT
DIPOLE_FIELD_COEF = 100.0


class AnomalyDetector:
    """
    在线异常检测器，边扫边算

    基线用滑动中位数，比均值抗干扰，碰到目标也不会把基线拉偏。
    阈值 = k_sigma * 1.4826 * MAD，MAD 是中位数绝对偏差，
    这样阈值会随环境噪声自动调整，不用手工调。
    """

    def __init__(self, window=51, k_sigma=4.0, min_peak_nt=25.0):
        self.window = int(window)
        self.k_sigma = float(k_sigma)
        self.min_peak_nt = float(min_peak_nt)
        self._buf = []
        self.trace = []

    def reset(self):
        self._buf = []
        self.trace = []

    def update(self, b_vec):
        """
        喂一帧磁场读数，返回这一帧的判定结果
        b_vec 单位 nT，用总场强 |B| 做标量检测
        """
        total = float(np.linalg.norm(b_vec))
        self._buf.append(total)
        if len(self._buf) > self.window:
            self._buf.pop(0)

        buf = np.asarray(self._buf)
        baseline = float(np.median(buf))
        mad = float(np.median(np.abs(buf - baseline)))
        sigma = 1.4826 * mad if mad > 0 else 1.0
        # 刚开始数据少，阈值放宽一点，防止一上来就狂报
        warmup = len(buf) < 10
        threshold = max(self.k_sigma * sigma, self.min_peak_nt)
        delta = total - baseline
        is_anomaly = (not warmup) and abs(delta) >= threshold

        rec = {
            "total_nt": round(total, 2),
            "baseline_nt": round(baseline, 2),
            "delta_nt": round(delta, 2),
            "threshold_nt": round(threshold, 2),
            "sigma_nt": round(sigma, 3),
            "is_anomaly": bool(is_anomaly),
            "warmup": bool(warmup),
        }
        self.trace.append(rec)
        return rec


def estimate_baseline(values, bins=160):
    """
    估背景场（没目标时的场强），用直方图峰值，不能用中位数

    踩过的坑：扫描区里强目标的异常铺开两三米，能占到一半测点，
    这时候中位数被抬高了近百 nT，减出来的 ΔT 只剩真值的六成，
    深度和磁矩跟着一起错。
    背景点的数值挤在一个窄峰上，异常的分布是散开的，
    所以直方图最高那根柱子才是真背景。
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0
    counts, edges = np.histogram(v, bins=bins)
    k = int(np.argmax(counts))
    center = 0.5 * (edges[k] + edges[k + 1])
    width = edges[1] - edges[0]
    near = np.abs(v - center) <= width
    return float(np.mean(v[near])) if np.any(near) else float(center)


def robust_sigma(values):
    """用 MAD 估噪声标准差，不怕个别大异常值"""
    arr = np.asarray(values, dtype=float)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return 1.4826 * mad if mad > 0 else float(np.std(arr))


def estimate_noise(points, values, k=6, max_points=2000):
    """
    估噪声，用相邻测点的差值算，不能直接对整片 ΔT 求 MAD

    原因：强目标的异常会铺开一两米，整片数据算出来的 MAD 能达到上百 nT，
    阈值被抬到几百 nT，旁边的小目标就全被吞了。
    相邻两点的异常量几乎一样，差值里剩下的基本只有噪声。

    相邻两点差值绝对值的中位数 ≈ 0.954σ，反过来除一下就是 σ
    """
    pts = np.asarray(points, dtype=float)
    vals = np.asarray(values, dtype=float)
    n = len(pts)
    if n < k + 2:
        return robust_sigma(vals)

    # 点太多就抽样，不然 O(N^2) 太慢
    if n > max_points:
        keep = np.random.default_rng(0).choice(n, max_points, replace=False)
        pts, vals = pts[keep], vals[keep]
        n = max_points

    diffs = []
    for i in range(n):
        d = np.linalg.norm(pts - pts[i], axis=1)
        idx = np.argsort(d)[1:k + 1]
        diffs.append(np.abs(vals[idx] - vals[i]))
    diffs = np.concatenate(diffs)
    sigma = float(np.median(diffs) / 0.954)
    return max(sigma, 1e-6)


def anomaly_basis(dx, dy, vert, inclination_deg=45.0):
    """
    垂直磁化偶极子的总场异常基函数，单位 nT / (A·m^2)

    这里不能简单用 1/r^3，因为磁力计测的是总场强，
    测到的异常是异常矢量在背景地磁场方向上的投影，
    45 度倾角下正上方的投影系数是 2sin(I) 而不是 2，差了 41%。

    坐标：x 指北，y 指东，z 向下；vert 是传感器到目标的垂直距离
    """
    r = np.sqrt(dx * dx + dy * dy + vert * vert)
    r = np.maximum(r, 1e-6)
    inc = np.radians(inclination_deg)
    rz = -vert / r                       # ẑ·r̂
    e_dot_r = (np.cos(inc) * dx - np.sin(inc) * vert) / r   # ê·r̂
    term = 3.0 * rz * e_dot_r - np.sin(inc)
    return DIPOLE_FIELD_COEF * term / (r ** 3)


def fit_dipole_profile(dx, dy, amps, sensor_height, inclination_deg=45.0,
                       vert_min=0.05, vert_max=3.0, n_grid=160):
    """
    用一小片测点的异常量反推埋深和磁矩

    做法：垂直距离 D 扫一遍网格，每个 D 下异常和磁矩是线性关系，
    直接最小二乘解 m，取残差最小的 D。

    返回 dict；depth_m 是目标埋深（扣掉传感器高度）
    注意这是偶极子近似，假设目标垂直磁化、尺寸远小于埋深
    """
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    a = np.asarray(amps, dtype=float)
    if len(a) < 4:
        return None

    scale = max(float(np.max(np.abs(a))), 1e-9)
    best = None
    for vert in np.linspace(vert_min, vert_max, n_grid):
        basis = anomaly_basis(dx, dy, vert, inclination_deg)
        denom = float(basis @ basis)
        if denom <= 0:
            continue
        m = float(a @ basis) / denom
        if m <= 0:
            continue
        err = float(np.sqrt(np.mean((a - m * basis) ** 2))) / scale
        if best is None or err < best["err"]:
            best = {"err": err, "m": m, "vert": float(vert)}

    if best is None:
        return None
    return {
        "depth_m": round(max(best["vert"] - float(sensor_height), 0.01), 3),
        "vertical_dist_m": round(best["vert"], 3),
        "moment_am2": round(best["m"], 4),
        "fit_residual": round(best["err"], 4),
    }


def _fit_at(center, pts, dt, radius, sensor_height, inclination_deg):
    """以 center 为中心取半径内的点做一次拟合"""
    dist = np.linalg.norm(pts - center, axis=1)
    near = dist <= radius
    if np.sum(near) < 6:
        near = dist <= np.sort(dist)[min(5, len(dist) - 1)] + 1e-9
    return fit_dipole_profile(
        pts[near, 0] - center[0], pts[near, 1] - center[1], dt[near],
        sensor_height, inclination_deg,
    )


def fit_dipole_centered(points, amps, peak_pos, sensor_height,
                        inclination_deg=45.0, search_xy_m=0.45,
                        vert_min=0.05, vert_max=2.5):
    """
    连目标平面位置一起解，不能拿异常峰值当目标位置

    为什么：地磁场是斜的（倾角约 45°），垂直磁化的目标在总场上
    形成的异常并不对称，正峰会往南偏 0.2 m 左右。
    直接把峰值当目标位置，平面误差就有这么大。

    做法：在峰值附近按 0.1 m 网格搜中心，垂向按 50 档搜深度，
    每个组合都最小二乘解磁矩，取残差最小，再细搜一轮。

    返回 dict，x/y 是目标平面坐标，depth_m 是埋深
    """
    pts = np.asarray(points, dtype=float)
    a = np.asarray(amps, dtype=float)
    if len(a) < 6:
        return None
    scale = max(float(np.max(np.abs(a))), 1e-9)

    def score(cx, cy, vert):
        b = anomaly_basis(pts[:, 0] - cx, pts[:, 1] - cy, vert, inclination_deg)
        den = float(b @ b)
        if den <= 0:
            return None
        m = float(a @ b) / den
        if m <= 0:
            return None
        err = float(np.sqrt(np.mean((a - m * b) ** 2))) / scale
        return err, m

    best = None
    for vert in np.linspace(vert_min, vert_max, 50):
        for cx in np.linspace(peak_pos[0] - search_xy_m, peak_pos[0] + search_xy_m, 10):
            for cy in np.linspace(peak_pos[1] - search_xy_m, peak_pos[1] + search_xy_m, 10):
                r = score(cx, cy, vert)
                if r is not None and (best is None or r[0] < best[0]):
                    best = (r[0], r[1], float(cx), float(cy), float(vert))
    if best is None:
        return None

    # 第二轮细搜，步长缩到 0.04 m
    e0, m0, cx0, cy0, v0 = best
    for vert in np.linspace(max(v0 - 0.15, vert_min), v0 + 0.15, 16):
        for cx in np.linspace(cx0 - 0.1, cx0 + 0.1, 6):
            for cy in np.linspace(cy0 - 0.1, cy0 + 0.1, 6):
                r = score(cx, cy, vert)
                if r is not None and r[0] < best[0]:
                    best = (r[0], r[1], float(cx), float(cy), float(vert))

    err, m, cx, cy, vert = best
    return {
        "x": round(cx, 3),
        "y": round(cy, 3),
        "depth_m": round(max(vert - float(sensor_height), 0.01), 3),
        "vertical_dist_m": round(vert, 3),
        "moment_am2": round(m, 4),
        "fit_residual": round(err, 4),
    }


def classify_metal(moment_am2):
    """
    按磁矩粗分目标量级
    这几档是按常见铁磁物体估的，实车实测后要重新标定
    """
    m = float(moment_am2)
    if m < 0.2:
        return "微金属/干扰"
    if m < 2.0:
        return "小型金属（弹片、弹壳、铁丝）"
    if m < 20.0:
        return "中型金属（金属盒、器材、弹匣）"
    return "大型金属（车体、大型构件）"


def extract_targets(points, deltas, sensor_height=0.15, k_sigma=4.0,
                    min_peak_nt=25.0, min_distance_m=0.5,
                    inclination_deg=45.0, fit_radius_m=1.0):
    """
    从一整片扫描数据里挑出目标

    points 形状 (N, 2)，测点坐标
    deltas 形状 (N,)，各点异常量
    返回 (目标列表, 统计信息)

    挑峰分两步，缺一不可：
      1. 半径 min_distance_m 内不能有比自己更高的点，保证是局部极大，
         否则一个目标的缓坡上会被挑出好几个假峰
      2. 和已接受峰的距离要够远，避免同一个目标报多次

    深度用两轮拟合：先用 0.6 m 半径粗估，再按估出来的深度放大取点范围重拟合，
    因为埋得深的目标异常范围大，窗口太小会把模型带偏
    """
    pts = np.asarray(points, dtype=float)
    dt = np.asarray(deltas, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points 形状必须是 (N, 2)")
    if len(pts) != len(dt):
        raise ValueError("points 和 deltas 长度不一致")

    sigma = estimate_noise(pts, dt)
    sigma_rough = robust_sigma(dt)
    threshold = max(k_sigma * sigma, float(min_peak_nt))

    above = dt >= threshold
    if not np.any(above):
        return [], {"sigma_nt": round(sigma, 3), "sigma_rough_nt": round(sigma_rough, 3),
                    "threshold_nt": round(threshold, 2),
                    "peaks": 0, "points_total": int(len(pts)), "points_above": 0}

    order = np.argsort(-dt)
    peaks = []
    for idx in order:
        if dt[idx] < threshold:
            break
        p = pts[idx]
        dist = np.linalg.norm(pts - p, axis=1)
        # 条件1：邻近范围内没有更高的点
        if np.any(dt[dist < min_distance_m] > dt[idx] + 1e-9):
            continue
        # 条件2：离已接受的峰足够远
        if any(np.linalg.norm(p - pts[j]) < min_distance_m for j in peaks):
            continue
        peaks.append(int(idx))

    targets = []
    # 逐次消减：先按峰值从大到小拟合，拟合完把这个目标的场从数据里扣掉，
    # 再拟合下一个。不然强目标的远场尾巴会把弱目标的拟合窗口带偏
    # （实测弱目标深度会被压浅 40%）
    work = dt.copy()
    for idx in peaks:
        peak = pts[idx]
        dist = np.linalg.norm(pts - peak, axis=1)
        near = dist <= fit_radius_m
        if np.sum(near) < 8:
            near = dist <= np.sort(dist)[min(7, len(dist) - 1)] + 1e-9

        # 第一轮用原始数据找大致位置，第二轮用扣减后的数据定参数
        fit = fit_dipole_centered(pts[near], work[near], peak, sensor_height,
                                  inclination_deg)
        if fit is not None:
            model = fit["moment_am2"] * anomaly_basis(
                pts[:, 0] - fit["x"], pts[:, 1] - fit["y"],
                fit["vertical_dist_m"], inclination_deg)
            work = work - model

        peak_amp = float(dt[idx])
        if fit is None:
            fit = {}
        targets.append({
            "x": fit.get("x", round(float(peak[0]), 3)),
            "y": fit.get("y", round(float(peak[1]), 3)),
            "peak_x": round(float(peak[0]), 3),
            "peak_y": round(float(peak[1]), 3),
            "peak_nt": round(peak_amp, 2),
            "snr": round(peak_amp / sigma, 1) if sigma > 0 else None,
            "depth_m": fit.get("depth_m"),
            "moment_am2": fit.get("moment_am2"),
            "category": classify_metal(fit["moment_am2"]) if fit.get("moment_am2") else "无法判定",
            "fit_residual": fit.get("fit_residual"),
        })

    targets.sort(key=lambda t: -t["peak_nt"])
    for i, t in enumerate(targets, start=1):
        t["id"] = "T%02d" % i

    info = {
        "sigma_nt": round(sigma, 3),
        "sigma_rough_nt": round(sigma_rough, 3),
        "threshold_nt": round(threshold, 2),
        "peaks": len(targets),
        "points_total": int(len(pts)),
        "points_above": int(np.sum(above)),
    }
    return targets, info
