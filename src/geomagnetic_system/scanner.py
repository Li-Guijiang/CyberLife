# 扫描规划与成图
# 机器人按弓字形走完一片区域，边走边采，最后把异常量拉成热图
import numpy as np

from .anomaly import AnomalyDetector, estimate_baseline, extract_targets


def plan_lawnmower(width, height, lane_spacing, step, margin=0.0):
    """
    弓字形（来回扫）路径规划

    width, height 是作业区域长宽，lane_spacing 是相邻测线间距，
    step 是测线上采样点间距，margin 是离边界的余量。
    返回 (N, 2) 的路径点，单位 m，左下角是原点。
    """
    if lane_spacing <= 0 or step <= 0:
        raise ValueError("lane_spacing 和 step 必须大于 0")

    x0, x1 = margin, width - margin
    y0, y1 = margin, height - margin
    if x1 <= x0 or y1 <= y0:
        raise ValueError("余量太大，区域被切没了")

    n_lanes = int(np.floor((y1 - y0) / lane_spacing)) + 1
    path = []
    for i in range(n_lanes):
        y = y0 + i * lane_spacing
        if y > y1 + 1e-9:
            break
        xs = np.arange(x0, x1 + 1e-9, step)
        # 偶数行走正向，奇数行走反向，连成一条不回头的线
        if i % 2 == 1:
            xs = xs[::-1]
        for x in xs:
            path.append((float(x), float(y)))
    return np.asarray(path, dtype=float)


class AreaScanner:
    """
    区域扫描器

    magnetometer 传进来的是哪个驱动都行，仿真、CSV、串口都走同一套逻辑。
    detector 不传就自己建一个。
    """

    def __init__(self, magnetometer, detector=None, sensor_height=0.15,
                 calibration=None):
        self.mag = magnetometer
        self.detector = detector or AnomalyDetector()
        self.sensor_height = float(sensor_height)
        self.calibration = calibration

    def scan_path(self, path):
        """
        按给定路径扫一遍，返回原始记录
        每到一个点：挪过去 -> 读一帧 -> 校准 -> 算异常
        """
        pts = []
        deltas = []
        totals = []
        flags = []
        for (x, y) in np.asarray(path, dtype=float):
            if hasattr(self.mag, "set_position"):
                self.mag.set_position(x, y)
            b = self.mag.read()
            if self.calibration is not None:
                b = self.calibration.apply(b)
            rec = self.detector.update(b)
            pts.append((x, y))
            deltas.append(rec["delta_nt"])
            totals.append(rec["total_nt"])
            flags.append(rec["is_anomaly"])
        totals_arr = np.asarray(totals, dtype=float)
        # 成图用整片扫描估的背景场，比滑动基线稳；
        # 也不能用中位数，强目标异常占了半片区域时中位数会被抬高
        baseline = estimate_baseline(totals_arr)
        return {
            "points": np.asarray(pts, dtype=float),
            "deltas": totals_arr - baseline,
            "deltas_online": np.asarray(deltas, dtype=float),
            "baseline_nt": round(baseline, 2),
            "totals": totals_arr,
            "online_flags": np.asarray(flags, dtype=bool),
        }

    def scan_area(self, width, height, lane_spacing, step, margin=0.0):
        """规划路径再扫，返回结果加目标列表"""
        path = plan_lawnmower(width, height, lane_spacing, step, margin)
        data = self.scan_path(path)
        targets, info = extract_targets(
            data["points"], data["deltas"],
            sensor_height=self.sensor_height,
        )
        data["path"] = path
        data["targets"] = targets
        data["info"] = info
        data["baseline_nt"] = round(estimate_baseline(data["totals"]), 2)
        return data


def grid_anomaly(points, values, resolution=81, radius_m=0.45, power=2.0):
    """
    把散点异常量拉成规则网格，用反距离加权插值

    不用 scipy 的 griddata，是为了少一个依赖，
    扫描点本来就密，反距离加权够用了。
    返回 X, Y, Z 三个二维数组，可以直接喂 imshow
    """
    pts = np.asarray(points, dtype=float)
    vals = np.asarray(values, dtype=float)
    if len(pts) < 3:
        raise ValueError("点太少，插不出网格")

    xs = np.linspace(pts[:, 0].min(), pts[:, 0].max(), resolution)
    ys = np.linspace(pts[:, 1].min(), pts[:, 1].max(), int(resolution * 0.7) + 1)
    gx, gy = np.meshgrid(xs, ys)

    flat = np.column_stack([gx.ravel(), gy.ravel()])
    out = np.zeros(len(flat))
    for i, p in enumerate(flat):
        d = np.linalg.norm(pts - p, axis=1)
        # 只取半径内的点，避免远处的点把结果抹平
        near = d <= radius_m
        if not np.any(near):
            j = int(np.argmin(d))
            out[i] = vals[j]
            continue
        w = 1.0 / np.power(np.maximum(d[near], 1e-6), power)
        out[i] = float(np.sum(w * vals[near]) / np.sum(w))
    return gx, gy, out.reshape(gx.shape)
