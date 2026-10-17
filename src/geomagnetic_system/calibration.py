# 磁力计校准
# 机身上的铁件会把地磁场扭歪，分两种：
#   硬磁误差：机身自带磁性，相当于在读数上加一个固定偏移
#   软磁误差：铁件把磁场线拉弯，相当于给读数乘一个矩阵
# 解决办法是转一圈采一圈数据，拟合一个椭球，再把椭球拉回球
import numpy as np


class EllipsoidCalibration:
    """
    椭球拟合校准

    原理：理想情况下磁力计转一圈，三个轴读数应该落在一个球面上。
    实际因为有硬磁软磁误差，落成一个歪掉的椭球。
    拟合出椭球的球心就是硬磁偏移，椭球三个半轴长度比就是软磁矩阵。
    把椭球平移加缩放回球，校准就完成了。
    """

    def __init__(self):
        self.offset = np.zeros(3)
        self.matrix = np.eye(3)
        self.field_strength = 0.0
        self.residual_rms = None
        self.n_samples = 0

    def fit(self, samples):
        """
        samples 形状 (N, 3)，单位 nT，要求至少 30 帧且三个方向都要有变化
        返回 self，方便链式调用
        """
        b = np.asarray(samples, dtype=float)
        if b.ndim != 2 or b.shape[1] != 3:
            raise ValueError("样本形状必须是 (N, 3)")
        if len(b) < 30:
            raise ValueError("样本太少，至少 30 帧，现在只有 %d 帧" % len(b))

        # 三个轴的跨度都得够，否则椭球退化成一条线，拟合出来是假的
        span = b.max(axis=0) - b.min(axis=0)
        if np.any(span < 50.0):
            raise ValueError(
                "某个轴的数据几乎没变化（%s nT），转的时候要三个方向都转到"
                % np.round(span, 1)
            )

        x, y, z = b[:, 0], b[:, 1], b[:, 2]
        # 一般二次曲面：a x^2 + b y^2 + c z^2 + 2d xy + 2e xz + 2f yz + 2g x + 2h y + 2i z = 1
        m = np.column_stack([
            x * x, y * y, z * z,
            2 * x * y, 2 * x * z, 2 * y * z,
            2 * x, 2 * y, 2 * z,
        ])
        rhs = np.ones(len(b))
        p, *_ = np.linalg.lstsq(m, rhs, rcond=None)

        a = np.array([
            [p[0], p[3], p[4]],
            [p[3], p[1], p[5]],
            [p[4], p[5], p[2]],
        ])
        v = np.array([p[6], p[7], p[8]])

        # 椭球中心 = -A^-1 v
        try:
            center = -np.linalg.solve(a, v)
        except np.linalg.LinAlgError:
            raise ValueError("拟合矩阵奇异，数据可能共面或太少，重新采一遍")

        # (B-c)^T A (B-c) = 1 + v^T A^-1 v = k，把右边归一成 1
        k = 1.0 + float(v @ np.linalg.solve(a, v))
        if k <= 0:
            raise ValueError("拟合结果不是椭球，数据有问题")
        w = a / k

        # 用 Cholesky 分解拿到从椭球到球的线性变换
        try:
            chol = np.linalg.cholesky(w)
        except np.linalg.LinAlgError:
            # 不是正定就先做特征分解，再拼一个对称平方根
            vals, vecs = np.linalg.eigh(w)
            vals = np.clip(vals, 1e-12, None)
            chol = vecs @ np.diag(np.sqrt(vals))

        self.offset = center
        self.field_strength = float(np.mean(np.linalg.norm(b - center, axis=1)))
        self.matrix = chol.T * self.field_strength
        self.n_samples = len(b)

        # 校准后每个样本的模长应该都等于 field_strength，看偏差多少
        cal = self.apply(b)
        norms = np.linalg.norm(cal, axis=1)
        self.residual_rms = float(np.sqrt(np.mean((norms - self.field_strength) ** 2)))
        return self

    def apply(self, b):
        """把原始读数搬进校准后的坐标系，返回同形状数组"""
        arr = np.asarray(b, dtype=float)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        out = (arr - self.offset) @ self.matrix.T
        return out[0] if single else out

    def quality(self):
        """
        校准质量自评
        残差占场强的比例低于 2% 算好，2%~5% 能用，超过 5% 要重新采
        """
        if self.residual_rms is None or self.field_strength <= 0:
            return {"ok": False, "reason": "还没校准"}
        ratio = self.residual_rms / self.field_strength
        if ratio < 0.02:
            level = "良好"
        elif ratio < 0.05:
            level = "可用"
        else:
            level = "偏差偏大，建议重采"
        return {
            "ok": ratio < 0.05,
            "residual_nt": round(self.residual_rms, 2),
            "residual_ratio": round(ratio, 4),
            "field_strength_nt": round(self.field_strength, 1),
            "level": level,
            "samples": self.n_samples,
        }


def _rot_from_to(a, b):
    """构造把单位向量 a 转到 b 的旋转矩阵（罗德里格斯公式）"""
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def simulate_rotation(sim, n=360, seed=3):
    """
    转一圈采样，给仿真和验收用

    姿态方向按斐波那契球面均匀取，这样测点铺满整个球面，
    拟合出来的椭球才靠谱。只绕一两个轴转的话，测点只覆盖一条带，
    椭球在带宽方向没有约束，拟合出来的参数是虚的。

    实物上不用这个函数，拿在手里各个方向都转一遍就行
    """
    rng = np.random.default_rng(seed)
    base = sim.earth.copy()
    e0 = base / np.linalg.norm(base)
    samples = []
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        # 斐波那契球面：z 均匀分布，方位角按黄金角递增
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = np.sqrt(max(0.0, 1.0 - z * z))
        phi = i * golden
        u = np.array([r * np.cos(phi), r * np.sin(phi), z])
        vec = _rot_from_to(e0, u) @ base
        vec = sim.soft_iron @ vec + sim.hard_iron
        vec = vec + rng.normal(0, sim.noise_nt, size=3)
        samples.append(vec)
    return np.asarray(samples)
