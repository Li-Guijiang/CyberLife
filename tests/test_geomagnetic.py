# 地磁模块单元测试
# 直接跑：python -m unittest tests.test_geomagnetic -v
import sys
import pathlib
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geomagnetic_system.anomaly import (
    anomaly_basis,
    classify_metal,
    estimate_baseline,
    extract_targets,
    fit_dipole_centered,
)
from src.geomagnetic_system.calibration import EllipsoidCalibration, simulate_rotation
from src.geomagnetic_system.magnetometer import (
    SimulatedMagnetometer,
    dipole_field,
    earth_field,
)
from src.geomagnetic_system.scanner import AreaScanner, plan_lawnmower


class TestDipole(unittest.TestCase):
    """偶极子场公式和正演模型"""

    def test_inverse_cube_law(self):
        # 距离翻倍，场强应该降到八分之一
        m = (0.0, 0.0, 1.0)
        b1 = np.linalg.norm(dipole_field(m, np.array([0.0, 0.0, -1.0])))
        b2 = np.linalg.norm(dipole_field(m, np.array([0.0, 0.0, -2.0])))
        self.assertAlmostEqual(b1 / b2, 8.0, places=3)

    def test_axial_value(self):
        # 轴向 1 m、1 A·m^2 时应该是 200 nT
        b = np.linalg.norm(dipole_field((0.0, 0.0, 1.0), np.array([0.0, 0.0, -1.0])))
        self.assertAlmostEqual(b, 200.0, places=6)

    def test_basis_matches_forward_model(self):
        # 投影模型要等于偶极子场在背景场方向上的投影
        # 注意不能拿 |E+b|-|E| 逐点做相对比较：异常过零点附近投影接近 0，
        # 而 |E+b|-|E| 里还剩下垂直分量的二阶项，相对误差会被放大成几十个百分点
        e = earth_field()
        e_hat = e / np.linalg.norm(e)
        m, depth, h = 2.0, 0.5, 0.15
        target = np.array([0.0, 0.0, depth])
        peak = abs(m * anomaly_basis(0.0, 0.0, h + depth))
        for x, y in [(0.0, 0.0), (0.3, 0.0), (-0.25, 0.2)]:
            sensor = np.array([x, y, -h])
            b = dipole_field((0.0, 0.0, m), sensor - target)
            projection = float(np.dot(b, e_hat))
            norm_diff = np.linalg.norm(e + b) - np.linalg.norm(e)
            model_amp = m * anomaly_basis(x, y, h + depth)
            self.assertLess(abs(projection - model_amp), 1e-6)
            # 和实际读数（模长之差）比，按峰值归一化，差 2% 以内
            self.assertLess(abs(norm_diff - model_amp), 0.02 * peak)


class TestCalibration(unittest.TestCase):
    """椭球校准"""

    def test_recovers_hard_iron(self):
        hard = np.array([320.0, -180.0, 450.0])
        mag = SimulatedMagnetometer(noise_nt=5.0, hard_iron=hard, seed=11)
        cal = EllipsoidCalibration().fit(simulate_rotation(mag, n=400))
        err = np.linalg.norm(cal.offset - hard)
        self.assertLess(err, 15.0, "硬磁偏移误差 %.1f nT 太大" % err)

    def test_reduces_spread(self):
        mag = SimulatedMagnetometer(noise_nt=8.0, hard_iron=(300.0, -200.0, 400.0), seed=5)
        raw = simulate_rotation(mag, n=300)
        cal = EllipsoidCalibration().fit(raw)
        before = np.std(np.linalg.norm(raw, axis=1))
        after = np.std(np.linalg.norm(cal.apply(raw), axis=1))
        self.assertLess(after, before * 0.1, "校准后场强离散度没降下来")

    def test_rejects_flat_data(self):
        # 只在一个方向上采的数据不能拿来拟合，必须报错
        flat = np.tile([1000.0, 2000.0, 3000.0], (60, 1))
        with self.assertRaises(ValueError):
            EllipsoidCalibration().fit(flat)


class TestBaseline(unittest.TestCase):
    """背景场估计"""

    def test_mode_beats_median(self):
        # 一半测点落在强异常上，中位数会被抬高，直方图峰值不受影响
        rng = np.random.default_rng(0)
        quiet = 50000.0 + rng.normal(0, 10, 200)
        anomaly = 50000.0 + rng.uniform(200, 900, 200)
        vals = np.concatenate([quiet, anomaly])
        self.assertGreater(abs(np.median(vals) - 50000.0), 30.0)
        self.assertLess(abs(estimate_baseline(vals) - 50000.0), 15.0)


class TestDetection(unittest.TestCase):
    """目标检测与参数反演"""

    def test_single_target_recovery(self):
        truth = (1.20, 0.80, 0.55)
        mag = SimulatedMagnetometer(
            height=0.15, noise_nt=8.0,
            targets=[{"name": "t", "pos": truth, "moment": (0, 0, 2.5)}],
            seed=3,
        )
        scanner = AreaScanner(mag, sensor_height=0.15)
        data = scanner.scan_area(2.4, 1.6, 0.2, 0.1)
        self.assertEqual(len(data["targets"]), 1, "应该只报一个目标")
        t = data["targets"][0]
        self.assertLess(np.hypot(t["x"] - truth[0], t["y"] - truth[1]), 0.12)
        # 深度精度受测线间距限制，0.2 m 间距下实测误差在 25%~30%，
        # 这属于偶极子反演的正常水平，不是说算错了
        self.assertLess(abs(t["depth_m"] - truth[2]) / truth[2], 0.30)

    def test_empty_area(self):
        mag = SimulatedMagnetometer(height=0.15, noise_nt=8.0, targets=[], seed=9)
        scanner = AreaScanner(mag, sensor_height=0.15)
        data = scanner.scan_area(2.0, 1.2, 0.2, 0.1)
        self.assertEqual(len(data["targets"]), 0, "空地不该报目标")

    def test_classify_thresholds(self):
        self.assertEqual(classify_metal(0.05), "微金属/干扰")
        self.assertIn("小型", classify_metal(0.8))
        self.assertIn("中型", classify_metal(5.0))
        self.assertIn("大型", classify_metal(50.0))


class TestPath(unittest.TestCase):
    """路径规划"""

    def test_lawnmower_covers_area(self):
        path = plan_lawnmower(3.0, 2.0, 0.25, 0.1)
        self.assertGreater(len(path), 0)
        self.assertGreaterEqual(path[:, 0].min(), 0.0)
        self.assertLessEqual(path[:, 0].max(), 3.0)
        self.assertLessEqual(path[:, 1].max(), 2.0)
        # 相邻测线首尾相连，不能跳着走
        ys = np.unique(np.round(path[:, 1], 6))
        self.assertEqual(len(ys), 9)

    def test_rejects_bad_spacing(self):
        with self.assertRaises(ValueError):
            plan_lawnmower(3.0, 2.0, 0.0, 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
