# -*- coding: utf-8 -*-
"""视觉离线检测器的精度量化测试。

用 测试图/ 下的 10 个场景、15 个带真值的目标核对四件事：
检出率、定位精度（IoU）、材质判别准确率、虚警数。
真值写在 测试图/真值.json，图中目标的位置与类别在生成时固定，因此本测试
结果可复现，不依赖随机数。

运行
----
python -m unittest tests.test_vision_accuracy -v
"""
import json
import pathlib
import unittest

import cv2
import numpy as np

import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.vision_system import object_detect as od  # noqa: E402

SCENE_DIR = ROOT / "测试图"
TRUTH_FILE = SCENE_DIR / "真值.json"
IOU_MATCH = 0.30          # 匹配阈值
IOU_GOOD = 0.50           # 定位合格阈值


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / float(aw * ah + bw * bh - inter)


def load(path):
    """中文路径用 imdecode，绕开 cv2.imread 的编码限制。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


class TestVisionAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth = json.loads(TRUTH_FILE.read_text(encoding="utf-8"))
        cls.result = {}
        for name, items in cls.truth.items():
            dets = od.detect(load(SCENE_DIR / (name + ".png")))
            used, rows = set(), []
            for it in items:
                best, bi = 0.0, -1
                for i, d in enumerate(dets):
                    if i in used:
                        continue
                    v = iou(d.box, it["box"])
                    if v > best:
                        best, bi = v, i
                rows.append((it, dets[bi] if best >= IOU_MATCH else None, best))
                if best >= IOU_MATCH:
                    used.add(bi)
            cls.result[name] = (rows, len(dets) - len(used))

    def test_all_scenes_present(self):
        """10 个场景图与真值文件齐全。"""
        self.assertEqual(len(self.truth), 10)
        for name in self.truth:
            self.assertTrue((SCENE_DIR / (name + ".png")).exists(), name)

    def test_detection_rate(self):
        """真值目标全部检出。"""
        for name, (rows, _) in self.result.items():
            for it, det, score in rows:
                self.assertIsNotNone(det, "%s 未检出 %s" % (name, it["label"]))

    def test_localization(self):
        """定位 IoU 不低于 0.50，保证框住的是目标本体。"""
        for name, (rows, _) in self.result.items():
            for it, det, score in rows:
                if det is None:
                    continue
                self.assertGreaterEqual(score, IOU_GOOD,
                                        "%s 的 %s IoU=%.2f" % (name, it["label"], score))

    def test_material(self):
        """材质判别与真值一致。"""
        for name, (rows, _) in self.result.items():
            for it, det, score in rows:
                if det is None or not it.get("material"):
                    continue
                self.assertEqual(det.material, it["material"],
                                 "%s 的 %s 判为 %s" % (name, it["label"], det.material))

    def test_no_false_alarm(self):
        """没有多余的目标框。"""
        for name, (_, extra) in self.result.items():
            self.assertEqual(extra, 0, "%s 多出 %d 个框" % (name, extra))

    def test_key_targets(self):
        """金属与手机这两类关键目标必须给出具体类别，不能落到兜底类别。"""
        labels = set()
        for rows, _ in self.result.values():
            for it, det, score in rows:
                if det is not None:
                    labels.add(det.label)
        self.assertIn("金属构件", labels)
        self.assertIn("手机", labels)

    def test_feature_vector(self):
        """目标区域的特征量：金属件的直边占比与明暗起伏都高于塑料袋。"""
        def feats_of(scene):
            dets = od.detect(load(SCENE_DIR / (scene + ".png")))
            self.assertTrue(dets, scene)
            return dets[0].features

        f_metal = feats_of("S01_金属罐")
        f_bag = feats_of("S05_塑料袋")
        self.assertGreater(f_metal["straight_ratio"], f_bag["straight_ratio"])
        self.assertGreater(f_metal["v_std"], f_bag["v_std"])
        self.assertLess(f_bag["smooth_ratio"], f_metal["smooth_ratio"])
        self.assertGreaterEqual(od.metal_score(f_metal), 0.55)
        self.assertLess(od.metal_score(f_bag), 0.55)


if __name__ == "__main__":
    unittest.main(verbosity=2)
