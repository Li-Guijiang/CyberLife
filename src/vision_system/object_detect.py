# -*- coding: utf-8 -*-
"""CyberLife 视觉感知模块：离线目标检测与金属材质判别。

设计约束
--------
1. 全程离线：不加载在线权重、不发起任何网络请求，不依赖 ultralytics / torch。
2. 只依赖 opencv-python 与 numpy：
   VideoCapture / imdecode 采集，cvtColor 色彩空间转换，inRange 阈值分割，
   morphologyEx 形态学，Canny / Sobel 边缘与梯度，connectedComponents 区域分割，
   findContours 轮廓，HoughLinesP 直线，Laplacian 纹理，rectangle / putText 绘制；
   numpy 负责掩膜运算与统计量。
3. 单帧 640×480 约 12~22 ms（本机 CPU 实测），可实时叠加。

检测分两路提议
--------------
甲路（颜色）：按黄橙高可视、红色目标、深色器件、中性浅色四组做阈值分割。
乙路（边缘）：对"边缘图之外的连通区"做区域分割，取中性色区域。
    这一路专门解决灰金属贴在灰墙、灰地面上的情况——纯颜色分割会把两者
    连成一片，而边缘闭合之后金属内腔是独立连通区。
两路结果按包围框重合度去重。

金属判别的依据是光学证据，不是电磁测量
--------------------------------------
金属在可见光下的稳定表现：色饱和度低、镜面高光带陡边、亮度沿曲面起伏、
内部大面积平整。四项加权得到 0~1 的金属置信度。同色系塑料件与金属特征
高度重叠，因此本模块给出的是概率判断；金属的最终确认由 geomagnetic_system
的电磁探测完成，两者互补。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

HUE_YELLOW = (12, 40)
HUE_RED = ((0, 8), (172, 179))


@dataclass
class Detection:
    """一个检测结果。box 为 (x, y, w, h)，单位像素。"""

    label: str
    conf: float
    box: Tuple[int, int, int, int]
    material: str
    material_conf: float
    features: dict

    def as_row(self) -> str:
        x, y, w, h = self.box
        return "%-14s %.2f  %-6s %.2f  (%d,%d,%d,%d)" % (
            self.label, self.conf, self.material, self.material_conf, x, y, w, h)


# ---------------------------------------------------------------- 材质特征
def _masked(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把掩膜外的像素压成均值，避免轮廓边界产生假梯度。"""
    sel = mask > 0
    if sel.sum() == 0:
        return gray
    out = gray.copy()
    out[~sel] = int(gray[sel].mean())
    return out


def _texture(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    m = int(0.12 * min(h, w))
    core = gray[m:h - m, m:w - m] if min(h, w) > 24 else gray
    lap = cv2.Laplacian(core, cv2.CV_32F, ksize=3)
    return float(lap.var() / max(1.0, core.size) * 1000.0)


def _straight_ratio(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    h, w = gray.shape[:2]
    seg = 0.0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=26,
                            minLineLength=max(16, int(0.16 * min(h, w))),
                            maxLineGap=6)
    if lines is not None:
        # OpenCV 4.x 返回 (N,1,4)，5.x 返回 (N,4)
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            seg += float(np.hypot(x2 - x1, y2 - y1))
    return seg / max(1.0, 2.0 * (h + w))


def material_features(bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """提取材质判别所需的光学特征。mask 为 None 时整幅参与统计。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if mask is None:
        mask = np.full(gray.shape, 255, np.uint8)
    sel = mask > 0
    if sel.sum() < 200:
        sel = np.ones_like(sel)

    s = hsv[:, :, 1][sel].astype(np.float32)
    v = hsv[:, :, 2][sel].astype(np.float32)
    gray_m = _masked(gray, mask)
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8))
    if (inner > 0).sum() < 0.3 * sel.sum():
        inner = mask
    inner_sel = inner > 0

    # 高光：亮且不饱和，并且 5 像素内存在强亮度台阶
    gx = cv2.Sobel(gray_m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_m, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    bright = ((hsv[:, :, 2] >= 228) & (hsv[:, :, 1] <= 42) & inner_sel)
    near = cv2.dilate((grad > 120).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    spec_ratio = float((bright & near).sum()) / float(max(1, sel.sum()))

    smooth_ratio = float((grad[inner_sel] < 14).sum()) / float(max(1, inner_sel.sum()))
    # 明显彩色的像素占比：裸金属、灰白塑料接近 0，印刷标签、涂层、织物图案偏高
    color_ratio = float((s >= 90).sum()) / float(max(1, s.size))

    return {
        "s_mean": float(s.mean()),
        "v_mean": float(v.mean()),
        "v_std": float(v.std()),
        "spec_ratio": spec_ratio,
        "smooth_ratio": smooth_ratio,
        "straight_ratio": _straight_ratio(gray_m),
        "texture": _texture(gray_m),
        "edge_density": float((grad > 60).mean()),
        "color_ratio": color_ratio,
    }


def metal_score(f: dict) -> float:
    """五项光学证据加权成 0~1 的金属置信度，权重由标定集确定。

    权重依据标定集实测分布：
    1. 饱和度——裸金属与灰白塑料低于 10，带商标的塑料瓶 59，彩色件 200 以上；
    2. 镜面高光比——实测金属 0.01~0.06；
    3. 亮度离散度——金属表面随曲率明暗起伏，v_std 39~45，
       而塑料袋、纸箱这类漫反射柔性件只有 13~20，这一项区分力最强；
    4. 直边占比——金属为刚性件，实测 0.9~1.6，塑料袋 0.3~0.6；
    5. 内部平整区占比。
    另有两项折扣：大面积彩色像素（印刷标签、涂层）不是裸金属；
    整片接近纯白（纸、白袋）降低权重。
    """
    if f["s_mean"] > 105:                    # 明显有颜色，排除裸金属
        return 0.0
    score = (
        0.32 * float(np.clip((95.0 - f["s_mean"]) / 60.0, 0.0, 1.0))
        + 0.18 * float(np.clip(f["spec_ratio"] / 0.04, 0.0, 1.0))
        + 0.24 * float(np.clip((f["v_std"] - 20.0) / 28.0, 0.0, 1.0))
        + 0.16 * float(np.clip((f["straight_ratio"] - 0.55) / 0.35, 0.0, 1.0))
        + 0.14 * float(np.clip((f["smooth_ratio"] - 0.45) / 0.35, 0.0, 1.0))
    )
    # 彩色像素占比越高，越可能是有印刷标签或涂层的非金属件
    score *= 1.0 - 0.80 * float(np.clip(f.get("color_ratio", 0.0) / 0.45, 0.0, 1.0))
    if f["v_mean"] >= 228:                   # 整片接近纯白，多为纸与白袋
        score *= 0.30
    return float(np.clip(score, 0.0, 1.0))


def classify_material(f: dict) -> Tuple[str, float]:
    """返回 (材质名, 置信度)。置信度按到阈值的距离折算到 0.5~0.98。

    金属与塑料的判别先走 metal_score；塑料与织物/纸张的区分另用"柔性度"：
    织物与纸张有褶皱，直边少且大面积平滑区少，两者得分同时偏低。
    安全帽、电饭锅内胆一类刚性塑料壳的平滑区占比高，柔性度低，仍判为塑料。
    """
    score = metal_score(f)
    if f["v_mean"] < 72 and f["s_mean"] < 70:
        return "非金属", float(np.clip(0.55 + (72 - f["v_mean"]) / 100.0, 0.5, 0.95))
    if score >= 0.55:
        return "金属", float(np.clip(0.5 + (score - 0.55) * 1.6, 0.5, 0.98))
    flex = (0.45 * float(np.clip((0.70 - f["straight_ratio"]) / 0.25, 0.0, 1.0))
            + 0.55 * float(np.clip((0.45 - f["smooth_ratio"]) / 0.20, 0.0, 1.0)))
    if f["texture"] >= 18.0 and flex >= 0.5:
        return "织物或纸张", float(np.clip(0.5 + min(1.0, f["texture"] / 60.0) * 0.4, 0.5, 0.9))
    if f["v_mean"] >= 140:
        return "塑料", float(np.clip(0.5 + (0.55 - score) * 1.4, 0.5, 0.92))
    return "未知", 0.5


# ---------------------------------------------------------------- 区域提议
def _clean(mask: np.ndarray, k: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _mask_groups(hsv: np.ndarray):
    yield "高可视", cv2.inRange(hsv, (HUE_YELLOW[0], 110, 110), (HUE_YELLOW[1], 255, 255))
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (HUE_RED[0][0], 105, 80), (HUE_RED[0][1], 255, 255)),
        cv2.inRange(hsv, (HUE_RED[1][0], 105, 80), (HUE_RED[1][1], 255, 255)))
    yield "红色", red
    yield "深色", cv2.inRange(hsv, (0, 0, 0), (179, 95, 78))
    yield "中性", cv2.inRange(hsv, (0, 0, 95), (179, 52, 255))


def _accept(cnt, w: int, h: int, min_area: int) -> Optional[Tuple[int, int, int, int]]:
    area = cv2.contourArea(cnt)
    if area < min_area:
        return None
    x, y, bw, bh = cv2.boundingRect(cnt)
    if bw < 28 or bh < 28 or bw * bh > 0.55 * w * h:
        return None
    if not (0.20 <= bw / float(bh) <= 5.0):
        return None
    if area / float(bw * bh) < 0.45:
        return None
    return int(x), int(y), int(bw), int(bh)


def _propose_color(frame: np.ndarray, hsv: np.ndarray, min_area: int):
    h, w = frame.shape[:2]
    for group, raw in _mask_groups(hsv):
        mask = _clean(raw)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            box = _accept(cnt, w, h, min_area)
            if box is None:
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.drawContours(m, [cnt], -1, 255, -1)
            yield group, m, box


def _propose_edge(frame: np.ndarray, min_area: int):
    """边缘闭合后的内部连通区，专门补充中性色（金属）目标。"""
    h, w = frame.shape[:2]
    gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 5)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    free = cv2.bitwise_not(edges)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area or bw < 28 or bh < 28:
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue                                  # 触边连通区按背景处理
        if bw * bh > 0.40 * w * h:
            continue
        if area / float(bw * bh) < 0.55:
            continue
        m = (lab == i).astype(np.uint8) * 255
        sel = m > 0
        if float(hsv[:, :, 1][sel].mean()) > 72:       # 只补中性色，彩色交给甲路
            continue
        yield "边缘", m, (int(x), int(y), int(bw), int(bh))


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / float(aw * ah + bw * bh - inter)


def _label_for(group: str, feats: dict, material: str) -> str:
    """按色彩分组定类别，分组不足以定类时退回材质。

    高可视与红色两条规则要求整片区域本身饱和（s_mean≥92）。局部彩色标签
    （例如塑料瓶身的一圈橙色商标）会把分组拉到"高可视"，但整体平均饱和度
    只有 60 上下，此时不能判成反光背心，应退回材质给出"塑料容器"。
    """
    h = feats.get("h_mean", 0.0)
    s = feats["s_mean"]
    v = feats["v_mean"]
    if s >= 92 and (group == "高可视" or HUE_YELLOW[0] <= h <= HUE_YELLOW[1]):
        return "安全头盔或反光背心"
    if s >= 92 and (group == "红色" or h <= HUE_RED[0][1] or h >= HUE_RED[1][0]):
        return "红色标记目标"
    if group == "深色" or (v <= 80 and s <= 80):
        # 深色低饱和、长宽比接近 16:9 到 3:2 的板状物，按手机计
        if 1.45 <= feats.get("aspect", 1.0) <= 2.85:
            return "手机"
        return "手持电子设备"
    if material == "金属":
        return "金属构件"
    if material == "塑料":
        return "塑料容器"
    if material == "织物或纸张":
        return "塑料袋或纸张"
    return "未分类目标"


def detect(frame: np.ndarray, min_area: int = 1400, max_objects: int = 20) -> List[Detection]:
    """对一帧 BGR 图像做检测，返回按置信度降序排列的结果。"""
    if frame is None or frame.size == 0:
        return []
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    h, w = frame.shape[:2]
    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 1) 收集候选，并记录各自的平均色，供合并时判断是否同一物体
    cands = []
    for group, mask, box in list(_propose_color(frame, hsv, min_area)) + \
            list(_propose_edge(frame, min_area)):
        sel = mask > 0
        if sel.sum() < 220:
            continue
        cands.append({
            "group": group, "mask": mask, "box": box,
            "s": float(hsv[:, :, 1][sel].mean()),
            "v": float(hsv[:, :, 2][sel].mean()),
            "area": int(sel.sum()),
        })

    # 2) 同一物体被颜色边界（如瓶身标签、帽檐）切开的碎片并回一个整体
    groups = _merge_groups(cands)

    out: List[Detection] = []
    for items in groups:
        mask = items[0]["mask"].copy()
        for it in items[1:]:
            mask = cv2.bitwise_or(mask, it["mask"])
        ys, xs = np.nonzero(mask)
        if len(xs) < 220:
            continue
        x, y = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max() - x + 1), int(ys.max() - y + 1)
        if bw * bh > 0.55 * w * h:
            continue

        roi = frame[y:y + bh, x:x + bw]
        roi_mask = mask[y:y + bh, x:x + bw]
        feats = material_features(roi, roi_mask)
        sel = roi_mask > 0
        feats["h_mean"] = float(hsv[y:y + bh, x:x + bw, 0][sel].mean())
        feats["extent"] = float(sel.sum()) / float(bw * bh)
        feats["aspect"] = float(max(bw, bh)) / float(max(1, min(bw, bh)))

        material, mconf = classify_material(feats)
        label = _label_for(items[0]["group"], feats, material)

        sat = feats["s_mean"] / 255.0
        shape = float(np.clip((feats["extent"] - 0.45) / 0.45, 0.0, 1.0))
        size = float(np.clip(np.log10(max(1.0, bw * bh) / 1500.0) / 1.4, 0.0, 1.0))
        conf = float(np.clip(0.44 + 0.22 * sat + 0.20 * shape + 0.10 * size, 0.5, 0.98))

        out.append(Detection(
            label=label, conf=conf, box=(x, y, bw, bh),
            material=material, material_conf=float(mconf),
            features={k: round(v, 4) for k, v in feats.items() if isinstance(v, float)},
        ))

    out.sort(key=lambda d: d.conf, reverse=True)
    kept: List[Detection] = []
    for d in out:
        if all(_iou(d.box, k.box) < 0.45 for k in kept) and \
                all(_contained(d.box, k.box) < 0.70 for k in kept):
            kept.append(d)
        if len(kept) >= max_objects:
            break
    return kept


def _contained(a: Sequence[int], b: Sequence[int]) -> float:
    """a 的框有多少比例落在 b 的框内，用于抑制大框里的小碎片。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1) / float(max(1, aw * ah))


def _near(a: dict, b: dict, gap: int = 26, ds: float = 45.0, dv: float = 45.0) -> bool:
    """两个候选是否属于同一物体。

    三种情形并回一个整体：
    1. 包围框重叠较多；
    2. 隔一条缝且垂直于缝的尺寸基本一致——同一物体的碎片（金属件被自身高光
       切成两半、瓶身被标签环切成上下两段）在垂直于缝隙方向上尺寸相等，
       而两个独立物体几乎不会这么巧；
    3. 缝隙很小（8 像素内）且颜色相近。
    """
    ax, ay, aw, ah = a["box"]
    bx, by, bw, bh = b["box"]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 > x1 and y2 > y1:
        inter = (x2 - x1) * (y2 - y1)
        if inter / float(min(aw * ah, bw * bh)) >= 0.15:
            return True

    dx = max(0, x1 - x2) if x2 <= x1 else 0
    dy = max(0, y1 - y2) if y2 <= y1 else 0
    if dx > 8 and dy > 8:
        return False

    # 上下叠放且宽度一致：同一物体被横向条带（标签环、反光条）切成两段的典型形态。
    # 左右并排不套用这条规则，并排且尺寸相近的多为两个独立物体。
    if dy > 0 and dx == 0 and dy <= 60 and abs(aw - bw) <= 0.12 * max(aw, bw):
        return True
    if dx <= 8 and dy <= 8 and abs(a["s"] - b["s"]) < ds and abs(a["v"] - b["v"]) < dv:
        return True
    return False


def _merge_groups(cands: List[dict]) -> List[List[dict]]:
    """并查集：把判定为同一物体的候选归到一组。"""
    n = len(cands)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _near(cands[i], cands[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    buckets: dict = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(cands[i])
    return list(buckets.values())


def summarize(dets: Sequence[Detection]) -> dict:
    by_label: dict = {}
    by_material: dict = {}
    for d in dets:
        by_label[d.label] = by_label.get(d.label, 0) + 1
        by_material[d.material] = by_material.get(d.material, 0) + 1
    return {"total": len(dets), "by_label": by_label, "by_material": by_material}


# ---------------------------------------------------------------- 绘制与输出
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
)
_FONT_CACHE: dict = {}

MATERIAL_COLOR = {"金属": (60, 220, 255), "塑料": (110, 220, 110),
                  "织物或纸张": (205, 205, 205), "非金属": (200, 160, 255),
                  "未知": (180, 180, 180)}


def _font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    try:
        from PIL import ImageFont
    except Exception:
        _FONT_CACHE[size] = None
        return None
    for path in _FONT_CANDIDATES:
        try:
            _FONT_CACHE[size] = ImageFont.truetype(path, size)
            return _FONT_CACHE[size]
        except Exception:
            continue
    _FONT_CACHE[size] = None
    return None


def _corner_ticks(draw, x, y, w, h, color, seg=20, width=3):
    """四角瞄准式刻线，比整圈细框更不遮挡目标本体。"""
    x2, y2 = x + w, y + h
    for (cx, cy, dx, dy) in ((x, y, 1, 1), (x2, y, -1, 1), (x, y2, 1, -1), (x2, y2, -1, -1)):
        draw.line([(cx, cy), (cx + dx * seg, cy)], fill=color, width=width)
        draw.line([(cx, cy), (cx, cy + dy * seg)], fill=color, width=width)


def _overlap(a, b) -> int:
    """两个矩形 [x0,y0,x1,y1] 的重叠面积。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return 0 if x2 <= x1 or y2 <= y1 else (x2 - x1) * (y2 - y1)


def _place_panel(x, y, w, h, pw, ph, frame_w, frame_h, boxes, taken):
    """为信息牌挑选一块不压住目标、不与已放信息牌重叠的位置。

    候选顺序：框上方左对齐、框上方右对齐、框下方左对齐、框下方右对齐、
    框右侧、框左侧。逐个算代价（压住其它目标、压住已放信息牌、越界），
    取代价最小者；同代价时取靠前者，使排布保持稳定。
    """
    cands = [
        (x, y - ph - 8),
        (x + w - pw, y - ph - 8),
        (x, y + h + 8),
        (x + w - pw, y + h + 8),
        (x + w + 10, y),
        (x - pw - 10, y),
    ]
    best, best_cost = None, None
    for cx, cy in cands:
        ccx = int(min(max(2, cx), max(2, frame_w - pw - 2)))
        ccy = int(min(max(32, cy), max(32, frame_h - ph - 2)))
        rect = (ccx, ccy, ccx + pw, ccy + ph)
        cost = 0.0
        # 越界按像素计，代价最高
        cost += 40.0 * (abs(ccx - cx) + abs(ccy - cy)) / 10.0
        for b in boxes:
            cost += _overlap(rect, b) / float(pw * ph) * 6.0
        for r in taken:
            cost += _overlap(rect, r) / float(pw * ph) * 12.0
        if best_cost is None or cost < best_cost:
            best, best_cost = (ccx, ccy), cost
    return best


def annotate(frame: np.ndarray, dets: Sequence[Detection],
             elapsed_ms: Optional[float] = None, engine: str = "离线检测器") -> np.ndarray:
    """标注：四角刻线 + 半透明细框 + 编号信息牌 + 顶部状态条。

    中文用 PIL 绘制；无中文字体时退回 OpenCV 英文标注，不影响检测本身。
    """
    from PIL import Image, ImageDraw

    img = frame.copy()
    h_frame, w_frame = img.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, "RGBA")
    f_title = _font(17)
    f_text = _font(15)
    boxes = [(d.box[0], d.box[1], d.box[0] + d.box[2], d.box[1] + d.box[3]) for d in dets]
    taken: list = []

    for idx, d in enumerate(dets, start=1):
        x, y, w, h = d.box
        color = MATERIAL_COLOR.get(d.material, (0, 255, 0))
        # 半透明填充标出目标区域，细框给出精确边界，四角刻线在不遮住本体的情况下加粗边界
        draw.rectangle([x, y, x + w, y + h], fill=color + (28,), outline=color + (150,), width=1)
        _corner_ticks(draw, x, y, w, h, color + (255,), seg=int(min(22, w // 3, h // 3)))
        cx, cy = x + w // 2, y + h // 2
        draw.line([(cx - 7, cy), (cx + 7, cy)], fill=color + (220,), width=1)
        draw.line([(cx, cy - 7), (cx, cy + 7)], fill=color + (220,), width=1)

        if f_title is None:
            cv2.putText(img, "#%d %s %.2f %s" % (idx, d.label, d.conf, d.material),
                        (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            continue

        lines = [
            "#%d  %s" % (idx, d.label),
            "类别置信度 %.2f    材质 %s %.2f" % (d.conf, d.material, d.material_conf),
            "尺寸 %d×%d px    中心 (%d,%d)" % (w, h, cx, cy),
        ]
        widths = [draw.textbbox((0, 0), s, font=f_title if i == 0 else f_text)[2]
                  for i, s in enumerate(lines)]
        panel_w = max(widths) + 14
        panel_h = 19 * len(lines) + 10
        px, py = _place_panel(x, y, w, h, panel_w, panel_h,
                              w_frame, h_frame, boxes, taken)
        taken.append((px, py, px + panel_w, py + panel_h))
        draw.rectangle([px, py, px + panel_w, py + panel_h], fill=(16, 18, 22, 214))
        draw.rectangle([px, py, px + 3, py + panel_h], fill=color + (255,))
        # 引线：从信息牌朝向目标的一边连到框上最近的边，长度短且不跨过其它目标
        lx = min(max(px + panel_w // 2, x + 2), x + w - 2)
        if py + panel_h <= y:
            draw.line([(lx, py + panel_h), (lx, y)], fill=color + (200,), width=1)
        elif py >= y + h:
            draw.line([(lx, py), (lx, y + h)], fill=color + (200,), width=1)
        elif px + panel_w <= x:
            draw.line([(px + panel_w, cy), (x, cy)], fill=color + (200,), width=1)
        elif px >= x + w:
            draw.line([(px, cy), (x + w, cy)], fill=color + (200,), width=1)
        for i, s in enumerate(lines):
            draw.text((px + 8, py + 5 + 19 * i), s,
                      font=f_title if i == 0 else f_text,
                      fill=(255, 255, 255) if i == 0 else (206, 212, 220))

    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # 顶部状态条：目标数、材质分布、单帧耗时
    n_metal = sum(1 for d in dets if d.material == "金属")
    bar = "CyberLife 视觉识别　目标 %d　金属 %d　其他 %d" % (len(dets), n_metal, len(dets) - n_metal)
    if elapsed_ms is not None:
        bar += "　耗时 %.1f ms" % elapsed_ms
    bar += "　%s" % engine
    cv2.rectangle(img, (0, 0), (w_frame, 30), (18, 20, 24), -1)
    cv2.line(img, (0, 30), (w_frame, 30), (60, 200, 255), 1)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _font(16)
    if font is not None:
        draw.text((8, 6), bar, font=font, fill=(230, 236, 244))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(img, bar.encode("ascii", "ignore").decode() or "CyberLife Vision",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 236, 244), 1)
    return img


def analyze_image(path: str) -> Tuple[List[Detection], np.ndarray]:
    """读取一张图片并检测。中文路径用 imdecode 绕过 cv2.imread 的编码限制。"""
    frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise IOError("图片读取失败：%s" % path)
    t0 = time.perf_counter()
    dets = detect(frame)
    return dets, annotate(frame, dets, elapsed_ms=(time.perf_counter() - t0) * 1000.0)


def save_result(path: str, dets: Sequence[Detection], extra: Optional[dict] = None) -> None:
    payload = {"image": path, "summary": summarize(dets),
               "objects": [asdict(d) for d in dets]}
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
