# -*- coding: utf-8 -*-
"""可伸缩视频编码：基础层 + 增强层。

分层思路与 H.265/SVC 的空间可伸缩档一致：基础层只保留低分辨率画面，
单独就能解码，占用的带宽小；增强层是对基础层重建结果的残差，
基础层 + 增强层一起解码才得到全分辨率画面。网络变差时只发基础层，
画面模糊但连续；网络恢复后补发增强层，细节随之恢复。

实现方式
--------
本机没有 x265 / ffmpeg，因此不调用外部编码器，改为在 OpenCV 编码器
之上实现同一套分层结构：

1. 基础层：按 base_scale 降采样后用 JPEG 编码，质量 base_quality；
2. 增强层：把基础层解码、上采样回原尺寸，与原图的差做量化，
   量化值加 128 偏移后当作一幅图，再用 JPEG 编码；
3. 解码端把残差反量化后加回基础层重建，得到全分辨率画面。

分层的收益不在静态图的压缩率——平滑画面用单层 JPEG 已经很省字节，
而在带宽收缩时可以只发基础层，画面变模糊但不丢帧；带宽回升再补发
增强层。这一条在 demos/stream_demo.py 里以三档带宽实测。

近景细节保护
------------
可传入 roi=(x, y, w, h)，框内用细量化步长、框外用粗步长。
roi 与两个步长都写在增强层头部，解码端据此还原，不会错位。
"""
from __future__ import annotations

import struct
from typing import Optional, Tuple

import cv2
import numpy as np

ENH_HEADER = struct.Struct("<BB4H")        # 细步长, 粗步长, roi x, y, w, h
RESID_BIAS = 128


def _encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def _decode_jpeg(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("JPEG 解码失败")
    return frame


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """峰值信噪比，单位 dB；两图完全相同时返回 99.0。"""
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-9:
        return 99.0
    return float(10.0 * np.log10(255.0 * 255.0 / mse))


class H265SVC:
    """基础层 + 增强层的两层可伸缩编码。"""

    def __init__(self, base_scale: float = 0.5, base_quality: int = 55,
                 fine_step: int = 1, coarse_step: int = 3, enh_quality: int = 86,
                 base_kbps: int = 500, enhance_kbps: int = 2000):
        self.base_scale = float(base_scale)
        self.base_quality = int(base_quality)
        self.fine_step = max(1, int(fine_step))
        self.coarse_step = max(self.fine_step, int(coarse_step))
        self.enh_quality = int(enh_quality)
        self.base_bitrate = int(base_kbps)       # 仅用于带宽档位判断
        self.enhance_bitrate = int(enhance_kbps)

    # -------------------------------------------------- 编码
    def encode_base(self, frame: np.ndarray) -> bytes:
        """基础层：低分辨率、低码率，单独可解。"""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(16, int(w * self.base_scale)),
                                   max(16, int(h * self.base_scale))),
                           interpolation=cv2.INTER_AREA)
        return _encode_jpeg(small, self.base_quality)

    def _rebuild_from_base(self, base: bytes, size: Tuple[int, int]) -> np.ndarray:
        """把基础层解回全尺寸，作为残差预测。"""
        small = _decode_jpeg(base)
        return cv2.resize(small, size, interpolation=cv2.INTER_CUBIC)

    def encode_enhance(self, frame: np.ndarray, base: bytes,
                       roi: Optional[Tuple[int, int, int, int]] = None) -> bytes:
        """增强层：全分辨率残差。

        不传 roi 时编码整幅残差（粗步长），用于带宽宽裕时的完整画质；
        传 roi 时只编码框内残差（细步长），框外不再占字节。机器人关心的是
        近处的目标，全幅补细节往往不如只补目标区域划算。
        """
        h, w = frame.shape[:2]
        pred = self._rebuild_from_base(base, (w, h))
        resid = frame.astype(np.int16) - pred.astype(np.int16)

        x, y, rw, rh = 0, 0, 0, 0
        step = self.coarse_step
        if roi is not None:
            x, y, rw, rh = [int(v) for v in roi]
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            rw = max(0, min(w - x, rw))
            rh = max(0, min(h - y, rh))
            if rw > 0 and rh > 0:
                step = self.fine_step
                resid = resid[y:y + rh, x:x + rw]
            else:
                x, y, rw, rh = 0, 0, 0, 0

        quant = np.round(resid / step).clip(-127, 127).astype(np.int16)
        # 残差有正有负，加偏移搬进 0~255 才能交给 JPEG 编码
        payload = (quant + RESID_BIAS).clip(0, 255).astype(np.uint8)
        head = ENH_HEADER.pack(self.fine_step, self.coarse_step, x, y, rw, rh)
        return head + _encode_jpeg(payload, self.enh_quality)

    def encode(self, frame: np.ndarray,
               roi: Optional[Tuple[int, int, int, int]] = None) -> dict:
        """编码一帧，返回两层数据与元信息。"""
        base = self.encode_base(frame)
        enh = self.encode_enhance(frame, base, roi)
        h, w = frame.shape[:2]
        return {"base": base, "enhance": enh, "size": (w, h), "roi": roi}

    # -------------------------------------------------- 解码
    def decode_base(self, layers: dict) -> np.ndarray:
        """只解基础层：模糊但可看全貌。"""
        w, h = layers["size"]
        return cv2.resize(_decode_jpeg(layers["base"]), (w, h),
                          interpolation=cv2.INTER_CUBIC)

    def decode(self, layers: dict, with_enhance: bool = True) -> np.ndarray:
        """解出全分辨率画面；with_enhance=False 时等价于 decode_base。"""
        base = self.decode_base(layers)
        if not with_enhance or not layers.get("enhance"):
            return base
        fine, coarse, x, y, rw, rh = ENH_HEADER.unpack_from(layers["enhance"])
        payload = _decode_jpeg(layers["enhance"][ENH_HEADER.size:])
        h, w = base.shape[:2]
        quant = payload.astype(np.int16) - RESID_BIAS
        out = base.astype(np.int16)
        if rw > 0 and rh > 0:                      # 只编码了 ROI 内的残差
            if quant.shape[:2] != (rh, rw):
                raise RuntimeError("增强层尺寸与 ROI 不符：%s != %s"
                                   % (quant.shape[:2], (rh, rw)))
            out[y:y + rh, x:x + rw] += quant * fine
        else:                                      # 整幅残差
            if quant.shape[:2] != (h, w):
                raise RuntimeError("增强层尺寸与基础层不符：%s != %s"
                                   % (quant.shape[:2], (h, w)))
            out += quant * coarse
        return out.clip(0, 255).astype(np.uint8)

    # -------------------------------------------------- 档位与统计
    def get_layer_config(self, network_quality: float) -> str:
        """按网络质量选传输档位。network_quality 为 0~1，0 表示网极差。"""
        q = float(network_quality)
        if q < 0.30:
            return "base_only"
        if q < 0.70:
            return "base + enhance_low"
        return "base + enhance_full"

    def plan(self, tiers: dict, kbps: float, fps: int = 15) -> dict:
        """按链路带宽在档位阶梯上取最高可用档。

        tiers 为 {档位名: 单帧字节}，须按字节数升序排列。kbps 为可用带宽，
        fps 为目标帧率；返回预算内字节数最高的档位，全都不满足时退最低档。
        """
        budget_bits = max(1.0, float(kbps) * 1000.0 / max(1, fps))
        chosen = None
        for name, size in tiers.items():
            need = float(size) * 8.0
            if need <= budget_bits:
                chosen = {"layer": name, "frame_bits": int(need),
                          "max_fps": round(float(kbps) * 1000.0 / max(1.0, need), 1)}
        if chosen is None:
            name, size = next(iter(tiers.items()))
            need = float(size) * 8.0
            chosen = {"layer": name, "frame_bits": int(need),
                      "max_fps": round(float(kbps) * 1000.0 / max(1.0, need), 1),
                      "over_budget": True}
        return chosen

    def report(self, frame: np.ndarray,
               roi: Optional[Tuple[int, int, int, int]] = None) -> dict:
        """对一帧做完整的分层编码并给出可核对的指标。"""
        layers = self.encode(frame, roi)
        only_base = self.decode(layers, with_enhance=False)
        both = self.decode(layers, True)
        h, w = frame.shape[:2]
        single = _encode_jpeg(frame, self.base_quality)     # 同质量单层对照
        return {
            "size": "%dx%d" % (w, h),
            "base_bytes": len(layers["base"]),
            "enhance_bytes": len(layers["enhance"]),
            "total_bytes": len(layers["base"]) + len(layers["enhance"]),
            "single_layer_bytes": len(single),
            "psnr_base": round(psnr(frame, only_base), 2),
            "psnr_full": round(psnr(frame, both), 2),
            "psnr_single": round(psnr(frame, _decode_jpeg(single)), 2),
            "roi": roi,
            "layers": layers,
        }
