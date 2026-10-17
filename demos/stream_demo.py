# -*- coding: utf-8 -*-
"""弱网传输演示：FEC 分组纠错与分层可伸缩编码。

两件事都做实测，不打空头结论：

1. FEC：把一帧 JPEG 数据切成定长包，按组加异或校验包，按比例随机丢包，
   看接收端能否逐字节还原，以及丢两个包同组时如何判废。
2. 分层编码：同一帧编出基础层与增强层，量出各自的字节数与 PSNR，
   再按链路带宽给出该发哪一层、能跑多少帧。

用法
----
python demos/stream_demo.py
python demos/stream_demo.py --image 测试图/S07_多目标混合.png
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.vision_system import fec_transmission as fec  # noqa: E402
from src.vision_system import h265_svc as svc  # noqa: E402
from src.vision_system import object_detect as od  # noqa: E402

OUT_DIR = ROOT / "运行结果示例"
DEFAULT_IMAGE = ROOT / "测试图" / "S07_多目标混合.png"


def load_frame(path: pathlib.Path, size=(640, 480)) -> np.ndarray:
    frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        # 没有测试图时用背景+几何体合成一帧，保证演示可跑
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        frame[:, :] = (96, 92, 88)
        cv2.rectangle(frame, (120, 120), (320, 360), (200, 200, 205), -1)
        cv2.circle(frame, (470, 240), 110, (40, 190, 240), -1)
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- FEC
def run_fec(frame: np.ndarray, group_size: int, seed: int = 20261017) -> list:
    """按固定丢包组合逐项验证，便于复核；另加一组随机 25% 丢包。"""
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    data = buf.tobytes()
    enc = fec.FECEncoder(group_size=group_size, max_payload=1024)
    packets = enc.encode_frame(1, data)
    n = group_size
    rng = random.Random(seed)
    cases = [
        ("无丢包", set()),
        ("组 0 丢 1 个数据包", {0}),
        ("组 0 与组 1 各丢 1 个", {0, n + 1}),
        ("组 0 丢 2 个数据包", {0, 1}),
        ("末组校验包丢失", {len(packets) - 1}),
        ("随机 25% 丢包", {i for i in range(len(packets)) if rng.random() < 0.25}),
    ]
    rows = []
    for name, drop in cases:
        rx = fec.FECReceiver(group_size=group_size)
        got = None
        for p in fec.simulate_loss(packets, drop):
            r = rx.feed(p)
            if r:
                got = r
        st = rx.finish()                       # 播放截止，剩下没凑齐的帧判废
        rows.append({
            "场景": name,
            "发送包数": len(packets),
            "丢弃包数": len(drop),
            "纠回包数": st["recovered"],
            "还原结果": "逐字节一致" if got == data else "未还原",
            "成帧": st["frames_done"],
            "弃帧": st["frames_dropped"],
        })
    return rows, len(data), enc.overhead(data)


# ---------------------------------------------------------------- SVC
def draw_compare(orig, base, roi_img, roi, values, path: pathlib.Path) -> None:
    """ROI 放大对照条：左为基础层，右为补增强层，标题带实测 PSNR。

    整帧两版差别很小，放大到四倍才看得出块状台阶；因此只出 ROI 条，
    条幅宽而矮，放进说明书占版面少。
    """
    from PIL import Image, ImageDraw
    font = od._font(24)
    font_s = od._font(20)
    x0, y0, rw, rh = roi
    zoom = 4
    crop_w, crop_h = rw * zoom, rh * zoom
    head_h = 62
    canvas = Image.new("RGB", (2 * (crop_w + 8), head_h + crop_h + 6), (24, 26, 30))
    draw = ImageDraw.Draw(canvas)

    cols = [("基础层单独", "ROI 峰值信噪比 %.2f dB" % values["base_roi"], base),
            ("基础层 + ROI 增强层", "ROI 峰值信噪比 %.2f dB" % values["roi_roi"], roi_img)]
    for i, (title, sub, img) in enumerate(cols):
        px = i * (crop_w + 8)
        if font is not None:
            draw.text((px + 8, 6), title, font=font, fill=(236, 242, 250))
        if font_s is not None:
            draw.text((px + 8, 36), sub, font=font_s, fill=(150, 210, 255))
        crop = cv2.resize(img[y0:y0 + rh, x0:x0 + rw], (crop_w, crop_h),
                          interpolation=cv2.INTER_NEAREST)
        canvas.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (px, head_h))
        draw.rectangle([px, head_h, px + crop_w - 1, head_h + crop_h - 1],
                       outline=(90, 94, 100), width=1)
    canvas.save(str(path))


def run_svc(frame: np.ndarray) -> dict:
    """三档对比：只发基础层、基础层+ROI 增强层、基础层+全幅增强层，另加单层对照。"""
    svcoder = svc.H265SVC()
    roi = (200, 90, 240, 160)
    full_layers = svcoder.encode(frame, None)
    roi_layers = svcoder.encode(frame, roi)
    base_img = svcoder.decode(full_layers, with_enhance=False)
    full_img = svcoder.decode(full_layers, True)
    roi_img = svcoder.decode(roi_layers, True)

    x0, y0, rw, rh = roi

    def roi_psnr(img):
        return round(svc.psnr(frame[y0:y0 + rh, x0:x0 + rw], img[y0:y0 + rh, x0:x0 + rw]), 2)

    cases = [
        ("基础层单独", len(full_layers["base"]), len(full_layers["base"]), 0,
         base_img, roi_psnr(base_img)),
        ("基础层+ROI 增强层", len(full_layers["base"]) + len(roi_layers["enhance"]),
         len(full_layers["base"]), len(roi_layers["enhance"]), roi_img, roi_psnr(roi_img)),
        ("基础层+全幅增强层", len(full_layers["base"]) + len(full_layers["enhance"]),
         len(full_layers["base"]), len(full_layers["enhance"]), full_img, roi_psnr(full_img)),
    ]
    rows = []
    for name, total, base_b, enh_b, img, rps in cases:
        rows.append({"档位": name, "字节": total, "基础层字节": base_b, "增强层字节": enh_b,
                     "全幅PSNR": round(svc.psnr(frame, img), 2), "ROI内PSNR": rps or 0.0,
                     "roi": roi if "ROI" in name or "全幅" in name else None})

    # 同质量单层编码对照：一帧整幅，没有分层
    single_bytes = cv2.imencode(".jpg", frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), svcoder.base_quality])[1]
    single = cv2.imdecode(np.frombuffer(single_bytes, np.uint8), cv2.IMREAD_COLOR)
    rows.append({"档位": "单层整帧对照", "字节": int(len(single_bytes)), "基础层字节": 0,
                 "增强层字节": 0, "全幅PSNR": round(svc.psnr(frame, single), 2),
                 "ROI内PSNR": roi_psnr(single), "roi": roi})
    return {"rows": rows, "svcoder": svcoder, "full_layers": full_layers,
            "roi_layers": roi_layers, "base_img": base_img, "full_img": full_img,
            "roi_img": roi_img, "roi": roi}


def main() -> int:
    ap = argparse.ArgumentParser(description="CyberLife 弱网传输演示（FEC + 分层编码）")
    ap.add_argument("--image", default=str(DEFAULT_IMAGE), help="测试图路径")
    ap.add_argument("--out", default=str(OUT_DIR), help="结果输出目录")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_frame(pathlib.Path(args.image))
    h, w = frame.shape[:2]
    print("=" * 78)
    print("CyberLife 弱网传输演示    测试帧 %dx%d    来源 %s" % (w, h, pathlib.Path(args.image).name))
    print("=" * 78)

    t0 = time.perf_counter()
    report = {"frame": "%dx%d" % (w, h), "fec": [], "svc": []}

    print("\n一、FEC 分组前向纠错（每 1024 字节切一个包）")
    for group in (4, 8):
        rows, size, overhead = run_fec(frame, group)
        print("\n  每 %d 个数据包加 1 个校验包；一帧 %d 字节，头部与校验开销 %.1f%%"
              % (group, size, overhead * 100))
        print("  %-24s %-8s %-8s %-8s %-12s %-6s %-6s" % (
            "场景", "发包", "丢包", "纠回", "还原结果", "成帧", "弃帧"))
        for r in rows:
            print("  %-24s %-8d %-8d %-8d %-12s %-6d %-6d" % (
                r["场景"], r["发送包数"], r["丢弃包数"], r["纠回包数"],
                r["还原结果"], r["成帧"], r["弃帧"]))
        report["fec"].append({"group_size": group, "overhead": round(overhead, 4), "rows": rows})

    print("\n二、分层可伸缩编码（基础层降采样 %d%%，增强层补残差）"
          % int(100 * svc.H265SVC().base_scale))
    svc_result = run_svc(frame)
    rows = svc_result["rows"]
    print("  %-18s %-8s %-10s %-10s %-10s" % ("档位", "字节", "全幅PSNR", "ROI内PSNR", "ROI位置"))
    for r in rows:
        print("  %-18s %-8d %-10.2f %-10.2f %s" % (
            r["档位"], r["字节"], r["全幅PSNR"], r["ROI内PSNR"], r["roi"] or "-"))
        report["svc"].append({k: v for k, v in r.items() if k != "layers"})

    layers = svc_result["full_layers"]
    roi_layers = svc_result["roi_layers"]
    svcoder = svc_result["svcoder"]
    base_n = len(layers["base"])
    tiers = {
        "base_only": base_n,
        "base + enhance_roi": base_n + len(roi_layers["enhance"]),
        "base + enhance_full": base_n + len(layers["enhance"]),
    }
    print("\n三、按链路带宽在档位阶梯上取最高可用档（目标 15 帧/秒）")
    print("  %-22s %-10s %s" % ("档位", "单帧字节", "维持 15fps 所需带宽"))
    for name, size in tiers.items():
        need = size * 8 * 15 / 1000.0
        print("  %-22s %-10d %.0f kbps" % (name, size, need))
    print("  %-12s %-22s %-12s %s" % ("实测带宽", "发送档位", "单帧字节", "可达帧率"))
    for kbps in (200, 400, 800, 2000, 5000):
        plan = svcoder.plan(tiers, kbps, 15)
        print("  %-12s %-22s %-12d %.1f fps" % (
            "%d kbps" % kbps, plan["layer"], plan["frame_bits"] // 8, plan["max_fps"]))
        report["plan_%d" % kbps] = plan

    png = out_dir / "06_stream_svc_compare.png"
    values = {"base": rows[0]["全幅PSNR"], "base_roi": rows[0]["ROI内PSNR"],
              "roi": rows[1]["全幅PSNR"], "roi_roi": rows[1]["ROI内PSNR"]}
    draw_compare(frame, svc_result["base_img"], svc_result["roi_img"],
                 svc_result["roi"], values, png)
    print("\n对照图已保存：%s" % png)

    report["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    js = out_dir / "06_stream_demo.json"
    js.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("指标已保存：%s" % js)
    print("总耗时 %.1f ms" % report["elapsed_ms"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
