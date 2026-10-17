# -*- coding: utf-8 -*-
"""CyberLife 视觉识别演示（离线版）。

与旧版的区别
------------
旧版在启动时调用 YOLO("yolov8n.pt")，该调用会把权重名解析成 GitHub 上的
下载地址，一旦下载失败（证书链、网络不通、限流）整机就只能显示一行红字，
识别功能完全不可用。本版把权重下载从运行链路里去掉：

1. 默认使用 src/vision_system/object_detect.py 的离线检测器，只用
   OpenCV 与 numpy，不联网、不用权重，启动即可出结果；
2. 若本目录已放置 models/yolov8n.pt 或 yolov8n.pt，加 --engine yolo 可优先用
   YOLO 推理（完全本地加载，仍然不联网）；
3. --diagnose 用于排查证书与网络问题，只做握手自检，不下载任何文件。

用法
----
python demos/vision_demo.py                          # 摄像头实时（默认）
python demos/vision_demo.py --image a.jpg            # 单张图片
python demos/vision_demo.py --dir 测试图              # 批量图片
python demos/vision_demo.py --image a.jpg --engine yolo
python demos/vision_demo.py --diagnose               # 证书与网络自检
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.vision_system import object_detect as od  # noqa: E402

WEIGHT_CANDIDATES = (ROOT / "models" / "yolov8n.pt", ROOT / "yolov8n.pt")
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# ---------------------------------------------------------------- 证书自检
def diagnose() -> int:
    """只做 TLS 握手与依赖检查，不下载任何文件。"""
    import socket
    import ssl

    print("=" * 66)
    print("视觉模块网络自检（不下载任何文件）")
    print("=" * 66)

    try:
        import certifi
        ca = certifi.where()
        print("certifi 证书包：%s" % ca)
    except Exception:
        ca = None
        print("certifi：未安装")

    host = "github.com"
    for name, cafile in (("certifi 证书包", ca), ("Windows 系统证书库", None)):
        try:
            ctx = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
            with socket.create_connection((host, 443), timeout=6) as s:
                with ctx.wrap_socket(s, server_hostname=host):
                    print("%-16s 与 %s 握手成功" % (name, host))
        except Exception as e:
            print("%-16s 与 %s 失败：%s" % (name, host, str(e)[:90]))

    try:
        import requests
        requests.get("https://github.com", timeout=6)
        print("requests 下载通道：可用")
    except Exception as e:
        print("requests 下载通道：不可用（%s）" % type(e).__name__)

    print("-" * 66)
    print("若出现“certifi 失败、系统证书库成功”，说明本机有代理在进行 TLS 解密，")
    print("它签发的根证书已装入 Windows 证书库，但不在 certifi 包里，而")
    print("requests / ultralytics 默认只认 certifi，于是下载必然报证书错误。")
    print("三种处理办法：")
    print("  1) 不联网：把 yolov8n.pt 手动放到本目录 models/ 下，用 --engine yolo；")
    print("  2) pip install truststore，再在代码里 truststore.inject_into_ssl()；")
    print("  3) 把系统根证书导出为 PEM，设置环境变量 REQUESTS_CA_BUNDLE 指向它。")
    print("=" * 66)
    return 0


# ---------------------------------------------------------------- 引擎
def load_yolo():
    """只在本地已有权重时加载，绝不触发下载。"""
    for p in WEIGHT_CANDIDATES:
        if p.exists():
            try:
                from ultralytics import YOLO
            except Exception as e:
                print("已找到权重 %s，但 ultralytics 不可用：%s" % (p, e))
                return None
            print("使用本地权重：%s" % p)
            return YOLO(str(p))
    print("未找到本地权重 models/yolov8n.pt，改用离线检测器（不联网）。")
    return None


def yolo_detect(model, frame):
    """把 YOLO 结果转成统一的 Detection 结构，材质一栏用颜色证据补齐。"""
    names = model.names
    dets = []
    for r in model(frame, verbose=False):
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = str(names.get(cls_id, cls_id))
            if name in ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"):
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            roi = frame[max(0, y1):y2, max(0, x1):x2]
            material, mconf = "未知", 0.5
            if roi.size:
                mask = np.full(roi.shape[:2], 255, np.uint8)
                material, mconf = od.classify_material(od.material_features(roi, mask))
            dets.append(od.Detection(
                label=name, conf=float(box.conf[0]),
                box=(x1, y1, x2 - x1, y2 - y1),
                material=material, material_conf=float(mconf), features={}))
    return dets


# ---------------------------------------------------------------- 输出
def print_result(tag: str, dets, cost_ms: float) -> None:
    print("\n%s  检出 %d 个目标，耗时 %.1f ms" % (tag, len(dets), cost_ms))
    if not dets:
        print("  未检出目标")
        return
    print("  %-3s %-14s %-6s %-8s %-8s %s" % ("序号", "类别", "置信度", "材质", "材质置信度", "位置(x,y,w,h)"))
    for i, d in enumerate(dets, 1):
        x, y, w, h = d.box
        print("  %-3d %-14s %-6.2f %-8s %-8.2f (%d,%d,%d,%d)"
              % (i, d.label, d.conf, d.material, d.material_conf, x, y, w, h))
    s = od.summarize(dets)
    print("  类别统计：%s" % "，".join("%s %d" % kv for kv in s["by_label"].items()))
    print("  材质统计：%s" % "，".join("%s %d" % kv for kv in s["by_material"].items()))


def run_image(path: pathlib.Path, engine: str, model, save_dir: pathlib.Path) -> dict:
    frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        print("读取失败：%s" % path)
        return {}
    t0 = time.perf_counter()
    dets = yolo_detect(model, frame) if model is not None else od.detect(frame)
    cost = (time.perf_counter() - t0) * 1000.0
    h, w = frame.shape[:2]
    print_result("%s  %dx%d" % (path.name, w, h), dets, cost)
    vis = od.annotate(frame, dets)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / (path.stem + "_detected.png")
    cv2.imencode(".png", vis)[1].tofile(str(out_png))
    record = {"image": path.name, "engine": engine, "cost_ms": round(cost, 2),
              "summary": od.summarize(dets),
              "objects": [{"label": d.label, "conf": round(d.conf, 3), "box": list(d.box),
                           "material": d.material,
                           "material_conf": round(d.material_conf, 3)} for d in dets]}
    (save_dir / (path.stem + "_detected.json")).write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  已保存 %s" % out_png.name)
    return record


def run_camera(engine: str, model, save: bool, save_dir: pathlib.Path) -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头打不开：被其他程序占用，或本机没有摄像头。")
        print("可改用图片模式：python demos/vision_demo.py --image 图片路径")
        return 2
    print("摄像头已打开，按 q 退出；每 25 帧打印一次统计。")
    frames = 0
    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        dets = yolo_detect(model, frame) if model is not None else od.detect(frame)
        cost = (time.perf_counter() - t0) * 1000.0
        vis = od.annotate(frame, dets)
        cv2.imshow("CyberLife Vision (offline)", vis)
        frames += 1
        if frames % 25 == 0:
            s = od.summarize(dets)
            fps = frames / max(1e-6, time.perf_counter() - t_start)
            print("第 %d 帧  目标 %d 个  单帧 %.1f ms  平均 %.1f fps  材质 %s"
                  % (frames, len(dets), cost, fps,
                     "，".join("%s%d" % kv for kv in s["by_material"].items())))
        if save and frames % 50 == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".png", vis)[1].tofile(str(save_dir / ("camera_%04d.png" % frames)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("共处理 %d 帧，结束。" % frames)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="CyberLife 视觉识别演示（离线，含金属材质判别）")
    ap.add_argument("--image", help="单张图片路径")
    ap.add_argument("--dir", help="图片目录，批量处理")
    ap.add_argument("--camera", action="store_true", help="摄像头实时模式（默认）")
    ap.add_argument("--engine", choices=("auto", "yolo", "offline"), default="auto",
                    help="auto：有本地权重用 YOLO，否则离线检测器")
    ap.add_argument("--out", default="识别结果", help="结果输出目录")
    ap.add_argument("--save", action="store_true", help="摄像头模式下每 50 帧存一张图")
    ap.add_argument("--diagnose", action="store_true", help="只做证书与网络自检")
    args = ap.parse_args()

    if args.diagnose:
        return diagnose()

    model = None
    engine = "offline"
    if args.engine in ("auto", "yolo"):
        model = load_yolo()
        engine = "yolo" if model is not None else "offline"

    save_dir = pathlib.Path(args.out)
    if args.dir:
        files = sorted(p for p in pathlib.Path(args.dir).iterdir() if p.suffix.lower() in IMG_EXT)
        if not files:
            print("目录里没有图片：%s" % args.dir)
            return 1
        records = [run_image(p, engine, model, save_dir) for p in files]
        (save_dir / "batch_summary.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        total = sum(r.get("summary", {}).get("total", 0) for r in records if r)
        print("\n批量完成：%d 张图片，共检出 %d 个目标，结果在 %s" % (len(files), total, save_dir))
        return 0

    if args.image:
        run_image(pathlib.Path(args.image), engine, model, save_dir)
        return 0

    return run_camera(engine, model, args.save, save_dir)


if __name__ == "__main__":
    sys.exit(main())
