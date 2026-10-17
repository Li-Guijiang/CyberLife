# -*- coding: utf-8 -*-
"""CyberLife 演示程序菜单。

双击“运行演示.bat”即调用本文件；也可在本目录下执行 python 演示菜单.py。
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
PY = sys.executable

ITEMS = [
    ("1", "整机综合演示", "demo_all.py", [],
     "六个区域同屏：运动俯视图、姿态融合、目标检测、实时信号、稳定性监控、系统状态"),
    ("2", "八足四类步态演示", "gait_animation_demo.py", [],
     "三角步态、波浪步态、交替步态、旋转步态依次切换"),
    ("3", "重心与稳定裕度演示", "demos/gait_demo.py", [],
     "随机地形下重心轨迹与支撑多边形、稳定裕度曲线"),
    ("4", "四路 IMU 姿态融合演示", "demos/imu_demo.py", [],
     "四路惯性单元互补滤波与姿态角收敛过程"),
    ("5", "地磁金属探测演示", "demos/mag_demo.py", [],
     "校准、扫描、异常检测、偶极子反演四步，落盘 PNG 与 JSON"),
    ("6", "视觉实拍检测（需摄像头）", "vision_camera_demo.py", [],
     "打开本机摄像头，实时检测红色障碍物与绿色植被，按 q 退出"),
    ("7", "单元测试（12 项）", "-m", ["unittest", "tests.test_geomagnetic", "-v"],
     "地磁模块的正演、反演、标定、扫描全链路断言"),
    ("8", "视觉识别（离线，含金属判别）", "demos/vision_demo.py", ["--dir", "测试图"],
     "对 测试图/ 下 10 张图逐张识别，输出类别、材质与坐标，结果存到 识别结果/"),
    ("9", "视觉识别（摄像头实时）", "demos/vision_demo.py", ["--camera"],
     "打开摄像头实时识别，按 q 退出；每 25 帧打印一次统计"),
    ("10", "弱网传输（FEC + 分层编码）", "demos/stream_demo.py", [],
     "分组异或纠错按丢包率实测，分层编码给出三档码率与 PSNR"),
    ("11", "视觉模块网络自检", "demos/vision_demo.py", ["--diagnose"],
     "检查证书链与下载通道，不下载任何文件"),
]


def check_env():
    missing = []
    for mod, pkg in (("numpy", "numpy"), ("matplotlib", "matplotlib"), ("cv2", "opencv-python")):
        try:
            __import__(mod)
        except Exception:
            missing.append(pkg)
    return missing


def show(missing):
    print("=" * 68)
    print("CyberLife 具身智能八足仿生平台 · 演示程序")
    print("Python %d.%d.%d" % sys.version_info[:3])
    print("目录 %s" % ROOT)
    if missing:
        print("缺少依赖：%s" % "、".join(missing))
        print("请先双击“一键安装依赖.bat”，或执行 python -m pip install -r requirements.txt")
    print("=" * 68)
    for no, name, script, _, note in ITEMS:
        print("  %s. %-20s %s" % (no, name, note))
    print("  0. 退出")
    print("=" * 68)


def run(item):
    no, name, script, args, _ = item
    print("\n>>> 正在运行：%s\n" % name)
    if script == "-m":
        cmd = [PY, "-m"] + args
    else:
        cmd = [PY, str(ROOT / script)] + args
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=False)
    except KeyboardInterrupt:
        print("\n已中断")
    print("\n<<< 结束：%s\n" % name)


def main():
    missing = check_env()
    while True:
        show(missing)
        try:
            choice = input("请选择编号：").strip()
        except EOFError:
            return
        if choice in ("0", ""):
            return
        hit = [it for it in ITEMS if it[0] == choice]
        if not hit:
            print("编号不存在，请重新选择。\n")
            continue
        run(hit[0])


if __name__ == "__main__":
    main()
