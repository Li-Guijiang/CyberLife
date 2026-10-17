# 地磁金属探测演示
# 走一遍完整流程：转圈校准 -> 弓字形扫描 -> 找异常 -> 估深度 -> 出图
# 数据是仿真的，因为本机没接磁力计实物，图上会写明
import sys
import json
import pathlib
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 让这个文件在 demos 目录里直接跑也能找到 src
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geomagnetic_system.calibration import EllipsoidCalibration, simulate_rotation
from src.geomagnetic_system.magnetometer import SimulatedMagnetometer
from src.geomagnetic_system.scanner import AreaScanner, grid_anomaly, plan_lawnmower

# 中文显示，Windows 上一般有微软雅黑
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

AREA_W, AREA_H = 3.0, 2.0     # 作业区域，m
LANE_SPACING = 0.20           # 测线间距
STEP = 0.10                   # 采样点间距
SENSOR_H = 0.15               # 传感器离地高度

# 埋在地下的目标，moment 是磁矩，单位 A·m^2
# 三个目标分开放，避免小目标被强目标的大范围异常盖掉
TARGETS = [
    {"name": "小型金属件（模拟弹片）", "pos": (0.70, 0.50, 0.45), "moment": (0.0, 0.0, 0.35)},
    {"name": "中型金属件（模拟器材）", "pos": (2.20, 1.45, 0.70), "moment": (0.0, 0.0, 4.0)},
    {"name": "微金属片（模拟弹壳）", "pos": (0.55, 1.60, 0.30), "moment": (0.0, 0.0, 0.10)},
]


def build_scene():
    """造一个带硬磁软磁误差的仿真场景，误差设得比实物大一点，好看出校准效果"""
    hard = np.array([320.0, -180.0, 450.0])
    soft = np.array([
        [1.06, 0.02, -0.01],
        [0.02, 0.95, 0.015],
        [-0.01, 0.015, 0.99],
    ])
    mag = SimulatedMagnetometer(
        height=SENSOR_H, targets=TARGETS, noise_nt=12.0,
        hard_iron=hard, soft_iron=soft, seed=7,
    )
    return mag, hard, soft


def draw(fig, data, raw_norms, cal_norms, path, truth):
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # 图1：校准前后，看 Bx-By 平面
    ax1.plot(raw_norms["x"], raw_norms["y"], ".", ms=2.5, color="#b0b7c3",
             label="校准前 σ=%.0f nT" % raw_norms["std"])
    ax1.plot(cal_norms["x"], cal_norms["y"], ".", ms=2.5, color="#1d4ed8",
             label="校准后 σ=%.0f nT" % cal_norms["std"])
    ax1.set_title("磁力计校准（Bx-By 平面）", fontsize=10)
    ax1.set_xlabel("Bx (nT)")
    ax1.set_ylabel("By (nT)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.25)

    # 图2：异常热图 + 扫描路径 + 目标
    gx, gy, gz = grid_anomaly(data["points"], data["deltas"], resolution=101)
    im = ax2.pcolormesh(gx, gy, gz, shading="auto", cmap="inferno")
    ax2.plot(path[:, 0], path[:, 1], "-", color="#7dd3fc", lw=0.6, alpha=0.55,
             label="扫描路径")
    for t in truth:
        ax2.plot(t["pos"][0], t["pos"][1], "x", color="#22d3ee", ms=8, mew=1.8)
    for t in data["targets"]:
        ax2.plot(t["x"], t["y"], "o", mfc="none", mec="#ffffff", ms=12, mew=1.2)
        ax2.annotate("%s\n%.0f nT" % (t["id"], t["peak_nt"]), (t["x"], t["y"]),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=7, color="white")
    ax2.set_title("磁异常热图（青色×为真值，白圈为检测）", fontsize=10)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_xlim(0, AREA_W)
    ax2.set_ylim(0, AREA_H)
    ax2.legend(fontsize=7, loc="lower left")
    fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02, label="ΔT (nT)")

    # 图3：过最近目标的测线剖面
    t0 = data["targets"][0] if data["targets"] else None
    if t0:
        # 找出离该目标最近的那条测线
        ys = np.unique(np.round(data["points"][:, 1], 3))
        lane_y = ys[np.argmin(np.abs(ys - t0["y"]))]
        m = np.isclose(data["points"][:, 1], lane_y)
        xs = data["points"][m, 0]
        ds = data["deltas"][m]
        ax3.plot(xs, ds, "-", color="#f59e0b", lw=1.2, label="实测 ΔT")
        thr = data["info"]["threshold_nt"]
        ax3.axhline(thr, ls="--", lw=0.9, color="#ef4444", label="阈值 %.0f nT" % thr)
        ax3.axhline(-thr, ls="--", lw=0.9, color="#ef4444")
        ax3.axvline(t0["x"], ls=":", lw=0.8, color="#22d3ee", label="目标真值 x")
        ax3.set_title("测线 y=%.2f m 的异常剖面" % lane_y, fontsize=10)
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("ΔT (nT)")
        ax3.legend(fontsize=7)
        ax3.grid(alpha=0.25)

    # 图4：目标清单，用真表格排版，别用等宽字体凑对齐
    ax4.axis("off")
    ax4.set_title("检测结果（共 %d 个目标）" % len(data["targets"]), fontsize=10)
    if data["targets"]:
        head = ["编号", "X (m)", "Y (m)", "峰值 nT", "深度 m", "磁矩 A·m²"]
        rows = [head] + [[
            t["id"], "%.2f" % t["x"], "%.2f" % t["y"], "%.0f" % t["peak_nt"],
            ("%.2f" % t["depth_m"]) if t["depth_m"] is not None else "-",
            ("%.3f" % t["moment_am2"]) if t["moment_am2"] is not None else "-",
        ] for t in data["targets"]]
        tb = ax4.table(cellText=rows, cellLoc="center", loc="upper center",
                       colWidths=[0.12, 0.13, 0.13, 0.17, 0.14, 0.19])
        tb.auto_set_font_size(False)
        tb.set_fontsize(8.5)
        tb.scale(1, 1.5)
        for j in range(len(head)):
            tb[0, j].set_facecolor("#eef2ff")
            tb[0, j].set_text_props(weight="bold")
        lines = []
        for t in data["targets"]:
            lines.append("%s  %s" % (t["id"], t["category"]))
        lines.append("")
        lines.append("噪声 σ=%.1f nT　阈值=%.1f nT　测点 %d 个"
                     % (data["info"]["sigma_nt"], data["info"]["threshold_nt"],
                        data["info"]["points_total"]))
        lines.append("真值目标 %d 个，检出 %d 个" % (len(truth), len(data["targets"])))
        ax4.text(0.02, 0.30, "\n".join(lines), va="top", ha="left",
                 fontsize=9, transform=ax4.transAxes, linespacing=1.6)
    else:
        ax4.text(0.5, 0.5, "未检出超过阈值的磁异常", ha="center", va="center",
                 fontsize=11, transform=ax4.transAxes)


def main():
    ap = argparse.ArgumentParser(description="地磁金属探测演示")
    ap.add_argument("--out", help="输出 PNG 路径，不给就弹窗显示")
    ap.add_argument("--json", help="检测结果另存 JSON 的路径")
    args = ap.parse_args()

    mag, hard, soft = build_scene()

    # 第一步：转一圈校准
    raw = simulate_rotation(mag, n=360)
    cal = EllipsoidCalibration().fit(raw)
    calibrated = cal.apply(raw)
    q = cal.quality()
    print("校准结果：")
    print("  拟合偏移 %s nT" % np.round(cal.offset, 1))
    print("  真实偏移 %s nT" % np.round(hard, 1))
    print("  场强 %.1f nT，残差 %.1f nT（占 %.2f%%），判定 %s"
          % (q["field_strength_nt"], q["residual_nt"],
             q["residual_ratio"] * 100, q["level"]))

    raw_norms = {
        "x": raw[:, 0], "y": raw[:, 1],
        "std": float(np.std(np.linalg.norm(raw, axis=1))),
    }
    cal_norms = {
        "x": calibrated[:, 0], "y": calibrated[:, 1],
        "std": float(np.std(np.linalg.norm(calibrated, axis=1))),
    }

    # 第二步：弓字形扫描
    path = plan_lawnmower(AREA_W, AREA_H, LANE_SPACING, STEP)
    scanner = AreaScanner(mag, sensor_height=SENSOR_H, calibration=cal)
    data = scanner.scan_area(AREA_W, AREA_H, LANE_SPACING, STEP)
    print("\n扫描完成：%d 个测点，异常超阈值 %d 个"
          % (data["info"]["points_total"], data["info"]["points_above"]))

    print("\n检测结果：")
    for t in data["targets"]:
        print("  %s  (%.2f, %.2f)  峰值 %.0f nT  深度 %s m  磁矩 %s  %s"
              % (t["id"], t["x"], t["y"], t["peak_nt"],
                 t["depth_m"], t["moment_am2"], t["category"]))

    # 和真值对账
    print("\n与真值对比：")
    for truth in TARGETS:
        tx, ty, tz = truth["pos"]
        if data["targets"]:
            d = [np.hypot(t["x"] - tx, t["y"] - ty) for t in data["targets"]]
            k = int(np.argmin(d))
            hit = data["targets"][k]
            print("  %s 埋深 %.2f m -> 最近检测 %s，平面偏差 %.2f m，深度估计 %s m"
                  % (truth["name"], tz, hit["id"], d[k], hit["depth_m"]))
        else:
            print("  %s 未检出" % truth["name"])

    fig = plt.figure(figsize=(13, 8.5))
    fig.suptitle("CyberLife 地磁金属探测演示（仿真数据：本机未接磁力计实物）",
                 fontsize=12)
    draw(fig, data, raw_norms, cal_norms, path, TARGETS)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if args.json:
        payload = {
            "mode": "simulation",
            "note": "仿真数据，本机未接磁力计实物",
            "area_m": [AREA_W, AREA_H],
            "lane_spacing_m": LANE_SPACING,
            "step_m": STEP,
            "sensor_height_m": SENSOR_H,
            "calibration": q,
            "scan": data["info"],
            "targets_detected": data["targets"],
            "targets_truth": [
                {"name": t["name"], "pos": list(t["pos"]), "moment": list(t["moment"])}
                for t in TARGETS
            ],
        }
        pathlib.Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nJSON 已存：%s" % args.json)

    if args.out:
        fig.savefig(args.out, dpi=130, facecolor=fig.get_facecolor())
        print("图像已存：%s" % args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
