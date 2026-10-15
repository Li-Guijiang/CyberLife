# CyberLife 统一演示入口
# 运行这个脚本就能展示所有三大系统的效果
# 老师下载源码后，pip install -r requirements.txt 然后 python demo_all.py 就行
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.gait_system.wave_gait import WaveGait
from src.gait_system.triangle_gait import TriangleGait
from src.gait_system.alternating_gait import AlternatingGait
from src.gait_system.rotation_gait import RotationGait
from src.bci_imu_system.imu_fusion import Madgwick

print("="*60)
print("CyberLife 八足仿生机器人系统演示")
print("="*60)
print("演示内容：")
print("  1. 行静仿生稳定系统：四种步态动画")
print("  2. 脑电方向控制系统：IMU姿态融合实时曲线")
print("  3. 全景视觉系统：目标检测+FEC传输")
print("="*60)
print("正在启动演示窗口...")

# ========== 窗口1：步态动画 ==========
fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
title1 = ax1.set_title("", fontsize=12)

body = plt.Circle((0, 0), 0.3, color='gray', alpha=0.5)
ax1.add_patch(body)
leg_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
legs = []
feet = []
for i in range(8):
    line, = ax1.plot([], [], 'b-', linewidth=3)
    legs.append(line)
    pt, = ax1.plot([], [], 'ro', markersize=8)
    feet.append(pt)

gaits = [
    (WaveGait(freq=1.0), "波浪步态 (Wave) - 最稳，适合不平路面"),
    (TriangleGait(freq=1.2), "三角步态 (Triangle) - 静态稳定"),
    (AlternatingGait(freq=1.5), "交替步态 (Alternating) - 速度中等"),
    (RotationGait(freq=1.2), "旋转步态 (Rotation) - 原地转向")
]
gait_idx = 0

def update_gait(frame):
    global gait_idx
    gait, name = gaits[gait_idx]
    angles = gait.step(0.05)
    for i in range(8):
        bx = 0.3 * np.cos(leg_angles[i])
        by = 0.3 * np.sin(leg_angles[i])
        fx = bx + 0.4 * np.cos(leg_angles[i] + angles[i, 0])
        fy = by + 0.4 * np.sin(leg_angles[i] + angles[i, 0])
        legs[i].set_data([bx, fx], [by, fy])
        feet[i].set_data([fx], [fy])
    title1.set_text(f"【步态系统】{name}")
    if frame % 200 == 199:
        gait_idx = (gait_idx + 1) % 4
    return legs + feet + [title1]

ani1 = animation.FuncAnimation(fig1, update_gait, interval=50, blit=False)

# ========== 窗口2：IMU姿态曲线 ==========
fig2, (ax2_1, ax2_2, ax2_3) = plt.subplots(3, 1, figsize=(10, 6))
fig2.suptitle("【方向控制系统】IMU姿态融合实时曲线")
imu = Madgwick(dt=0.05)
roll_data, pitch_data, yaw_data = [], [], []
x_data = []

line_roll, = ax2_1.plot([], [], 'r-', label='Roll 横滚')
line_pitch, = ax2_2.plot([], [], 'g-', label='Pitch 俯仰')
line_yaw, = ax2_3.plot([], [], 'b-', label='Yaw 偏航')
for ax in [ax2_1, ax2_2, ax2_3]:
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)

def update_imu(frame):
    gyro = np.array([0.2*np.sin(frame*0.1), 0.1*np.cos(frame*0.08), 0.05*np.sin(frame*0.05)])
    accel = np.array([0.0, 0.0, 9.8])
    imu.update(gyro, accel)
    roll, pitch, yaw = imu.get_euler()
    
    x_data.append(frame)
    roll_data.append(roll)
    pitch_data.append(pitch)
    yaw_data.append(yaw)
    
    if len(x_data) > 100:
        x_data.pop(0)
        roll_data.pop(0)
        pitch_data.pop(0)
        yaw_data.pop(0)
    
    line_roll.set_data(x_data, roll_data)
    line_pitch.set_data(x_data, pitch_data)
    line_yaw.set_data(x_data, yaw_data)
    for ax in [ax2_1, ax2_2, ax2_3]:
        ax.set_xlim(max(0, frame-100), frame+10)
    return line_roll, line_pitch, line_yaw

ani2 = animation.FuncAnimation(fig2, update_imu, interval=50, blit=False)

# ========== 窗口3：视觉系统演示 ==========
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.set_title("【视觉系统】YOLO检测 + H.265编码 + FEC传输")
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.grid(True, alpha=0.3)

# 模拟检测框
boxes = [
    (2, 7, "障碍物", 'red'),
    (5, 5, "物品", 'green'),
    (7, 3, "植被", 'blue'),
]
def update_vision(frame):
    ax3.clear()
    ax3.set_title("【视觉系统】YOLO检测 + H.265编码 + FEC传输")
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.grid(True, alpha=0.3)
    
    # 画三个模拟检测框
    for x, y, name, color in boxes:
        offset = 0.2 * np.sin(frame*0.1)
        rect = plt.Rectangle((x+offset, y), 1.5, 1, color=color, fill=False, linewidth=2)
        ax3.add_patch(rect)
        ax3.text(x+offset, y+1.2, f"{name} (conf: 0.92)", color=color, fontsize=10)
    
    # 底部信息
    ax3.text(0.5, 0.5, "动物类别已移除 | H.265分层编码: 基础层+增强层 | FEC: 4+1抗丢包", 
             fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
    return []

ani3 = animation.FuncAnimation(fig3, update_vision, interval=100, blit=False)

print("演示窗口已启动！关闭窗口退出演示。")
plt.show()
print("演示结束！")
