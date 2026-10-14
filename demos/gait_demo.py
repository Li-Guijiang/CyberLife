"""
CyberLife 八足机器人步态仿真演示
[CyberLife Modified] 2026-10 | 适配排雷蜘蛛项目，优化三角/波浪步态生成 | 软著标记
直接运行即可生成动画，录屏就是演示视频。
展示：波浪步态、三角步态、重心控制
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===================== 配置参数 =====================
GAIT_TYPE = "triangle"  # triangle 三角步态 / wave 波浪步态
BODY_LENGTH = 0.6
BODY_WIDTH = 0.4
LEG_LENGTH = 0.7
STEP_HEIGHT = 0.15

# 8条腿相对机身中心的安装位置
leg_offsets = np.array([
    [-BODY_LENGTH/2,  BODY_WIDTH/2],
    [ BODY_LENGTH/2,  BODY_WIDTH/2],
    [ BODY_LENGTH/2, -BODY_WIDTH/2],
    [-BODY_LENGTH/2, -BODY_WIDTH/2],
    [-BODY_LENGTH/3,  BODY_WIDTH/2],
    [ BODY_LENGTH/3,  BODY_WIDTH/2],
    [ BODY_LENGTH/3, -BODY_WIDTH/2],
    [-BODY_LENGTH/3, -BODY_WIDTH/2],
])

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-2, 4)
ax.set_ylim(-2, 2)
ax.set_title("CyberLife Bionic Locomotion Simulation\nEight-Legged Robot Gait Demonstration", fontsize=14)
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.grid(alpha=0.3)

# 绘图对象
body_rect, = ax.plot([], [], 'b-', lw=3)
foot_lines = []
foot_dots = []
for i in range(8):
    line, = ax.plot([], [], 'g-', lw=2)
    dot, = ax.plot([], [], 'ro', markersize=6)
    foot_lines.append(line)
    foot_dots.append(dot)

com_marker, = ax.plot([], [], '*', color='gold', markersize=12, label="Center of Gravity")
traj_line, = ax.plot([], [], 'b--', alpha=0.6, label="Body Trajectory")
ax.legend(loc="lower right")

body_x = 0.0
traj_x_list = []
traj_y_list = []

def get_leg_end(base_x, base_y, phase, gait, leg_idx):
    t = phase % 1.0
    if gait == "triangle":
        # 三角步态：分成两组交替
        swing_group = leg_idx % 4 < 2
    else:  # wave 波浪步态
        swing_group = (leg_idx + int(phase * 4)) % 4 == 0

    if swing_group:
        z = STEP_HEIGHT * np.sin(np.pi * t)
        x_off = LEG_LENGTH * np.cos(np.pi * t - np.pi/2)
    else:
        z = 0
        x_off = LEG_LENGTH * (t - 0.5)
    return base_x + x_off, base_y - z

def animate(frame):
    global body_x
    body_x += 0.02
    t_global = frame / 50.0
    body_center = np.array([body_x, 0.0])

    # 机身矩形四个顶点
    bx = [
        body_center[0] - BODY_LENGTH/2,
        body_center[0] + BODY_LENGTH/2,
        body_center[0] + BODY_LENGTH/2,
        body_center[0] - BODY_LENGTH/2,
        body_center[0] - BODY_LENGTH/2
    ]
    by = [
        body_center[1] + BODY_WIDTH/2,
        body_center[1] + BODY_WIDTH/2,
        body_center[1] - BODY_WIDTH/2,
        body_center[1] - BODY_WIDTH/2,
        body_center[1] + BODY_WIDTH/2
    ]
    body_rect.set_data(bx, by)

    # 重心
    com_x, com_y = body_center
    com_marker.set_data([com_x], [com_y])
    traj_x_list.append(com_x)
    traj_y_list.append(com_y)
    traj_line.set_data(traj_x_list, traj_y_list)

    for i in range(8):
        leg_base = body_center + leg_offsets[i]
        phase = t_global + i * (1.0/8.0)
        fx, fy = get_leg_end(leg_base[0], leg_base[1], phase, GAIT_TYPE, i)
        foot_lines[i].set_data([leg_base[0], fx], [leg_base[1], fy])
        foot_dots[i].set_data([fx], [fy])

    return [body_rect, com_marker, traj_line] + foot_lines + foot_dots

ani = FuncAnimation(fig, animate, frames=600, interval=40, blit=True)

# 保存为视频（可选）
# ani.save('gait_demo.mp4', writer='ffmpeg', fps=20, dpi=100)

plt.show()
