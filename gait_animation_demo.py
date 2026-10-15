# 八足机器人步态动画演示
# 用matplotlib画一个俯视图，展示四个步态的运动效果
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 解决matplotlib中文显示方块的问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from src.gait_system.wave_gait import WaveGait
from src.gait_system.triangle_gait import TriangleGait
from src.gait_system.alternating_gait import AlternatingGait
from src.gait_system.rotation_gait import RotationGait

# 机器人身体参数
body_radius = 0.3
leg_length = 0.4
leg_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)

# 四种步态
gaits = [
    (WaveGait(freq=1.0), "波浪步态 (Wave) - 最稳，适合烂路"),
    (TriangleGait(freq=1.2), "三角步态 (Triangle) - 静态稳定"),
    (AlternatingGait(freq=1.5), "交替步态 (Alternating) - 速度中等"),
    (RotationGait(freq=1.2), "旋转步态 (Rotation) - 原地转向")
]
gait_idx = 0

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
title = ax.set_title("八足机器人步态演示", fontsize=14)

# 画身体
body = plt.Circle((0, 0), body_radius, color='gray', alpha=0.5)
ax.add_patch(body)

# 腿的线
legs = []
for i in range(8):
    line, = ax.plot([], [], 'b-', linewidth=3)
    legs.append(line)

# 脚的点
feet = []
for i in range(8):
    pt, = ax.plot([], [], 'ro', markersize=8)
    feet.append(pt)

def init():
    for i in range(8):
        legs[i].set_data([], [])
        feet[i].set_data([], [])
    return legs + feet + [title]

def update(frame):
    global gait_idx
    gait, gait_name = gaits[gait_idx]
    
    # 更新步态
    angles = gait.step(0.05)
    
    # 计算每条腿的末端位置
    for i in range(8):
        base_x = body_radius * np.cos(leg_angles[i])
        base_y = body_radius * np.sin(leg_angles[i])
        
        # 髋关节角决定脚的位置
        hip = angles[i, 0]
        foot_x = base_x + leg_length * np.cos(leg_angles[i] + hip)
        foot_y = base_y + leg_length * np.sin(leg_angles[i] + hip)
        
        legs[i].set_data([base_x, foot_x], [base_y, foot_y])
        feet[i].set_data([foot_x], [foot_y])
    
    title.set_text(f"八足机器人步态演示\n{gait_name}\n切换步态按空格")
    
    # 每200帧切换一次步态
    if frame % 200 == 199:
        gait_idx = (gait_idx + 1) % 4
    
    return legs + feet + [title]

def on_key(event):
    global gait_idx
    if event.key == ' ':
        gait_idx = (gait_idx + 1) % 4
        print(f"切换到: {gaits[gait_idx][1]}")

fig.canvas.mpl_connect('key_press_event', on_key)

ani = animation.FuncAnimation(fig, update, init_func=init, interval=50, blit=False)
print("="*50)
print("八足机器人步态动画演示")
print("按空格键切换四个步态")
print("关闭窗口退出")
print("="*50)
plt.show()
