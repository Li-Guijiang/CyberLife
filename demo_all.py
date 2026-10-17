# CyberLife 智能系统演示 - 科幻HUD风格
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.gait_system.wave_gait import WaveGait
from src.gait_system.triangle_gait import TriangleGait
from src.gait_system.alternating_gait import AlternatingGait
from src.gait_system.rotation_gait import RotationGait
from src.bci_imu_system.imu_fusion import Madgwick

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('dark_background')

fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor='#0a0e17')
fig.suptitle('CYBERLIFE EMBODIED COGNITION SYSTEM  |  v2.0  |  8-LIMB BIONIC PLATFORM', 
             color='#00ffff', fontsize=14, y=0.98, fontweight='bold')

ax1, ax2, ax3 = axes[0]
ax4, ax5, ax6 = axes[1]

# 1. 步态俯视图
ax1.set_facecolor('#0d1117')
ax1.set_xlim(-1,1); ax1.set_ylim(-1,1); ax1.set_aspect('equal')
ax1.set_title('LOCOMOTION - TOP VIEW', color='#00ff88', fontsize=10)
body = plt.Circle((0,0), 0.25, color='#1f6feb', alpha=0.3, ec='#00ffff', lw=2)
ax1.add_patch(body)
leg_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
legs, feet = [], []
for i in range(8):
    l, = ax1.plot([], [], color='#00ffff', lw=2, alpha=0.8)
    legs.append(l)
    f, = ax1.plot([], [], 'o', color='#ff0055', ms=6)
    feet.append(f)

# 2. 姿态显示
ax2.set_facecolor('#0d1117')
ax2.set_xlim(-1,1); ax2.set_ylim(-1,1); ax2.set_aspect('equal')
ax2.set_title('ATTITUDE - IMU FUSION', color='#ff00aa', fontsize=10)
horizon = plt.Circle((0,0), 0.8, color='#1f6feb', alpha=0.2, ec='#ff00aa', lw=1.5)
ax2.add_patch(horizon)
roll_line, = ax2.plot([], [], color='#00ff88', lw=2)
att_text = ax2.text(-0.9, 0.9, '', color='#ff00aa', fontsize=9)

# 3. 视觉检测
ax3.set_facecolor('#0d1117')
ax3.set_xlim(0,10); ax3.set_ylim(0,10)
ax3.set_title('PERCEPTION - OBJECT DETECTION', color='#ffff00', fontsize=10)
det_boxes = []
for i in range(3):
    b, = ax3.plot([], [], color='#00ff00', lw=2)
    det_boxes.append(b)
det_labels = [ax3.text(0,0,'', color='#00ff00', fontsize=8) for _ in range(3)]

# 4. IMU曲线
ax4.set_facecolor('#0d1117')
ax4.set_title('IMU - REAL-TIME SIGNALS', color='#00ffff', fontsize=10)
line_roll, = ax4.plot([], [], color='#ff0055', label='Roll', lw=1.5)
line_pitch, = ax4.plot([], [], color='#00ff88', label='Pitch', lw=1.5)
line_yaw, = ax4.plot([], [], color='#ff00aa', label='Yaw', lw=1.5)
ax4.legend(loc='upper right', facecolor='#161b22', labelcolor='white')
ax4.grid(alpha=0.2, color='#444')
ax4.set_ylim(-1, 1)

# 5. 稳定面板
ax5.set_facecolor('#0d1117')
ax5.set_title('STABILITY MONITOR', color='#ffaa00', fontsize=10)
status_text = ax5.text(0.1, 0.7, '', color='#00ff88', fontsize=12, fontweight='bold')
ax5.set_xlim(0,1); ax5.set_ylim(0,1)

# 6. 系统状态
ax6.set_facecolor('#0d1117')
ax6.set_title('SYSTEM STATUS', color='#00ffff', fontsize=10)
sys_text = ax6.text(0.1, 0.7, '', color='#00ff88', fontsize=10, linespacing=1.5)
ax6.set_xlim(0,1); ax6.set_ylim(0,1)

gaits = [
    (WaveGait(freq=1.0), 'WAVE GAIT - HIGH STABILITY'),
    (TriangleGait(freq=1.2), 'TRIANGLE GAIT - STATIC STABLE'),
    (AlternatingGait(freq=1.5), 'ALTERNATING GAIT - BALANCED'),
    (RotationGait(freq=1.2), 'ROTATION GAIT - IN-PLACE TURN')
]
gait_idx = 0
imu = Madgwick(dt=0.05)
roll_hist, pitch_hist, yaw_hist, x_hist = [], [], [], []

def update(frame):
    global gait_idx
    gait, gait_name = gaits[gait_idx]
    angles = gait.step(0.05)
    
    for i in range(8):
        bx = 0.25 * np.cos(leg_angles[i])
        by = 0.25 * np.sin(leg_angles[i])
        fx = bx + 0.35 * np.cos(leg_angles[i] + angles[i, 0])
        fy = by + 0.35 * np.sin(leg_angles[i] + angles[i, 0])
        legs[i].set_data([bx, fx], [by, fy])
        feet[i].set_data([fx], [fy])
    
    gyro = np.array([0.2*np.sin(frame*0.1), 0.1*np.cos(frame*0.08), 0.05*np.sin(frame*0.05)])
    imu.update(gyro, np.array([0,0,9.8]))
    roll, pitch, yaw = imu.get_euler()
    roll_line.set_data([-0.8*np.cos(roll), 0.8*np.cos(roll)], 
                       [-0.8*np.sin(roll), 0.8*np.sin(roll)])
    att_text.set_text(f'Pitch: {np.degrees(pitch):.1f}°\nRoll: {np.degrees(roll):.1f}°')
    
    positions = [(1, 6), (5, 4), (7, 7)]
    names = ['obstacle | 0.92', 'vegetation | 0.85', 'target | 0.78']
    for i in range(3):
        x, y = positions[i]
        dx = 0.5*np.sin(frame*0.05 + i)
        det_boxes[i].set_data([x+dx, x+dx+1.2], [y, y])
        det_labels[i].set_position((x+dx, y+1.2))
        det_labels[i].set_text(names[i])
    
    x_hist.append(frame)
    roll_hist.append(roll)
    pitch_hist.append(pitch)
    yaw_hist.append(yaw)
    if len(x_hist) > 100:
        x_hist.pop(0); roll_hist.pop(0); pitch_hist.pop(0); yaw_hist.pop(0)
    line_roll.set_data(x_hist, roll_hist)
    line_pitch.set_data(x_hist, pitch_hist)
    line_yaw.set_data(x_hist, yaw_hist)
    ax4.set_xlim(max(0, frame-100), frame+10)
    
    margin = 0.7 - 0.2*abs(np.sin(frame*0.05))
    status_text.set_text(f'STATUS: STABLE\nMargin: {margin:.2f}m')
    status_text.set_color('#00ff88' if margin > 0.3 else '#ff0055')
    
    sys_text.set_text(
        f'Gait Mode: {gait_name}\n'
        f'IMU Fusion: ACTIVE\n'
        f'Vision: DETECTING\n'
        f'FEC: ENABLED\n'
        f'Animal Filter: ON'
    )
    
    if frame % 200 == 199:
        gait_idx = (gait_idx + 1) % 4
    
    return legs + feet + [roll_line, att_text, line_roll, line_pitch, line_yaw, status_text, sys_text]

ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
plt.tight_layout()
print("="*60)
print("CyberLife 智能系统演示 (HUD风格)")
print("关闭窗口退出")
print("="*60)
plt.show()
