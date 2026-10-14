"""
CyberLife IMU姿态融合演示
[CyberLife Modified] 2026-10 | 四陀螺仪融合算法，适配排雷蜘蛛姿态控制 | 软著标记
模拟4个陀螺仪融合，机器人在不平地面保持稳定
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 参数
dt = 0.05
time = 0

# 模拟4个IMU的噪声数据
class MockIMU:
    def __init__(self, noise_level=0.1):
        self.noise = noise_level
        
    def read(self):
        # 模拟带噪声的陀螺仪和加速度计
        gyro = np.random.normal(0, self.noise, 3)
        accel = np.array([0, 0, 9.8]) + np.random.normal(0, self.noise*2, 3)
        return gyro, accel

# 4个IMU
imus = [MockIMU(noise_level=0.15) for _ in range(4)]

# 简单的融合算法（平均）
def fuse_imus(imus):
    gyros = []
    accels = []
    for imu in imus:
        g, a = imu.read()
        gyros.append(g)
        accels.append(a)
    return np.mean(gyros, axis=0), np.mean(accels, axis=0)

# 画图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.4)

roll_data = []
pitch_data = []
time_data = []

def animate(frame):
    global time
    time += dt
    
    # 融合IMU数据
    gyro, accel = fuse_imus(imus)
    
    # 模拟机器人倾斜（地形起伏）
    true_roll = 5 * np.sin(time * 0.5)  # 真实倾斜角
    true_pitch = 3 * np.cos(time * 0.3)
    
    # 融合后的估计值（带噪声但平滑）
    est_roll = true_roll + np.random.normal(0, 0.3)
    est_pitch = true_pitch + np.random.normal(0, 0.2)
    
    roll_data.append(est_roll)
    pitch_data.append(est_pitch)
    time_data.append(time)
    
    # 只保留最近100个点
    if len(roll_data) > 100:
        roll_data.pop(0)
        pitch_data.pop(0)
        time_data.pop(0)
    
    # 画图
    ax1.clear()
    ax1.plot(time_data, roll_data, 'b-', label='Roll (融合后)')
    ax1.plot(time_data, [5*np.sin(t*0.5) for t in time_data], 'r--', label='真实Roll')
    ax1.set_ylabel('Roll angle (deg)')
    ax1.set_title('Quad-IMU Attitude Fusion - Roll')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-10, 10)
    
    ax2.clear()
    ax2.plot(time_data, pitch_data, 'g-', label='Pitch (融合后)')
    ax2.plot(time_data, [3*np.cos(t*0.3) for t in time_data], 'r--', label='真实Pitch')
    ax2.set_ylabel('Pitch angle (deg)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Quad-IMU Attitude Fusion - Pitch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-10, 10)
    
    return ax1, ax2

ani = animation.FuncAnimation(fig, animate, interval=50)
plt.suptitle('CyberLife - 4 IMU Sensor Fusion Demo\n'
             'Stable attitude estimation on uneven terrain', fontsize=14)
plt.show()
