# 四IMU姿态融合
# 参考了Madgwick的AHRS算法，改了简化版
# 我们蜘蛛上装了4个陀螺仪，融合一下更准
import numpy as np

class Madgwick:
    """
    Madgwick姿态估计
    用梯度下降法融合陀螺仪和加速度计
    比卡尔曼简单，速度快，适合我们这种小机器人
    """
    def __init__(self, dt=0.01, beta=0.1):
        self.dt = dt  # 采样时间
        self.beta = beta  # 滤波增益
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # 四元数
        
    def update(self, gyro, accel):
        """
        更新姿态
        gyro是角速度[roll, pitch, yaw]，单位rad/s
        accel是加速度计[x, y, z]
        """
        q = self.q
        
        # 加速度计归一化
        accel = accel / np.linalg.norm(accel)
        
        # 梯度下降修正
        wx, wy, wz = gyro
        qw, qx, qy, qz = q
        
        # 代价函数的梯度
        f = np.array([
            2*(qx*qz - qw*qy) - accel[0],
            2*(qw*qx + qy*qz) - accel[1],
            2*(0.5 - qx*qx - qy*qy) - accel[2]
        ])
        
        # 雅可比转置
        jt = np.array([
            [-2*qz,  2*qy, -2*qx],
            [ 2*qy,  2*qz,  2*qw],
            [-2*qw, -2*qx,  2*qy],
            [ 2*qx,  2*qw,  2*qz]
        ])
        
        gradient = jt @ f
        norm = np.linalg.norm(gradient)
        if norm > 1e-6:
            gradient = gradient / norm
        
        # 四元数导数
        q_dot = 0.5 * np.array([
            -qx*wx - qy*wy - qz*wz,
             qw*wx + qy*wz - qz*wy,
             qw*wy - qx*wz + qz*wx,
             qw*wz + qx*wy - qy*wx
        ])
        
        # 更新四元数
        self.q = q + self.dt * (q_dot - self.beta * gradient)
        self.q = self.q / np.linalg.norm(self.q)
        
    def get_euler(self):
        """
        转换成欧拉角，返回(roll, pitch, yaw)
        """
        qw, qx, qy, qz = self.q
        
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        pitch = np.arcsin(2*(qw*qy - qz*qx))
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        return roll, pitch, yaw


class QuadIMUFusion:
    """
    四个IMU融合
    就是把四个陀螺仪的数据平均一下，更稳
    """
    def __init__(self, n_imus=4):
        self.n = n_imus
        self.filters = [Madgwick() for _ in range(n_imus)]
        
    def update(self, gyros, accels):
        """
        gyros形状是(4, 3)，四个陀螺仪的数据
        accels形状是(4, 3)，四个加速度计的数据
        返回平均后的(roll, pitch, yaw)
        """
        eulers = []
        for i in range(self.n):
            self.filters[i].update(gyros[i], accels[i])
            eulers.append(self.filters[i].get_euler())
            
        return np.mean(eulers, axis=0)
