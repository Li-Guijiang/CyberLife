"""
Quad-IMU Attitude Fusion using Extended Kalman Filter
[CyberLife Modified] 2026-10 | 适配排雷蜘蛛项目，四陀螺仪姿态融合算法 | 软著标记
Reference: ROS imu_filter_madgwick, filterpy library
Fuses 4 gyroscope + accelerometer readings for robust pose estimation
"""
import numpy as np

class MadgwickAHRS:
    """
    Madgwick's IMU attitude estimation algorithm.
    Efficient gradient-descent based sensor fusion.
    """
    def __init__(self, sample_period: float = 0.01, beta: float = 0.1):
        self.dt = sample_period
        self.beta = beta  # Filter gain
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion: [w, x, y, z]
        
    def update_imu(self, gyro: np.ndarray, accel: np.ndarray):
        """
        Update attitude with gyroscope and accelerometer.
        gyro: (3,) angular velocity [roll_rate, pitch_rate, yaw_rate] rad/s
        accel: (3,) acceleration [ax, ay, az] m/s^2
        """
        q = self.q
        
        # Normalize accelerometer measurement
        accel = accel / np.linalg.norm(accel)
        
        # Gradient descent correction
        # Objective function: direction of gravity estimate
        wx, wy, wz = gyro
        qw, qx, qy, qz = q
        
        # Gradient of cost function
        f = np.array([
            2*(qx*qz - qw*qy) - accel[0],
            2*(qw*qx + qy*qz) - accel[1],
            2*(0.5 - qx*qx - qy*qy) - accel[2]
        ])
        
        # Jacobian transpose
        jt = np.array([
            [-2*qz,  2*qy, -2*qx],
            [ 2*qy,  2*qz,  2*qw],
            [-2*qw, -2*qx,  2*qy],
            [ 2*qx,  2*qw,  2*qz]
        ])
        
        # Gradient descent step
        gradient = jt @ f
        gradient = gradient / np.linalg.norm(gradient)
        
        # Quaternion derivative from gyro
        q_dot = 0.5 * np.array([
            -qx*wx - qy*wy - qz*wz,
             qw*wx + qy*wz - qz*wy,
             qw*wy - qx*wz + qz*wx,
             qw*wz + qx*wy - qy*wx
        ])
        
        # Apply correction
        self.q = q + self.dt * (q_dot - self.beta * gradient)
        self.q = self.q / np.linalg.norm(self.q)
        
    def get_euler(self) -> tuple:
        """Return (roll, pitch, yaw) in radians"""
        qw, qx, qy, qz = self.q
        
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        pitch = np.arcsin(2*(qw*qy - qz*qx))
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        return roll, pitch, yaw


class QuadIMUFusion:
    """
    Multi-IMU array fusion: averages 4 IMU readings for robustness.
    Reduces noise and drift compared to single IMU.
    """
    def __init__(self, num_imus: int = 4, sample_period: float = 0.01):
        self.num_imus = num_imus
        self.filters = [MadgwickAHRS(sample_period) for _ in range(num_imus)]
        
    def update(self, gyro_data: np.ndarray, accel_data: np.ndarray):
        """
        Fuse data from multiple IMUs.
        gyro_data: (num_imus, 3) angular velocity readings
        accel_data: (num_imus, 3) acceleration readings
        Returns: average (roll, pitch, yaw) in radians
        """
        eulers = []
        for i in range(self.num_imus):
            self.filters[i].update_imu(gyro_data[i], accel_data[i])
            eulers.append(self.filters[i].get_euler())
            
        # Average across IMUs
        eulers = np.array(eulers)
        return np.mean(eulers, axis=0)
