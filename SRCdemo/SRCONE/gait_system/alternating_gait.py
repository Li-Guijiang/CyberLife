# 对角交替步态生成器
# 参考了Stanford Doggo的步态实现，改了八足的相位分组
# 改了下相位偏移，适配我们的蜘蛛机器人
import numpy as np

class AlternatingGait:
    """
    对角交替步态
    就是对角的两条腿一起动，速度中等，比较稳
    """
    def __init__(self, num_legs=8, freq=1.5):
        self.n = num_legs
        self.freq = freq
        self.phase = 0.0
        # 对角分组：0和4一组，1和5一组...
        self.group_a = [0, 3, 4, 7]
        self.group_b = [1, 2, 5, 6]
        
    def step(self, dt):
        """
        每步更新一次关节角
        返回形状是(8, 3)的数组，每一行对应一条腿的三个关节
        """
        self.phase = (self.phase + self.freq * dt) % (2 * np.pi)
        angles = np.zeros((self.n, 3))
        
        for i in range(self.n):
            if i in self.group_a:
                p = self.phase
            else:
                p = self.phase + np.pi
                
            # 摆动相：腿抬起来往前迈
            if np.sin(p) > 0:
                angles[i, 0] = 0.12 * np.cos(p)  # 髋关节前后
                angles[i, 1] = 0.08 * np.sin(p)  # 膝关节弯一下
            else:
                # 支撑相：腿踩在地上往后蹬
                angles[i, 0] = 0.12 * np.cos(p)
                angles[i, 1] = 0.01  # 稍微弯一点，不打直
                
        return angles
