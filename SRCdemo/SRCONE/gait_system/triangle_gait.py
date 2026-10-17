# 三角步态生成器
# 参考了Stanford Doggo的三脚架步态，改成八足的版本
# 这个步态最稳，任何时候都有4条腿踩在地上，不会翻
import numpy as np

class TriangleGait:
    """
    三角步态（三脚架步态）
    把8条腿分成两组，一组站着的时候另一组迈腿
    这样任何时候都有至少4条腿在地上，静态稳定
    """
    def __init__(self, num_legs=8, freq=1.2):
        self.n = num_legs
        self.freq = freq
        self.phase = 0.0
        # 分组：A组和B组交替动
        self.group_a = [0, 2, 5, 7]
        self.group_b = [1, 3, 4, 6]
        
    def step(self, dt):
        """
        更新步态，返回所有腿的关节角
        """
        self.phase = (self.phase + self.freq * dt) % (2 * np.pi)
        angles = np.zeros((self.n, 3))
        
        for i in range(self.n):
            if i in self.group_a:
                p = self.phase
            else:
                p = self.phase + np.pi
                
            swing = np.sin(p) > 0
            
            if swing:
                # 摆动相：抛物线轨迹，中间抬最高
                t = (np.sin(p) + 1) / 2
                angles[i, 0] = 0.12 * np.cos(p)
                angles[i, 1] = 0.06 * 4 * t * (1 - t)  # 抛物线抬脚
                angles[i, 2] = -0.5 * angles[i, 1]
            else:
                # 支撑相：脚踩在地上不动
                angles[i, 0] = 0.12 * np.cos(p)
                angles[i, 1] = 0.02
                angles[i, 2] = 0.0
                
        return angles
