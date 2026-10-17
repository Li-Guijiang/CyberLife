# 原地旋转步态
# 参考了ANYmal的原地转向实现，改了八足的旋转方向
# 左边的腿往后蹬，右边的腿往前迈，这样就能原地转了
import numpy as np

class RotationGait:
    """
    原地旋转步态
    不需要往前走，就是在原地转方向
    左边腿和右边腿方向相反，产生转动力矩
    """
    def __init__(self, num_legs=8, freq=1.2, direction=1):
        self.n = num_legs
        self.freq = freq
        self.direction = direction  # 1是顺时针，-1是逆时针
        self.phase = 0.0
        
    def step(self, dt):
        """
        更新旋转步态的关节角
        """
        self.phase = (self.phase + self.freq * dt) % (2 * np.pi)
        angles = np.zeros((self.n, 3))
        
        for i in range(self.n):
            # 前4条腿和后4条腿相位错开
            p = self.phase + (i % 4) * (np.pi / 2)
            
            if np.sin(p) > 0:
                # 摆动相
                # 左边的腿和右边的腿方向相反
                side = 1 if i < 4 else -1
                angles[i, 0] = 0.1 * np.cos(p) * side * self.direction
                angles[i, 1] = 0.06 * np.sin(p)
            else:
                angles[i, 0] = 0.05 * np.cos(p)
                angles[i, 1] = 0.01
                
        return angles
