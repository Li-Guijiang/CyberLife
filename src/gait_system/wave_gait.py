# 波浪步态生成器
# 参考了MIT迷你猎豹的步态代码，改了八足的相位
# 我们蜘蛛机器人用这个步态在不平的地方走比较稳
import numpy as np

class LegState:
    """单条腿的状态"""
    def __init__(self):
        self.hip = 0.0
        self.knee = 0.0
        self.ankle = 0.0
        self.is_swing = False
        self.foot_pos = [0.0, 0.0]

class WaveGait:
    """
    波浪步态
    每条腿按顺序依次摆动，像波浪一样，所以叫波浪步态
    优点是稳，缺点是慢，适合走比较烂的路
    """
    def __init__(self, num_legs=8, freq=1.0):
        self.n = num_legs
        self.freq = freq
        self.phase = 0.0
        # 每条腿相位错开一点，这样就不会一起动了
        self.leg_phases = np.linspace(0, 2*np.pi, num_legs, endpoint=False)
        
    def step(self, dt):
        """
        每步更新一次，返回所有腿的关节角
        返回形状是(8, 3)，每一行是[髋, 膝, 踝]三个关节
        """
        self.phase = (self.phase + self.freq * dt) % (2 * np.pi)
        angles = np.zeros((self.n, 3))
        
        for i in range(self.n):
            p = self.phase + self.leg_phases[i]
            swing = np.sin(p) > 0
            
            if swing:
                # 摆动相：腿抬起来往前迈
                angles[i, 0] = 0.15 * np.cos(p)  # 髋关节前后摆
                angles[i, 1] = 0.1 * np.sin(p)   # 膝关节弯一下
                angles[i, 2] = -0.05 * np.sin(p) # 踝关节调整
            else:
                # 支撑相：腿踩在地上往后蹬
                angles[i, 0] = 0.15 * np.cos(p)
                angles[i, 1] = 0.0
                angles[i, 2] = 0.0
                
        return angles
