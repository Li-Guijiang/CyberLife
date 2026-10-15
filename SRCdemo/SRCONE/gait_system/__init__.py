# 仿生运动子系统
# 四个步态：波浪、三角、交替、旋转
# 改了相位适配我们的八足蜘蛛机器人 10月
from .wave_gait import WaveGait
from .triangle_gait import TriangleGait
from .alternating_gait import AlternatingGait
from .rotation_gait import RotationGait
from .center_of_gravity import CoGController

__all__ = ['WaveGait', 'TriangleGait', 'AlternatingGait', 'RotationGait', 'CoGController']
