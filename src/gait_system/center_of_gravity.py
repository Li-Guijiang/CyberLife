# 动态重心控制
# 参考了MIT猎豹机器人的重心调整算法，改了简化版
# 走路的时候自动调整重心，防止翻倒
import numpy as np

class CoGController:
    """
    重心控制器
    就是走路的时候把重心往支撑多边形中间挪，这样就稳了
    还有个安全余量，不会贴边走
    """
    def __init__(self, safety_margin=0.15, max_shift=0.1):
        self.safety = safety_margin  # 安全余量
        self.max_shift = max_shift   # 一次最多挪多少
        self.cur_cog = [0.0, 0.0]    # 当前重心位置
        
    def get_support_polygon(self, feet):
        """
        算一下现在踩在地上的脚围成的多边形
        feet是所有脚的坐标，形状(n, 2)
        """
        center = np.mean(feet, axis=0)
        # 按角度排序，就是凸包了
        angles = np.arctan2(feet[:,1] - center[1], feet[:,0] - center[0])
        idx = np.argsort(angles)
        return feet[idx]
    
    def update(self, feet, slope=0.0):
        """
        更新重心位置
        slope是地面坡度，上坡的时候重心往后挪一点
        """
        polygon = self.get_support_polygon(feet)
        # 目标就是多边形的中心
        target = np.mean(polygon, axis=0)
        # 坡度补偿
        target[1] += slope * 0.1
        
        # 限制一次挪动的距离，别挪太猛
        diff = target - self.cur_cog
        dist = np.linalg.norm(diff)
        if dist > self.max_shift:
            diff = diff / dist * self.max_shift
            
        self.cur_cog = self.cur_cog + diff
        return self.cur_cog
    
    def stability_margin(self, feet):
        """
        算一下现在还有多少稳定余量
        就是重心到支撑多边形边的最短距离
        """
        polygon = self.get_support_polygon(feet)
        min_dist = 999
        
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i+1) % len(polygon)]
            # 点到线段的距离
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            t = np.dot(self.cur_cog - p1, p2 - p1) / (dx*dx + dy*dy)
            t = max(0, min(1, t))
            proj = p1 + t * (p2 - p1)
            d = np.linalg.norm(self.cur_cog - proj)
            min_dist = min(min_dist, d)
            
        return min_dist - self.safety
