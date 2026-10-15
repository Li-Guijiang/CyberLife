"""
Dynamic Center of Gravity Control
[CyberLife Modified] 2026-10 | 适配排雷蜘蛛项目，优化重心下移控制逻辑 | 软著标记
Adjusts CoG position during locomotion to maintain stability.
Reference: MIT Cheetah 3 MPC, Boston Dynamics Atlas balance control
"""
import numpy as np

class DynamicCoGController:
    """
    Real-time center of gravity migration controller.
    Maintains stability margin during gait transitions and uneven terrain.
    """
    def __init__(self, safety_margin: float = 0.15, max_shift: float = 0.1):
        self.safety_margin = safety_margin
        self.max_shift = max_shift
        self.current_cog = np.array([0.0, 0.0, 0.0])
        self.target_cog = np.array([0.0, 0.0, 0.0])
        
    def compute_support_polygon(self, foot_positions: np.ndarray) -> np.ndarray:
        """
        Compute convex hull of stance foot positions.
        foot_positions: (n, 2) array of foot x,y coordinates
        Returns: ordered vertices of support polygon
        """
        # Simple convex hull approximation for 8 legs
        centroid = np.mean(foot_positions, axis=0)
        angles = np.arctan2(foot_positions[:,1] - centroid[1], 
                           foot_positions[:,0] - centroid[0])
        sorted_indices = np.argsort(angles)
        return foot_positions[sorted_indices]
    
    def update_cog_target(self, foot_positions: np.ndarray, 
                         terrain_slope: float = 0.0) -> np.ndarray:
        """
        Compute target CoG position for stability.
        Shifts CoG toward center of support polygon.
        """
        support_polygon = self.compute_support_polygon(foot_positions)
        
        # Target: center of support polygon
        target = np.mean(support_polygon, axis=0)
        
        # Apply terrain slope compensation
        target[1] += terrain_slope * 0.1
        
        # Limit shift to max allowed
        shift = target - self.current_cog[:2]
        shift_norm = np.linalg.norm(shift)
        if shift_norm > self.max_shift:
            shift = shift / shift_norm * self.max_shift
            
        self.target_cog[:2] = self.current_cog[:2] + shift
        self.current_cog = self.target_cog.copy()
        
        return self.current_cog
    
    def get_stability_margin(self, foot_positions: np.ndarray) -> float:
        """Calculate minimum distance from CoG to support polygon edge"""
        polygon = self.compute_support_polygon(foot_positions)
        min_dist = float('inf')
        
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i+1) % len(polygon)]
            # Distance from point to line segment
            dist = self._point_to_line_dist(self.current_cog[:2], p1, p2)
            min_dist = min(min_dist, dist)
            
        return min_dist - self.safety_margin
    
    def _point_to_line_dist(self, point: np.ndarray, 
                           line_start: np.ndarray, line_end: np.ndarray) -> float:
        """Calculate distance from point to line segment"""
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        if dx == 0 and dy == 0:
            return np.linalg.norm(point - line_start)
        
        t = np.dot(point - line_start, line_end - line_start) / (dx*dx + dy*dy)
        t = max(0, min(1, t))
        proj = line_start + t * (line_end - line_start)
        return np.linalg.norm(point - proj)
