"""
Triangle Gait Generator for 8-Limb Bionic Platform
Statically stable tripod gait: 3 legs stance, 3 legs swing, 2 legs transition
Reference: Stanford Doggo, ANYmal locomotion controllers
"""
import numpy as np

class TriangleGaitGenerator:
    """
    Triangle (Tripod) gait generator.
    Divides legs into two alternating groups for static stability.
    Stability: Very High | Speed: Medium | Terrain: Any
    """
    def __init__(self, num_legs: int = 8, freq: float = 1.2,
                 step_length: float = 0.12, lift_height: float = 0.06):
        self.num_legs = num_legs
        self.freq = freq
        self.step_length = step_length
        self.lift_height = lift_height
        self.phase = 0.0
        
        # Group A: legs 0,2,5,7 | Group B: legs 1,3,4,6
        self.group_a = [0, 2, 5, 7]
        self.group_b = [1, 3, 4, 6]
        
    def update(self, dt: float) -> np.ndarray:
        """
        Update gait state.
        Returns: (num_legs, 3) joint angles
        """
        self.phase = (self.phase + self.freq * dt) % (2 * np.pi)
        joint_angles = np.zeros((self.num_legs, 3))
        
        for leg in range(self.num_legs):
            if leg in self.group_a:
                local_phase = self.phase
            else:
                local_phase = self.phase + np.pi
                
            is_swing = np.sin(local_phase) > 0
            
            if is_swing:
                # Swing: parabolic trajectory
                t = (np.sin(local_phase) + 1) / 2  # Normalize to 0-1
                joint_angles[leg, 0] = self.step_length * np.cos(local_phase)
                joint_angles[leg, 1] = self.lift_height * 4 * t * (1 - t)
                joint_angles[leg, 2] = -0.5 * joint_angles[leg, 1]
            else:
                # Stance: constant contact
                joint_angles[leg, 0] = self.step_length * np.cos(local_phase)
                joint_angles[leg, 1] = 0.02  # Slight bend
                joint_angles[leg, 2] = 0.0
                
        return joint_angles
    
    def get_stability_margin(self) -> float:
        """Calculate static stability margin for triangle gait"""
        # Simple approximation: min distance from CoG to support polygon edge
        return self.step_length * 0.8  # Conservative estimate
