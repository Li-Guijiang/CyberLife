"""
Wave Gait Generator for 8-Limb Bionic Platform
[CyberLife Modified] 2026-10 | 适配排雷蜘蛛项目，优化八足波浪步态生成 | 软著标记
Based on open-source legged robot gait generation algorithms
Reference: MIT Mini Cheetah, OLOC locomotion frameworks
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class LegState:
    """State of a single leg"""
    hip_angle: float
    knee_angle: float
    ankle_angle: float
    is_swing: bool
    foot_position: np.ndarray

class WaveGaitGenerator:
    """
    Wave gait generator for 8-limb bionic carrier.
    Each leg follows a sequential phase offset for smooth, stable locomotion.
    Stability: High | Speed: Slow | Terrain: Rough
    """
    def __init__(self, num_legs: int = 8, freq: float = 1.0, 
                 stance_width: float = 0.5, step_length: float = 0.15):
        self.num_legs = num_legs
        self.freq = freq
        self.stance_width = stance_width
        self.step_length = step_length
        self.phase = 0.0
        # Wave gait: sequential phase offset across legs
        self.leg_phases = np.linspace(0, 2*np.pi, num_legs, endpoint=False)
        
    def update(self, dt: float) -> np.ndarray:
        """
        Update gait state.
        Returns: (num_legs, 3) joint angles [hip, knee, ankle]
        """
        self.phase = (self.phase + self.freq * dt) % (2 * np.pi)
        joint_angles = np.zeros((self.num_legs, 3))
        
        for leg in range(self.num_legs):
            local_phase = self.phase + self.leg_phases[leg]
            is_swing = np.sin(local_phase) > 0
            
            if is_swing:
                # Swing phase: leg lifts and moves forward
                joint_angles[leg, 0] = self.step_length * np.cos(local_phase)  # Hip
                joint_angles[leg, 1] = 0.3 * np.sin(local_phase)  # Knee lift
                joint_angles[leg, 2] = -0.1 * np.sin(local_phase)  # Ankle
            else:
                # Stance phase: leg pushes backward on ground
                joint_angles[leg, 0] = self.step_length * np.cos(local_phase)
                joint_angles[leg, 1] = 0.0  # Knee straight
                joint_angles[leg, 2] = 0.0  # Ankle neutral
                
        return joint_angles
    
    def get_foot_positions(self) -> np.ndarray:
        """Get foot positions relative to body center"""
        positions = np.zeros((self.num_legs, 2))
        for leg in range(self.num_legs):
            angle = (leg / self.num_legs) * 2 * np.pi
            positions[leg, 0] = self.stance_width * np.cos(angle)
            positions[leg, 1] = self.stance_width * np.sin(angle)
        return positions
