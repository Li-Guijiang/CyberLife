# 地磁金属探测模块
# 排雷蜘蛛靠地磁找地下金属，这个包负责采集、校准、找异常、估深度
# 单位统一：磁场用 nT，位置用 m，磁矩用 A·m^2
from .anomaly import (
    AnomalyDetector,
    classify_metal,
    estimate_noise,
    extract_targets,
    fit_dipole_profile,
    robust_sigma,
)
from .calibration import EllipsoidCalibration
from .magnetometer import (
    CsvReplayMagnetometer,
    MagnetometerBase,
    SerialMagnetometer,
    SimulatedMagnetometer,
    dipole_field,
    earth_field,
    make_magnetometer,
)
from .scanner import AreaScanner, grid_anomaly, plan_lawnmower

__all__ = [
    "AnomalyDetector",
    "AreaScanner",
    "CsvReplayMagnetometer",
    "EllipsoidCalibration",
    "MagnetometerBase",
    "SerialMagnetometer",
    "SimulatedMagnetometer",
    "classify_metal",
    "dipole_field",
    "earth_field",
    "estimate_noise",
    "extract_targets",
    "fit_dipole_profile",
    "grid_anomaly",
    "make_magnetometer",
    "plan_lawnmower",
    "robust_sigma",
]
