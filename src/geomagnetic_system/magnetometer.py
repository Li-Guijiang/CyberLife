# 磁力计采集层
# 有实物就接串口，没实物就用仿真数据，接口一样，上层不用改
#
# 单位说明：
#   磁场 B        纳特斯拉 nT
#   位置 x,y,z    米 m，z 轴向下为正，所以传感器离地高度 h 时 z = -h
#   磁矩 m        A·m^2
import math
import numpy as np

# 真空磁导率除 4π，磁偶极子公式里要用
MU0_4PI = 1e-7

# 地球主磁场，国内大概 48000~55000 nT，这里取 50000
EARTH_TOTAL_NT = 50000.0
# 磁倾角，贵州一带约 40~45 度
EARTH_INCLINATION_DEG = 45.0


def earth_field(total_nt=EARTH_TOTAL_NT, inclination_deg=EARTH_INCLINATION_DEG,
                declination_deg=0.0):
    """
    算地球主磁场矢量，nT
    x 指北，y 指东，z 向下，倾角向下为正
    """
    inc = math.radians(inclination_deg)
    dec = math.radians(declination_deg)
    horizontal = total_nt * math.cos(inc)
    return np.array([
        horizontal * math.cos(dec),
        horizontal * math.sin(dec),
        total_nt * math.sin(inc),
    ])


def dipole_field(moment, rel_pos):
    """
    磁偶极子在 rel_pos 处产生的磁场，返回 nT
    moment 是磁矩矢量 A·m^2，rel_pos 是测点减目标的位置差，单位 m

    公式 B = (μ0/4π) * [3(m·r̂)r̂ - m] / r^3
    这个是标准偶极子近似，距离大于目标尺寸 3 倍以上才准
    """
    r = float(np.linalg.norm(rel_pos))
    if r < 1e-6:
        return np.zeros(3)
    rh = np.asarray(rel_pos, dtype=float) / r
    m = np.asarray(moment, dtype=float)
    b = MU0_4PI * (3.0 * np.dot(m, rh) * rh - m) / (r ** 3)
    return b * 1e9  # T 换成 nT


class MagnetometerBase:
    """磁力计基类，所有驱动都实现这个接口"""

    def read(self):
        """返回一帧磁场矢量 [Bx, By, Bz]，单位 nT"""
        raise NotImplementedError

    def close(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__


class SimulatedMagnetometer(MagnetometerBase):
    """
    仿真磁力计

    重要：这是仿真，不是真传感器。数据由三部分组成：
      1. 地球主磁场
      2. 地下金属目标的偶极子异常（可多个）
      3. 硬磁/软磁误差 + 高斯噪声

    加硬磁软磁误差是为了让校准算法有东西可校，接实物时这两项是真实存在的。
    """

    def __init__(self, height=0.15, targets=None, noise_nt=12.0,
                 hard_iron=(0.0, 0.0, 0.0), soft_iron=None, seed=0,
                 earth=None):
        # height 是传感器离地高度，单位 m
        self.height = float(height)
        # targets 里每个是 dict，形如
        # {'name': '弹片', 'pos': (x, y, depth), 'moment': (mx, my, mz)}
        self.targets = list(targets or [])
        self.noise_nt = float(noise_nt)
        self.hard_iron = np.asarray(hard_iron, dtype=float)
        if soft_iron is None:
            soft_iron = np.eye(3)
        self.soft_iron = np.asarray(soft_iron, dtype=float)
        self.earth = earth_field() if earth is None else np.asarray(earth, dtype=float)
        self._rng = np.random.default_rng(seed)
        self._pos = np.array([0.0, 0.0])

    def set_position(self, x, y):
        """告诉仿真器机器人走到哪了"""
        self._pos = np.array([float(x), float(y)])

    def add_target(self, name, pos, moment):
        self.targets.append({"name": name, "pos": tuple(pos), "moment": tuple(moment)})

    def true_anomaly(self):
        """
        不含误差的真实异常量，nT
        调试和验收时拿它和检测结果对账用
        """
        sensor = np.array([self._pos[0], self._pos[1], -self.height])
        total = np.zeros(3)
        for t in self.targets:
            tx, ty, tz = t["pos"]
            total = total + dipole_field(t["moment"], sensor - np.array([tx, ty, tz]))
        return total

    def read(self):
        b = self.earth + self.true_anomaly()
        # 软磁误差：地磁场被机器铁件扭曲
        b = self.soft_iron @ b
        # 硬磁误差：机身自带磁化
        b = b + self.hard_iron
        # 传感器噪声
        b = b + self._rng.normal(0.0, self.noise_nt, size=3)
        return b


class CsvReplayMagnetometer(MagnetometerBase):
    """
    回放磁力计：读现场录的 CSV，三列 Bx,By,Bz（nT），可带表头
    现场采完数据拿回来跑算法就用这个
    """

    def __init__(self, path, loop=False):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                try:
                    vals = [float(p) for p in parts[:3]]
                except ValueError:
                    continue  # 表头跳过
                if len(vals) == 3:
                    rows.append(vals)
        if not rows:
            raise ValueError("CSV 里没读到有效数据：%s" % path)
        self.data = np.asarray(rows, dtype=float)
        self.loop = loop
        self._i = 0

    def read(self):
        if self._i >= len(self.data):
            if not self.loop:
                raise StopIteration("CSV 数据读完了")
            self._i = 0
        row = self.data[self._i]
        self._i += 1
        return row.copy()


class SerialMagnetometer(MagnetometerBase):
    """
    串口磁力计，接实物用

    默认按文本协议解析，一行一帧，格式：
        MAG,1234.5,678.9,1011.2
    单位 nT。如果买的是 RM3100 或 HMC5883L 这类模块，
    把它们的二进制协议换成对应解析即可，上层不用动。

    注意：本机没有接实物，这条路径没有做过上机验证，
    只做了协议解析的单元测试。
    """

    def __init__(self, port="COM3", baudrate=115200, timeout=1.0, prefix="MAG"):
        try:
            import serial  # pyserial
        except ImportError as e:
            raise RuntimeError(
                "没装 pyserial，接实物前先 pip install pyserial"
            ) from e
        self._ser = serial.Serial(port, baudrate, timeout=timeout)
        self.prefix = prefix

    def read(self):
        # 最多读 20 行，防止串口里全是噪声卡死
        for _ in range(20):
            line = self._ser.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 4 and parts[0].strip().upper() == self.prefix:
                try:
                    return np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    continue
        raise TimeoutError("串口连续 20 行没读到合法数据帧")

    def close(self):
        try:
            self._ser.close()
        except Exception:
            pass


def make_magnetometer(mode="sim", **kwargs):
    """
    工厂函数，省得上层判断用哪种
    mode: sim / csv / serial
    """
    mode = mode.lower()
    if mode == "sim":
        return SimulatedMagnetometer(**kwargs)
    if mode == "csv":
        return CsvReplayMagnetometer(**kwargs)
    if mode == "serial":
        return SerialMagnetometer(**kwargs)
    raise ValueError("不认识的模式：%s，只能是 sim / csv / serial" % mode)
