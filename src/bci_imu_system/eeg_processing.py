# EEG脑电信号预处理
# 参考了MNE-Python的预处理流程，简化了一下
# 我们脑机头套采到的信号先过这个预处理，滤掉噪声
import numpy as np
from scipy import signal

class EEGPreprocess:
    """
    脑电信号预处理
    步骤：去基线 -> 带通滤波 -> 切片
    就是把原始的脑电信号洗一遍，去掉没用的噪声
    """
    def __init__(self, fs=250):
        self.fs = fs  # 采样率
        # 我们要的是8-30Hz的信号，就是mu波和beta波
        self.low = 8.0
        self.high = 30.0
        self._make_filter()
        
    def _make_filter(self):
        """设计一个4阶巴特沃斯带通滤波器"""
        nyq = 0.5 * self.fs
        self.b, self.a = signal.butter(4, [self.low/nyq, self.high/nyq], btype='band')
        
    def process(self, raw):
        """
        处理原始脑电信号
        raw形状是(通道数, 采样点数)
        返回滤波后的信号
        """
        # 去基线：减去每个通道的平均值
        baseline = np.mean(raw, axis=1, keepdims=True)
        corrected = raw - baseline
        
        # 零相位滤波，不会延迟信号
        filtered = signal.filtfilt(self.b, self.a, corrected, axis=1)
        return filtered
    
    def cut_epochs(self, eeg, events, pre=0.2, post=0.8):
        """
        按事件时间切片
        比如看到一个动作，就把动作前0.2秒到后0.8秒切出来
        """
        pre_n = int(pre * self.fs)
        post_n = int(post * self.fs)
        epochs = []
        
        for t in events:
            center = int(t * self.fs)
            s = center - pre_n
            e = center + post_n
            if s >= 0 and e < eeg.shape[1]:
                epochs.append(eeg[:, s:e])
                
        return np.array(epochs)
