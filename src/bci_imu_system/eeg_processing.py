"""
EEG Signal Processing Pipeline
[CyberLife Modified] 2026-10 | 适配排雷蜘蛛项目，优化EEG信号预处理流程 | 软著标记
Reference: MNE-Python, BraindaCode open-source BCI frameworks
"""
import numpy as np
from scipy import signal

class EEGPreprocessor:
    """
    EEG signal preprocessing pipeline for non-invasive BCI.
    Steps: baseline correction -> bandpass filter -> artifact removal -> epoch extraction
    """
    def __init__(self, sampling_freq: int = 250, 
                 lowcut: float = 8.0, highcut: float = 30.0):
        self.fs = sampling_freq
        self.lowcut = lowcut
        self.highcut = highcut
        self._build_filter()
        
    def _build_filter(self):
        """Design 4th order Butterworth bandpass filter"""
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        self.b, self.a = signal.butter(4, [low, high], btype='band')
        
    def process(self, raw_eeg: np.ndarray) -> np.ndarray:
        """
        Process raw EEG signal.
        raw_eeg: (n_channels, n_samples) array
        Returns: filtered EEG
        """
        # Baseline correction
        baseline = np.mean(raw_eeg, axis=1, keepdims=True)
        corrected = raw_eeg - baseline
        
        # Bandpass filter (zero-phase)
        filtered = signal.filtfilt(self.b, self.a, corrected, axis=1)
        
        return filtered
    
    def extract_epochs(self, eeg: np.ndarray, event_times: list,
                      pre_event: float = 0.2, post_event: float = 0.8) -> np.ndarray:
        """
        Extract epochs around event markers.
        event_times: list of event timestamps in seconds
        Returns: (n_epochs, n_channels, n_samples) array
        """
        epochs = []
        pre_samples = int(pre_event * self.fs)
        post_samples = int(post_event * self.fs)
        
        for t in event_times:
            center = int(t * self.fs)
            start = center - pre_samples
            end = center + post_samples
            if start >= 0 and end < eeg.shape[1]:
                epochs.append(eeg[:, start:end])
                
        return np.array(epochs)
