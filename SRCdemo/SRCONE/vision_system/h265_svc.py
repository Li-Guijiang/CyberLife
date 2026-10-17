# H.265-SVC分层编码
# 参考了FFmpeg和x265的实现，做了简化封装
# 远景用低码率，近景用高码率，这样带宽不够的时候至少能看清大概
import numpy as np

class H265SVC:
    """
    H.265可伸缩编码
    分两层：
    - 基础层：低码率，看全景远景
    - 增强层：高码率，看近处细节
    网络不好的时候就只传基础层，好的时候两层一起传
    """
    def __init__(self, base_kbps=500, enhance_kbps=2000):
        self.base_bitrate = base_kbps  # 基础层码率
        self.enhance_bitrate = enhance_kbps  # 增强层码率
        
    def encode_base(self, frame):
        """
        编码基础层（远景全景）
        分辨率低一点，码率低
        """
        h, w = frame.shape[:2]
        # 实际用FFmpeg/x265编码，这里是接口
        # cmd = f"ffmpeg -i - -c:v libx265 -b:v {self.base_bitrate}k -f h265 pipe:1"
        return b""  # 编码后的数据
    
    def encode_enhance(self, frame):
        """
        编码增强层（近景细节）
        分辨率高，码率高，细节清楚
        """
        # 近景ROI区域编码
        return b""
    
    def get_layer_config(self, network_quality=0.5):
        """
        根据网络质量选传哪层
        network_quality是0-1，0是网很差，1是很好
        """
        if network_quality < 0.3:
            return "base_only"
        elif network_quality < 0.7:
            return "base + enhance_low"
        else:
            return "base + enhance_full"
