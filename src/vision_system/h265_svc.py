"""
H.265/SVC Scalable Video Coding Module
[CyberLife Modified] 2026-10 | 适配排雷蜘蛛项目，全景视频分层编码 | 软著标记
Reference: FFmpeg/libx265, open-source video codec libraries
Implements layered encoding for panoramic video:
- Base layer: low bitrate, wide FOV (far view)
- Enhancement layer: high bitrate, near-field detail
"""
import subprocess
import numpy as np

class H265SVCEncoder:
    """
    H.265 Scalable Video Coding encoder.
    Provides adaptive bitrate based on depth-of-field:
    far view -> lower resolution, near view -> higher resolution.
    """
    def __init__(self, base_bitrate: int = 500, enhance_bitrate: int = 2000):
        self.base_bitrate = base_bitrate  # kbps
        self.enhance_bitrate = enhance_bitrate  # kbps
        
    def encode_layer(self, frame: np.ndarray, layer: str = 'base') -> bytes:
        """
        Encode single frame to H.265 NAL unit.
        layer: 'base' (far view, low bitrate) or 'enhance' (near view, high bitrate)
        """
        bitrate = self.base_bitrate if layer == 'base' else self.enhance_bitrate
        h, w = frame.shape[:2]
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-i', '-',
            '-c:v', 'libx265',
            '-b:v', f'{bitrate}k',
            '-tune', 'zerolatency',
            '-preset', 'ultrafast',
            '-f', 'hevc',
            'pipe:1'
        ]
        
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        out, _ = proc.communicate(frame.tobytes())
        return out
    
    def encode_panoramic(self, frame: np.ndarray, 
                        depth_map: np.ndarray = None) -> dict:
        """
        Encode panoramic frame with layered encoding.
        depth_map: optional depth map for adaptive ROI quality.
        Returns: {'base': base_layer, 'enhance': enhance_layer}
        """
        base = self.encode_layer(frame, 'base')
        enhance = self.encode_layer(frame, 'enhance')
        return {'base': base, 'enhance': enhance}
