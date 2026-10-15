# FEC前向纠错传输
# 参考了OpenFEC和WebRTC的FEC机制，做了简化版
# 弱网环境下丢包了也能恢复，不用重传
import numpy as np

class FECEncoder:
    """
    FEC前向纠错编码器
    就是把几个包异或一下生成一个校验包
    丢一个包的话，用校验包就能恢复出来
    比重传快，适合实时视频
    """
    def __init__(self, group_size=4):
        self.n = group_size  # 每4个数据包加1个校验包
        
    def encode(self, packets):
        """
        对一组数据包编码
        packets是n个数据包的列表
        返回加了校验包的列表
        """
        if len(packets) < self.n:
            return packets
            
        # 异或所有包生成校验包
        parity = packets[0]
        for p in packets[1:]:
            parity = bytes(a ^ b for a, b in zip(parity, p))
            
        return list(packets) + [parity]
    
    def decode(self, received):
        """
        解码，如果丢了一个包就恢复
        received是收到的包，可能缺一个
        """
        if len(received) == self.n + 1:
            return received[:-1]  # 全收到了，去掉校验包
        
        # 丢了一个包，用校验包恢复
        if len(received) == self.n:
            # 找出丢的是哪个，异或其他所有包
            recovered = bytes(self.n)
            for p in received:
                recovered = bytes(a ^ b for a, b in zip(recovered, p))
            return list(received) + [recovered]
            
        return received


class VideoStream:
    """
    视频流传输封装
    就是把编码好的视频包发出去，带FEC保护
    """
    def __init__(self):
        self.fec = FECEncoder(group_size=4)
        self.buffer = []
        
    def send_frame(self, frame_data):
        """
        发一帧数据
        """
        self.buffer.append(frame_data)
        if len(self.buffer) >= 4:
            packets = self.fec.encode(self.buffer)
            # 实际这里发网络包
            self.buffer = []
            return packets
        return None
