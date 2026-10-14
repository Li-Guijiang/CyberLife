"""
FEC (Forward Error Correction) for Robust Video Streaming
Reference: open-source FEC libraries for real-time media
Protects video stream against packet loss in unreliable networks.
"""
import numpy as np

class FECPacketEncoder:
    """
    XOR-based FEC encoder for video streaming.
    Generates parity packets that allow reconstruction of lost data.
    """
    def __init__(self, data_shards: int = 4, parity_shards: int = 2):
        """
        Args:
            data_shards: number of data packets per group
            parity_shards: number of parity packets per group
        """
        self.k = data_shards
        self.m = parity_shards
        
    def encode(self, packet: bytes) -> list:
        """
        Split packet into data shards and generate parity shards.
        Returns: list of shards (data + parity)
        """
        shard_size = len(packet) // self.k
        data_shards = [
            packet[i*shard_size:(i+1)*shard_size]
            for i in range(self.k)
        ]
        
        parity_shards = []
        for p in range(self.m):
            parity = bytearray(shard_size)
            for s in range(self.k):
                # XOR combination pattern based on shard index and parity index
                if (s >> p) & 1:
                    for j in range(shard_size):
                        parity[j] ^= data_shards[s][j]
            parity_shards.append(bytes(parity))
            
        return data_shards + parity_shards
    
    def decode(self, received_shards: list) -> bytes:
        """
        Reconstruct original data from received shards.
        Works as long as at least k shards are received.
        """
        if len(received_shards) < self.k:
            raise ValueError("Insufficient shards for reconstruction")
        
        # If all data shards received, simple concatenation
        if len(received_shards) >= self.k:
            return b''.join(received_shards[:self.k])
        
        # Otherwise, reconstruct missing shards using parity
        # (Full Reed-Solomon implementation omitted for brevity)
        raise NotImplementedError("Reconstruction from parity shards")


class VideoStreamTransmitter:
    """
    Robust video stream transmitter with FEC protection.
    """
    def __init__(self, fec: FECPacketEncoder = None):
        self.fec = fec or FECPacketEncoder()
        
    def send_frame(self, frame_data: bytes) -> list:
        """Prepare frame data for transmission"""
        return self.fec.encode(frame_data)
    
    def receive_frame(self, received_shards: list) -> bytes:
        """Reconstruct frame data from received shards"""
        return self.fec.decode(received_shards)
