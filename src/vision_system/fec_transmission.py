# -*- coding: utf-8 -*-
"""弱网视频包的前向纠错与重组。

发送端把一帧切成定长数据包，每 n 个包算出一个校验包；接收端少一个包时，
用校验包与其余数据包异或即可还原，不必等重传。异或只能补一个包，
因此滑窗按 n 包一组，组内丢两个及以上时整组交给上层丢弃并由编码器
在下一个关键帧补齐。

包头字段
--------
magic       固定 0xCAFE，用于丢弃非本协议数据
frame_id    帧序号
group_id    组序号，帧内从 0 递增
index       组内数据包序号；0xFF 表示本包是校验包
count       本组数据包个数（末组可能不足 n）
flags       位 0 置位表示本组是该帧的末组
length      本包负载字节数；校验包此处为最长负载长度

校验包额外携带本组各数据包的原始长度表，这样丢包后仍能知道
缺失包应还原成多少字节——若把长度写在数据包自身头部，包一丢长度就丢了，
这正是旧版实现无法正确恢复的原因。

与标准 FEC 的关系
----------------
思路与 WebRTC 的 RED/ULPFEC 一致：按组异或、带组序号与长度表，
不同之处是这里只做定长切包与单组单校验，代码量小，便于在
本机离线复现验证。
"""
from __future__ import annotations

import struct
from typing import Dict, List, Optional

MAGIC = 0xCAFE
PARITY_INDEX = 0xFF
HEADER = struct.Struct("<IHBBBBH")     # magic, frame_id, group_id, index, count, flags, length
FLAG_LAST_GROUP = 0x01

_T = HEADER.size


def _len_table(count: int) -> struct.Struct:
    return struct.Struct("<%dH" % count)


class FECEncoder:
    """定长切包 + 分组异或校验。"""

    def __init__(self, group_size: int = 4, max_payload: int = 1024):
        if group_size < 2:
            raise ValueError("group_size 至少为 2，否则校验包没有意义")
        self.n = int(group_size)
        self.max_payload = int(max_payload)

    def split(self, data: bytes) -> List[bytes]:
        """把一帧切成定长块，空数据也产出 1 个空块。"""
        if not data:
            return [b""]
        return [data[i:i + self.max_payload] for i in range(0, len(data), self.max_payload)]

    def encode_frame(self, frame_id: int, data: bytes) -> List[bytes]:
        """编码一帧，返回数据包与校验包交错的列表。"""
        chunks = self.split(data)
        out: List[bytes] = []
        for group_id, start in enumerate(range(0, len(chunks), self.n)):
            part = chunks[start:start + self.n]
            count = len(part)
            last = 1 if start + count >= len(chunks) else 0
            flags = FLAG_LAST_GROUP if last else 0
            for idx, body in enumerate(part):
                out.append(HEADER.pack(MAGIC, frame_id, group_id, idx, count, flags, len(body)) + body)
            # 单包组再补一个校验包等于原包重传，没有意义，跳过
            if count >= 2:
                max_len = max(len(b) for b in part)
                parity = bytearray(max_len)
                for body in part:
                    for i, byte in enumerate(body):
                        parity[i] ^= byte
                head = HEADER.pack(MAGIC, frame_id, group_id, PARITY_INDEX, count, flags, max_len)
                out.append(head + _len_table(count).pack(*[len(b) for b in part]) + bytes(parity))
        return out

    def overhead(self, data: bytes) -> float:
        """校验包带来的字节开销比例（含每包定长头）。"""
        raw = sum(len(p) for p in self.encode_frame(0, data))
        return raw / float(max(1, len(data))) - 1.0


def parse(packet: bytes) -> Optional[dict]:
    """解析单个包，格式不符返回 None。"""
    if len(packet) < _T:
        return None
    magic, fid, gid, idx, count, flags, length = HEADER.unpack_from(packet)
    if magic != MAGIC or count == 0:
        return None
    off = _T
    lengths = None
    if idx == PARITY_INDEX:
        if count < 1 or len(packet) < off + 2 * count:
            return None
        lengths = list(_len_table(count).unpack_from(packet, off))
        off += 2 * count
    body = packet[off:off + length]
    if len(body) != length:
        return None
    return {"frame_id": fid, "group_id": gid, "index": idx, "count": count,
            "flags": flags, "length": length, "lengths": lengths, "body": body}


class FECReceiver:
    """收包、按组纠错、按帧重组，并记录统计量。"""

    def __init__(self, group_size: int = 4):
        self.n = int(group_size)
        self.groups: Dict[int, Dict[int, Dict[int, dict]]] = {}   # frame -> group -> index -> 包
        self.dropped: set = set()
        self.completed: set = set()
        self.stats = {"packets": 0, "recovered": 0, "unrecoverable_groups": 0,
                      "frames_done": 0, "frames_dropped": 0}

    # -------------------------------------------------- 收包
    def feed(self, packet: bytes) -> Optional[bytes]:
        """喂入一个包；若因此凑齐一帧，返回该帧的完整字节。"""
        info = parse(packet)
        if info is None:
            return None
        self.stats["packets"] += 1
        fid, gid = info["frame_id"], info["group_id"]
        # 整帧已判废、或已重组完毕时，迟到与重复的包（含末组校验包）直接丢弃
        if fid in self.dropped or fid in self.completed:
            return None
        bucket = self.groups.setdefault(fid, {}).setdefault(gid, {})
        bucket[info["index"]] = info
        self._try_recover(fid, gid)
        return self._try_assemble(fid)

    def _drop_frame(self, fid: int) -> None:
        if self.groups.pop(fid, None) is not None:
            self.dropped.add(fid)
            self.stats["frames_dropped"] += 1

    def _try_recover(self, fid: int, gid: int) -> None:
        """组内缺一个包时用校验包还原；缺两个及以上整组判废。"""
        bucket = self.groups[fid][gid]
        parity = bucket.get(PARITY_INDEX)
        if parity is None:
            return
        data = {i: p for i, p in bucket.items() if i != PARITY_INDEX}
        count = parity["count"]
        missing = [i for i in range(count) if i not in data]
        if len(missing) != 1:
            if len(missing) > 1:
                self.stats["unrecoverable_groups"] += 1
                self._drop_frame(fid)
            return
        lost = missing[0]
        buf = bytearray(parity["body"])
        for i, p in data.items():
            for j, byte in enumerate(p["body"]):
                buf[j] ^= byte
        need = parity["lengths"][lost]
        if need > len(buf):
            self.stats["unrecoverable_groups"] += 1
            self._drop_frame(fid)
            return
        bucket[lost] = {"index": lost, "count": count, "flags": parity["flags"],
                        "body": bytes(buf[:need]), "recovered": True}
        self.stats["recovered"] += 1

    # -------------------------------------------------- 重组
    def _try_assemble(self, fid: int) -> Optional[bytes]:
        groups = self.groups.get(fid)
        if not groups:
            return None
        last = None
        for gid, bucket in groups.items():
            sample = next(iter(bucket.values()))
            if sample["flags"] & FLAG_LAST_GROUP:
                last = gid
        if last is None:
            return None
        for gid in range(last + 1):
            bucket = groups.get(gid)
            if bucket is None:
                return None
            count = next(iter(bucket.values()))["count"]
            if any(i not in bucket for i in range(count)):
                return None
        pieces = []
        for gid in range(last + 1):
            bucket = groups[gid]
            count = next(iter(bucket.values()))["count"]
            for i in range(count):
                pieces.append(bucket[i]["body"])
        self.groups.pop(fid, None)
        self.stats["frames_done"] += 1
        self.completed.add(fid)
        if len(self.completed) > 512:              # 只保留最近的完成记录，避免无界增长
            keep = max(self.completed)
            self.completed = {keep}
        return b"".join(pieces)

    def dropped_frames(self) -> List[int]:
        """仍缺包的帧号，供上层请求关键帧。"""
        return sorted(self.groups.keys())

    def finish(self) -> dict:
        """播放截止：把仍未凑齐的帧判废并回收。

        实时流里不会无限等一个缺包的帧，解码端有播放期限；到期未凑齐就丢弃，
        由编码端在下一个关键帧补齐。这里用显式调用代替超时计时。
        """
        for fid in list(self.groups.keys()):
            self._drop_frame(fid)
        return self.report()

    def report(self) -> dict:
        s = dict(self.stats)
        s["pending_frames"] = len(self.groups)
        return s


def simulate_loss(packets: List[bytes], drop: set) -> List[bytes]:
    """按索引集合丢包，用于离线复现弱网。"""
    return [p for i, p in enumerate(packets) if i not in drop]
