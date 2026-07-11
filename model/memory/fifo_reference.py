"""Cycle reference models for Stage 1 FIFO and SRAM wrappers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class StreamItem:
    data: int
    meta: int = 0
    last: bool = False


class SyncFIFOReference:
    def __init__(self, depth: int, almost_full_threshold: int | None = None) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.depth = depth
        self.almost_full_threshold = depth - 1 if almost_full_threshold is None else almost_full_threshold
        if not 0 <= self.almost_full_threshold <= depth:
            raise ValueError("almost_full_threshold out of range")
        self.queue: deque[StreamItem] = deque()

    @property
    def full(self) -> bool:
        return len(self.queue) == self.depth

    @property
    def empty(self) -> bool:
        return not self.queue

    @property
    def occupancy(self) -> int:
        return len(self.queue)

    @property
    def almost_full(self) -> bool:
        return len(self.queue) >= self.almost_full_threshold

    def cycle(
        self,
        wr_valid: bool,
        wr_item: StreamItem | None,
        rd_ready: bool,
    ) -> tuple[bool, bool, StreamItem | None]:
        rd_valid = not self.empty
        wr_ready = (not self.full) or (rd_valid and rd_ready)
        rd_item = self.queue[0] if rd_valid else None

        pop = rd_valid and rd_ready
        push = wr_valid and wr_ready

        if pop:
            self.queue.popleft()
        if push:
            if wr_item is None:
                raise ValueError("wr_item is required when wr_valid is true and wr_ready is true")
            self.queue.append(wr_item)
        if len(self.queue) > self.depth:
            raise AssertionError("FIFO overflow")
        return wr_ready, rd_valid, rd_item


class Sram1PReference:
    def __init__(self, depth: int, data_mask: int) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.depth = depth
        self.data_mask = data_mask
        self.mem = [0 for _ in range(depth)]
        self.pending_read: int | None = None

    def cycle(
        self,
        req_valid: bool,
        req_write: bool,
        addr: int,
        wdata: int = 0,
        rsp_ready: bool = True,
    ) -> tuple[bool, bool, int | None]:
        if not 0 <= addr < self.depth:
            raise ValueError("address out of range")
        rsp_valid = self.pending_read is not None
        rsp_data = self.pending_read
        req_ready = rsp_ready or not rsp_valid

        if rsp_ready:
            self.pending_read = None
        if req_valid and req_ready:
            if req_write:
                self.mem[addr] = wdata & self.data_mask
            else:
                self.pending_read = self.mem[addr]
        return req_ready, rsp_valid, rsp_data


class Sram2PReference:
    """1-read/1-write SRAM model with READ_LATENCY=1 and read-first collision."""

    def __init__(self, depth: int, data_mask: int) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.depth = depth
        self.data_mask = data_mask
        self.mem = [0 for _ in range(depth)]
        self.pending_read: int | None = None

    def cycle(
        self,
        wr_valid: bool,
        wr_addr: int,
        wr_data: int,
        rd_valid: bool,
        rd_addr: int,
        rsp_ready: bool = True,
    ) -> tuple[bool, bool, bool, int | None]:
        for addr in (wr_addr, rd_addr):
            if not 0 <= addr < self.depth:
                raise ValueError("address out of range")

        rsp_valid = self.pending_read is not None
        rsp_data = self.pending_read
        wr_ready = True
        rd_ready = rsp_ready or not rsp_valid

        old_rd_data = self.mem[rd_addr] if rd_valid and rd_ready else None
        if rsp_ready:
            self.pending_read = None
        if rd_valid and rd_ready:
            self.pending_read = old_rd_data
        if wr_valid and wr_ready:
            self.mem[wr_addr] = wr_data & self.data_mask
        return wr_ready, rd_ready, rsp_valid, rsp_data
