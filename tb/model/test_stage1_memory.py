from __future__ import annotations

from model.memory.fifo_reference import Sram1PReference, Sram2PReference, StreamItem, SyncFIFOReference


def test_fifo_single_and_continuous_transactions():
    fifo = SyncFIFOReference(depth=4)
    for value in range(4):
        wr_ready, rd_valid, rd_item = fifo.cycle(True, StreamItem(value, meta=value + 10), False)
        assert wr_ready
        assert not rd_valid or rd_item is not None
    assert fifo.full
    assert fifo.occupancy == 4

    seen = []
    for _ in range(4):
        wr_ready, rd_valid, rd_item = fifo.cycle(False, None, True)
        assert wr_ready
        assert rd_valid
        assert rd_item is not None
        seen.append((rd_item.data, rd_item.meta))
    assert seen == [(0, 10), (1, 11), (2, 12), (3, 13)]
    assert fifo.empty


def test_fifo_full_allows_same_cycle_pop_push_without_reordering():
    fifo = SyncFIFOReference(depth=2)
    assert fifo.cycle(True, StreamItem(1), False)[0]
    assert fifo.cycle(True, StreamItem(2), False)[0]
    assert fifo.full

    wr_ready, rd_valid, rd_item = fifo.cycle(True, StreamItem(3), True)
    assert wr_ready
    assert rd_valid
    assert rd_item == StreamItem(1)
    assert fifo.full

    assert fifo.cycle(False, None, True)[2] == StreamItem(2)
    assert fifo.cycle(False, None, True)[2] == StreamItem(3)
    assert fifo.empty


def test_fifo_random_backpressure_preserves_order():
    fifo = SyncFIFOReference(depth=5, almost_full_threshold=4)
    expected = []
    observed = []
    next_value = 0

    for cycle in range(100):
        wr_valid = cycle % 3 != 0
        rd_ready = cycle % 4 != 1
        item = StreamItem(next_value, meta=next_value ^ 0x55, last=(next_value % 7 == 0))
        wr_ready, rd_valid, rd_item = fifo.cycle(wr_valid, item, rd_ready)
        if wr_valid and wr_ready:
            expected.append(item)
            next_value += 1
        if rd_valid and rd_ready:
            observed.append(rd_item)
        assert 0 <= fifo.occupancy <= fifo.depth
        assert fifo.almost_full == (fifo.occupancy >= 4)

    while not fifo.empty:
        _, rd_valid, rd_item = fifo.cycle(False, None, True)
        assert rd_valid
        observed.append(rd_item)

    assert observed == expected


def test_sram_1p_read_latency_and_write_response_behavior():
    sram = Sram1PReference(depth=8, data_mask=0xFFFF)
    req_ready, rsp_valid, rsp_data = sram.cycle(True, True, 3, wdata=0x1234)
    assert req_ready
    assert not rsp_valid
    assert rsp_data is None

    req_ready, rsp_valid, rsp_data = sram.cycle(True, False, 3)
    assert req_ready
    assert not rsp_valid

    req_ready, rsp_valid, rsp_data = sram.cycle(False, False, 0)
    assert req_ready
    assert rsp_valid
    assert rsp_data == 0x1234


def test_sram_2p_read_first_same_address_collision():
    sram = Sram2PReference(depth=8, data_mask=0xFFFF)
    sram.cycle(True, 2, 0xAAAA, False, 0)
    sram.cycle(False, 0, 0, True, 2)
    _, _, rsp_valid, rsp_data = sram.cycle(False, 0, 0, False, 0)
    assert rsp_valid
    assert rsp_data == 0xAAAA

    sram.cycle(True, 2, 0x5555, True, 2)
    _, _, rsp_valid, rsp_data = sram.cycle(False, 0, 0, False, 0)
    assert rsp_valid
    assert rsp_data == 0xAAAA

    sram.cycle(False, 0, 0, True, 2)
    _, _, rsp_valid, rsp_data = sram.cycle(False, 0, 0, False, 0)
    assert rsp_valid
    assert rsp_data == 0x5555
