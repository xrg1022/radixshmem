# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RadixServer + RadixClient over the gRPC channel, standalone (no etcd, no RDMA):
the server starts waiting, a client brings the geometry, clients attach on use."""

import os
import threading
import time
import uuid
from contextlib import contextmanager

import numpy as np
import pytest

import radixshmem
from radixshmem import Geometry, RadixClient, RadixServer, ServerConfig

pytestmark = pytest.mark.skipif(not hasattr(radixshmem, "_data"), reason="radixshmem._data not built")

OK = radixshmem.InsertError.OK
FULL, SWA = radixshmem.ComponentType.FULL, radixshmem.ComponentType.SWA


@pytest.fixture
def cfg(tmp_path):
    """cfg(**server_kw) -> ServerConfig with a unique name and socket."""
    tag = uuid.uuid4().hex[:8]

    def make(**kw):
        kw.setdefault("background_evict_ratio", 0.0)
        return ServerConfig(name=make.name, endpoint=make.endpoint, **kw)

    make.name, make.endpoint = f"/py_rpc_{tag}", f"unix://{tmp_path}/{tag}.sock"
    return make


def data(**kw):
    """Data-mode server settings: a 4 MiB budget, 64 B alignment so small test
    slots keep their size."""
    return {"data_bytes": 4 << 20, "slot_align": 64, "prefault": False, **kw}


def geometry(**kw):
    return Geometry(**{"block_size": 4, "full_slot_bytes": 8192, **kw})


def client(c, geometry=None, **kw):
    return RadixClient(c.name, geometry, endpoint=c.endpoint, **kw)


def up(c, geo, timeout=30):
    """start -> a client brings the geometry -> wait_ready, the way an engine's
    bootstrap does it; the configuring client is closed again."""
    srv = RadixServer(c).start()
    try:
        cl = client(c, geo)
        cl.wait_ready(timeout)
        cl.close()
    except BaseException:
        srv.close()
        raise
    return srv


def hashes(n, base=0):
    return radixshmem.compute_hashes(np.arange(base, base + 4 * n, dtype=np.int64), 4)


def test_client_configures_a_waiting_server_and_attaches_when_ready(cfg):
    c = cfg(**data(swa_ratio=8 * 1024 / (4 << 20)))           # room for 8 SWA slots of 1 KiB
    srv = RadixServer(c).start()
    try:
        assert srv.mode == "waiting" and srv.geometry is None
        # a client exists before the server is ready; it sees the state and cannot use the index yet
        a = client(c)
        assert a.info.mode == "waiting" and a.info.data_plane and a.geometry is None
        assert a.info.last_error == "" and a.info.data_bytes == 4 << 20
        with pytest.raises(radixshmem.ServerNotReady) as ei:
            a.query(hashes(1))
        assert ei.value.mode == "waiting"
        with pytest.raises(radixshmem.ServerNotReady):
            a.store
        with pytest.raises(TimeoutError, match="waiting"):
            a.wait_ready(0.2)

        # the geometry rides with a client; the constructor returns before the regions exist
        geo = geometry(swa_slot_bytes=1024, swa_window_blocks=2)
        b = client(c, geo)
        assert b.info.mode in ("configuring", "ready")
        assert b.info.geometry["pools"]["swa"] == {"slot_bytes": 1024, "num_slots": 8, "window_blocks": 2}
        info = b.wait_ready(30)
        assert info.mode == "ready" and srv.mode == "ready"
        assert a.status().mode == "ready"
        # the same geometry again is a no-op (dict form too); another one is refused
        d = client(c, geo.to_dict())
        assert d.info.mode == "ready"
        d.close()
        assert b.configure(geo).mode == "ready"
        with pytest.raises(radixshmem.GeometryMismatch, match="block_size"):
            client(c, geometry(block_size=8, swa_slot_bytes=1024, swa_window_blocks=2))
        with pytest.raises(ValueError, match="num_slots"):
            client(c, geometry(full_slots=10))

        assert a.geometry == srv.geometry
        assert a.geometry["pools"]["full"]["num_slots"] == ((4 << 20) - 8 * 1024) // 8192
        assert a.geometry["register_chunk_tokens"] == 4096
        assert a.block_size() == 4 and int(a.component_config.swa_window_blocks) == 2   # attaches now
        assert a.store.pool(FULL).slot_bytes == 8192 and a.store.pool(SWA).num_slots == 8
        assert b.geometry == a.geometry and b.info.index_name == c.name

        h = hashes(2)
        slots = a.allocate_slots(2)
        assert a.insert(h, slots).error == OK
        a.slot_view(slots[0])[:4] = b"abcd"
        assert bytes(b.slot_view(slots[0])[:4]) == b"abcd"

        job = b.pull_async(h)
        r = b.wait(job, 5)
        assert job.done() and r.common_hit == 2 and r.remote_blocks == 0
        assert list(r.full_slots) == list(slots)
        assert bool(r.finalize)
        r.finalize()
        assert not bool(r.finalize)
        assert b.check_health() == 0
        a.close()
        b.close()
    finally:
        srv.close()
    assert srv.mode == "closed"
    assert not os.path.exists(f"/dev/shm{c.name}")
    assert not os.path.exists(f"/dev/shm{c.name}_data")
    assert not os.path.exists(c.endpoint[len("unix://"):])


def test_index_only_takes_the_counts_from_the_client(cfg):
    c = cfg()
    srv = up(c, geometry(full_slots=16))            # bytes are ignored, published as 0
    try:
        cl = client(c)
        assert cl.info.mode == "ready" and not cl.info.data_plane
        assert cl.geometry["pools"] == {"full": {"slot_bytes": 0, "num_slots": 16}}
        assert cl.block_size() == 4 and cl.mempool_total() == 16
        h = hashes(2)
        assert cl.insert(h, cl.allocate_slots(2)).error == OK
        assert cl.query(h).common_hit == 2
        with pytest.raises(radixshmem.DataPlaneUnavailable):
            cl.pull_async(h)
        with pytest.raises(radixshmem.DataPlaneUnavailable):
            cl.slot_view(0)
        with pytest.raises(radixshmem.DataPlaneUnavailable):
            srv.submit([(1, 0, [0], [0], radixshmem.OP_READ)], 100)
        with pytest.raises(radixshmem.DataPlaneUnavailable):
            cl.push_async(h, rank=1)
        with pytest.raises(ValueError, match="index-only"):
            client(c, geometry())                                    # no counts
        with pytest.raises(radixshmem.GeometryMismatch, match="num_slots"):
            cl.configure(geometry(full_slots=32))
        cl.close()
    finally:
        srv.close()
    assert not os.path.exists(f"/dev/shm{c.name}")


def test_failed_configuration_returns_to_waiting_with_the_error(cfg):
    c = cfg(**data())
    srv = RadixServer(c).start()
    try:
        def boom():
            raise RuntimeError("no memory for you")

        srv._bring_up = boom
        cl = client(c, geometry())
        assert cl.info.mode in ("configuring", "waiting")
        with pytest.raises(RuntimeError, match="no memory for you"):
            cl.wait_ready(30)
        info = cl.status()
        assert info.mode == "waiting" and "no memory for you" in info.last_error and info.geometry is None
        with pytest.raises(radixshmem.ServerNotReady, match="no memory for you"):
            cl.query(hashes(1))
        assert not os.path.exists(f"/dev/shm{c.name}_data")
        del srv._bring_up                                   # the next configure starts over
        assert cl.configure(geometry()).mode in ("configuring", "ready")
        assert cl.wait_ready(30).mode == "ready"
        assert cl.status().last_error == ""
        assert cl.query(hashes(1)).common_hit == 0          # attaches now
        cl.close()
    finally:
        srv.close()


def test_cli_builds_the_server_config():
    from radixshmem import cli

    c = cli.build_config(cli.parse_args([
        "--name", "/x", "--data-bytes", "64M", "--swa-ratio", "0.25", "--transfer-dev", "mlx5_0,mlx5_1",
        "--transfer-dev", "mlx5_2", "--no-prefault", "--expected-min-nodes", "2", "--rpc-address", "0.0.0.0"]))
    assert c.name == "/x" and c.data_bytes == 64 << 20 and c.swa_ratio == 0.25
    assert c.transfer_devices == ["mlx5_0", "mlx5_1", "mlx5_2"] and not c.prefault and c.distributed
    assert cli.build_config(cli.parse_args([])).index_only
    with pytest.raises(ValueError, match="--swa-ratio"):
        cli.build_config(cli.parse_args(["--swa-ratio", "0.1"]))
    with pytest.raises(ValueError, match="ratio"):
        cli.build_config(cli.parse_args(["--data-bytes", "1M", "--swa-ratio", "0.6", "--mamba-ratio", "0.5"]))


def test_local_read_only_client_reads_index_and_store_but_never_writes(cfg):
    c = cfg(**data())
    srv = up(c, geometry())
    try:
        w = client(c)
        h = hashes(3)
        slots = w.allocate_slots(3)
        assert w.insert(h, slots).error == OK
        w.slot_view(slots[1])[:4] = b"ro!!"

        ro = client(c, local_read_only=True)
        assert ro.local_read_only and not ro.is_distributed() and ro.rank() == 0
        assert ro.info.world_size == 1 and ro.geometry == w.geometry
        assert bytes(ro.slot_view(slots[1])[:4]) == b"ro!!"
        q = ro.query(h, lock=True)
        assert q.common_hit == 3 and list(q.full_fragments[0][2]) == list(slots)
        q.finalize()
        assert ro.query(h, local_only=False, update_meta=False).common_hit == 3
        for call in (lambda: ro.pull_async(h), lambda: ro.allocate_slots(1),
                     lambda: ro.insert(h, slots), lambda: ro.recycle_slots(slots), ro.reset):
            with pytest.raises(RuntimeError):
                call()
        assert w.query(h, update_meta=False).common_hit == 3     # the refused reset changed nothing
        assert ro.check_health() == 0
        ro.close()

        core = radixshmem._core.RadixClient(c.name, local_read_only=True)     # index-only callers
        assert not core.is_distributed()
        assert core.query(h, update_meta=False).common_hit == 3
        with pytest.raises(RuntimeError):
            core.allocate_slots(1)
        del core
        w.close()
    finally:
        srv.close()


class _Q:
    """Scripted cluster query: local prefix + a rank-7 suffix; `pin` is a real
    locked local query so the prefix is pinned like the C++ query would."""

    status = 0
    swa_source_rank, swa_slots, mamba = -1, np.empty(0, np.int32), None

    def __init__(self, local, remote_off, remote_slots, common_hit):
        self.full_fragments = [(0, 0, local), (7, remote_off, remote_slots)]
        self.common_hit = common_hit
        self.pin = None

    def finalize(self):
        if self.pin is not None:
            self.pin.finalize()


def scripted_index(real, script):
    """Index whose cluster query returns script(hashes) with the real local pin."""

    class Index:
        def __getattr__(self, name):
            return getattr(real, name)

        def query(self, hs, **kw):
            if kw.get("local_only"):
                return real.query(hs, **kw)
            q = script(hs)
            q.pin = real.query(hs, local_only=True, lock=True, update_meta=False)
            return q

    return Index()


class _Transport:
    """Complete newest tickets first; script whole/per-item failures, lost tickets and busy submissions."""

    def __init__(self):
        self.items, self.submitted, self.n, self.order = {}, [], 0, []
        self.fail, self.lose, self.busy = set(), set(), False
        self.fail_items = {}

    def submit(self, items, timeout_ms):
        if self.busy:
            raise radixshmem.ServerBusy("full")
        self.n += 1
        self.items[self.n] = items
        self.submitted.append(items)
        return self.n

    def wait_any(self, tickets, timeout_ms):
        time.sleep(0.002)
        pick = sorted(t for t in tickets if t not in self.lose)[-2:]
        self.order += pick
        done = {}
        for t in pick:
            bad = self.fail_items.get(t, set())
            done[t] = [(ok, 0, 1, "" if ok else "boom")
                       for ok in (t not in self.fail and i not in bad
                                  for i in range(len(self.items.pop(t))))]
        return {"done": done, "unknown": [t for t in tickets if t in self.lose]}

    def close(self):
        pass


def test_pull_async_promotes_remote_suffixes_out_of_order(cfg):
    """The cluster query is scripted and the server is a fake; allocation,
    promotion and pinning run on the real index for eight concurrent jobs."""
    c = cfg(**data(data_bytes=1 << 20))
    srv = up(c, geometry())
    try:
        cl = client(c)
        real, scripted = cl.index, {}
        transport = _Transport()
        cl._index, cl._t = scripted_index(real, lambda hs: scripted[int(hs[0])]), transport

        def prefix(k, local_blocks, total=8):
            h = hashes(total, base=1000 * k)
            local = cl.allocate_slots(local_blocks)
            assert cl.insert(h[:local_blocks], local).error == OK
            remote = np.arange(100 * k, 100 * k + total - local_blocks, dtype=np.int32)
            scripted[int(h[0])] = _Q(local, local_blocks, remote, total)
            return h, local, remote

        prefixes = [prefix(k, 3) for k in range(8)]
        nodes = real.total_radix_nodes()
        jobs = [cl.pull_async(h) for h, _, _ in prefixes]
        assert len({j.job_id for j in jobs}) == 8
        done = cl.wait_any(jobs, timeout=10)
        assert done and all(j.done() for j in done)
        for (h, local, remote), job in zip(prefixes, jobs):
            r = job.wait(10)
            assert r.common_hit == 8 and r.remote_blocks == 5 and r.source_rank == 7
            assert list(r.full_slots[:3]) == list(local)
            r.finalize()
            assert real.query(h, local_only=True).common_hit == 8
        assert real.total_radix_nodes() == nodes       # promoted suffixes merged into the unpinned prefixes
        assert transport.order != sorted(transport.order)
        assert sorted(items[0][2] for items in transport.submitted) == \
            [list(range(100 * k, 100 * k + 5)) for k in range(8)]
        assert all(items[0][:2] == (7, 0) for items in transport.submitted)
        assert cl.pull_transfer_failures == 0

        h, local, _ = prefix(8, 2)
        transport.fail.add(transport.n + 1)
        r = cl.pull_async(h).wait(10)
        assert r.common_hit == 2 and r.remote_blocks == 0 and list(r.full_slots) == list(local)
        r.finalize()
        h, local, _ = prefix(9, 2)
        transport.lose.add(transport.n + 1)
        r = cl.wait(cl.pull_async(h).job_id, 10)
        assert r.common_hit == 2 and r.remote_blocks == 0
        r.finalize()
        transport.busy = True
        h, local, _ = prefix(10, 2)
        r = cl.pull_async(h).wait(10)
        assert r.common_hit == 2 and r.remote_blocks == 0
        r.finalize()
        assert cl.pull_transfer_failures == 3 and cl.pull_promotion_shortfalls == 0
        assert real.mempool_used() == 8 * 3 + 8 * 5 + 3 * 2
        assert real.check_health() == 0
        assert cl._jobs == {} and cl._pending == {}
        cl.close()
    finally:
        srv.close()


def test_promotion_survives_a_prefix_evicted_in_the_unpinned_window(cfg):
    """The pin drops before the promoted blocks are published; if the prefix is
    evicted in between, the insert lands nothing, its slots return to the pool
    and the job completes with the re-queried (here empty) hit."""
    c = cfg(**data(data_bytes=1 << 20))
    srv = up(c, geometry())
    try:
        cl = client(c)
        real = cl.index
        h = hashes(8)
        local = cl.allocate_slots(3)
        assert cl.insert(h[:3], local).error == OK
        index = scripted_index(real, lambda hs: _Q(local, 3, np.arange(100, 105, dtype=np.int32), 8))

        def insert(hs, slots, **kw):
            hoard = real.allocate_slots(real.mempool_free() + 3)    # only the unpinned prefix can be evicted
            try:
                return real.insert(hs, slots, **kw)
            finally:
                real.recycle_slots(hoard)

        index.insert = insert
        cl._index, cl._t = index, _Transport()
        r = cl.pull_async(h).wait(10)
        assert r.common_hit == 0 and r.remote_blocks == 5 and len(r.full_slots) == 0
        assert cl.pull_promotion_shortfalls == 1 and cl.pull_transfer_failures == 0
        assert real.query(h, local_only=True).common_hit == 0
        assert real.mempool_used() == 0 and real.check_health() == 0
        cl.close()
    finally:
        srv.close()


def test_stage_recycles_a_short_aux_allocation(cfg):
    """A partial SWA allocation (impossible with the all-or-none pool, forced
    here) must go back to the pool instead of leaking."""
    c = cfg(**data(data_bytes=1 << 20, swa_ratio=8 * 1024 / (1 << 20)))
    srv = up(c, geometry(swa_slot_bytes=1024, swa_window_blocks=2))
    try:
        cl = client(c)
        real, recycled = cl.index, []
        h = hashes(4)
        local = cl.allocate_slots(2)
        assert cl.insert(h[:2], local).error == OK

        class Q(_Q):
            swa_source_rank, swa_slots = 7, np.array([5, 6], dtype=np.int32)

        index = scripted_index(real, lambda hs: Q(local, 2, np.array([40, 41], np.int32), 4))
        index.allocate_slots = lambda n, kind=FULL: real.allocate_slots(n - 1 if kind == SWA else n, kind)

        def recycle_slots(slots, kind=FULL):
            recycled.append((kind, list(slots)))
            real.recycle_slots(slots, kind)

        index.recycle_slots = recycle_slots
        cl._index, cl._t = index, _Transport()
        r = cl.pull_async(h).wait(10)
        assert r.common_hit == 4 and r.remote_blocks == 2 and len(r.swa_slots) == 0
        r.finalize()
        assert [k for k, s in recycled if s] == [SWA]
        assert real.swa_mempool_total() == 8 and real.check_health() == 0
        cl.close()
    finally:
        srv.close()


class _Prepare:
    def __init__(self, status, hit, full=(), swa=(), mamba=()):
        self.status, self.remote_hit = status, hit
        self.full_slots, self.swa_slots, self.mamba_slots = (
            np.asarray(s, dtype=np.int32) for s in (full, swa, mamba))


class _Commit:
    def __init__(self, status, remote_hit, errors):
        self.status, self.remote_hit, self.errors = status, remote_hit, errors


FULLC, SWAC, MAMBAC = (radixshmem.ComponentType.FULL, radixshmem.ComponentType.SWA,
                     radixshmem.ComponentType.MAMBA)
AUX_MASK = (radixshmem.COMPONENT_MASK_FULL | radixshmem.COMPONENT_MASK_SWA |
            radixshmem.COMPONENT_MASK_MAMBA)


def push_index(real, hit, pool, swa_pool=(), mamba_pool=(), prepare_status=0,
               commit_status=0, insert_error=OK, aux_errors=None, probe=None):
    """Script remote grants/publication; local data and pins use the real index."""
    pools = {FULLC: list(pool), SWAC: list(swa_pool), MAMBAC: list(mamba_pool)}

    class Index:
        calls = []

        def __getattr__(self, name):
            return getattr(real, name)

        def rht_probe(self, hs):
            return probe

        def remote_prepare(self, rank, hs, full_hit, entries):
            if int(prepare_status) != 0:
                return _Prepare(prepare_status, 0)
            full = pools[FULLC][:max(0, full_hit - hit)]
            if not full:
                return _Prepare(0, hit)
            end, counts = hit + len(full), {SWAC: 0, MAMBAC: 0}
            for kind, at, count in entries:
                counts[kind] += (max(0, min(at, end) - (at - count)) if kind == SWAC
                                 else int(at <= end))
            grants = [pools[k][:counts[k]] if len(pools[k]) >= counts[k] else []
                      for k in (SWAC, MAMBAC)]
            return _Prepare(0, hit, full, *grants)

        def remote_commit(self, rank, hs, entries):
            Index.calls.append(("commit", rank, len(hs),
                                [(int(k), int(at), list(s)) for k, at, s in entries]))
            if int(commit_status) != 0:
                return _Commit(commit_status, 0, [])
            errors = [insert_error]
            errors.extend((aux_errors or {}).get((kind, at), OK) if insert_error == OK
                          else radixshmem.InsertError.FULL_PATH_MISSING
                          for kind, at, _ in entries[1:])
            return _Commit(0, len(hs), errors)

        def remote_recycle(self, rank, slots, component=FULLC):
            Index.calls.append(("recycle", rank, list(slots), int(component)))
            return 0

    return Index()


def commit_entries(idx):
    return [entry for call in idx.calls if call[0] == "commit" for entry in call[3]]


@contextmanager
def push_client(cfg, *, aux=False):
    """A 1 MiB data-mode server; with ``aux`` the ratios plan 8 SWA slots of
    1 KiB and 4 Mamba slots of 4 KiB next to the FULL pool."""
    if aux:
        c = cfg(**data(data_bytes=1 << 20, swa_ratio=8 * 1024 / (1 << 20),
                       mamba_ratio=4 * 4096 / (1 << 20)))
        srv = up(c, geometry(swa_slot_bytes=1024, swa_window_blocks=2, mamba_slot_bytes=4096))
    else:
        c = cfg(**data(data_bytes=1 << 20))
        srv = up(c, geometry())
    try:
        cl = client(c)
        try:
            yield cl
        finally:
            cl.close()
    finally:
        srv.close()


def seed_aux_path(c, h, islands=((0, 2), (3, 5), (6, 8)), checkpoints=(4, 8)):
    full = c.allocate_slots(len(h))
    assert c.insert(h, full).error == OK
    cells, cps = {}, {}
    for lo, hi in islands:
        s = c.allocate_slots(hi - lo, SWAC)
        assert c.insert(h[:hi], s, component=SWAC).error == OK
        cells[(lo, hi)] = list(s)
    for end in checkpoints:
        m = c.allocate_slots(1, MAMBAC)
        assert c.insert(h[:end], m, component=MAMBAC).error == OK
        cps[end] = int(m[0])
    return full, cells, cps


def seed_aux(c, h):
    seed_aux_path(c, h, islands=((len(h) - 2, len(h)),), checkpoints=(len(h),))


@pytest.mark.parametrize("rank", [7, None], ids=["explicit_job", "rht_job_id"])
def test_push_async_writes_the_missing_suffix(cfg, rank):
    with push_client(cfg) as c:
        real, h = c.index, hashes(8)
        local = c.allocate_slots(8)
        assert c.insert(h, local).error == OK
        idx = push_index(real, hit=3, pool=list(range(200, 205)), probe=(7, 6))
        transport = _Transport()
        c._index, c._t = idx, transport

        job = c.push_async(h, rank=rank)
        r = c.wait(job if rank is not None else job.job_id, 10)
        assert job.remote_hit == 3 and job.planned_push == 5 and job.done()
        assert (r.committed, r.remote_blocks, r.remote_bytes, r.target_rank) == (8, 5, 5 * 8192, 7)
        assert transport.submitted == [[
            (7, 0, list(range(200, 205)), list(local[3:]), radixshmem.OP_WRITE)]]
        assert c.push_transfer_failures == c.push_commit_shortfalls == 0
        assert real.check_health() == 0 and c._jobs == {} and c._pending == {}


def test_push_without_missing_target_data_does_not_transfer(cfg):
    with push_client(cfg) as c:
        real, h = c.index, hashes(8)
        assert c.insert(h, c.allocate_slots(8)).error == OK
        transport = _Transport()
        c._index, c._t = push_index(real, hit=0, pool=[]), transport
        r = c.push_async(h).wait(10)
        assert (r.committed, r.remote_blocks, r.target_rank) == (0, 0, -1)

        c._index = push_index(real, hit=8, pool=[])
        job = c.push_async(h, rank=7)
        r = job.wait(10)
        assert job.remote_hit == 8 and job.planned_push == 0
        assert (r.committed, r.remote_blocks, r.remote_bytes) == (8, 0, 0)
        assert transport.submitted == [] and real.query(h, local_only=True).common_hit == 8
        assert real.check_health() == 0


@pytest.mark.parametrize("end, blocks", [(8, 13), (7, 10)], ids=["available_source", "partial_grant"])
def test_push_carries_available_full_and_historical_aux(cfg, end, blocks):
    with push_client(cfg, aux=True) as c:
        real, h = c.index, hashes(10)
        full, cells, cps = seed_aux_path(c, h[:8])
        full_slots = list(range(200, 200 + end - 3))
        swa_slots = list(range(900, 900 + end - 2))
        mamba_slots = [950, 951] if end == 8 else [950]
        idx = push_index(real, hit=3, pool=full_slots, swa_pool=swa_slots, mamba_pool=mamba_slots)
        transport = _Transport()
        c._index, c._t = idx, transport

        r = c.push_async(h, AUX_MASK, rank=7).wait(10)
        assert r.committed == end and r.remote_blocks == blocks
        assert r.remote_bytes == len(full_slots) * 8192 + len(swa_slots) * c.store.pool(SWAC).slot_bytes + \
            len(mamba_slots) * c.store.pool(MAMBAC).slot_bytes
        assert transport.submitted == [[
            (7, 0, full_slots, list(full[3:end]), radixshmem.OP_WRITE),
            (7, 1, swa_slots, cells[(0, 2)] + cells[(3, 5)] + cells[(6, 8)][:end - 6], radixshmem.OP_WRITE),
            (7, 2, mamba_slots, [cps[4]] + ([cps[8]] if end == 8 else []), radixshmem.OP_WRITE)]]
        assert commit_entries(idx) == [
            (0, end, full_slots), (1, 2, [900, 901]), (1, 5, [902, 903]),
            (1, end, swa_slots[4:]), (2, 4, [950])] + ([(2, 8, [951])] if end == 8 else [])
        assert c.push_staging_shortfalls == int(end < 8)
        assert real.query(h, local_only=True).common_hit == 8 and real.check_health() == 0


def test_push_evict_local_preserves_shared_pinned_or_unmoved_data(cfg):
    with push_client(cfg, aux=True) as c:
        real, h = c.index, hashes(8)
        transport = _Transport()

        def push(mask=AUX_MASK, **pools):
            c._index, c._t = push_index(real, hit=0, pool=list(range(100, 108)), **pools), transport
            return c.push_async(h, mask, rank=7, evict_local=True).wait(10)

        def local(hs):
            return int(real.query(hs, local_only=True).common_hit)

        seed_aux(c, h)
        r = push(swa_pool=[900, 901], mamba_pool=[950])
        assert (r.committed, r.local_evicted, local(h)) == (8, 8, 0)
        assert real.total_blocks() == real.mempool_used() == 0

        h2 = np.concatenate([h[:4], hashes(4, base=500)])
        seed_aux(c, h)
        seed_aux(c, h2)
        r = push(swa_pool=[900, 901], mamba_pool=[950])
        assert (r.committed, r.local_evicted, local(h), local(h2)) == (8, 4, 4, 8)

        seed_aux(c, h)
        q = real.query(h, mask=radixshmem.COMPONENT_MASK_FULL, lock=True)
        r = push(swa_pool=[900, 901], mamba_pool=[950])
        assert (r.committed, r.local_evicted, local(h)) == (8, 0, 8)
        q.finalize()

        r = push(mamba_pool=[950])
        assert r.committed == 8 and r.local_evicted == 0 and c.push_staging_shortfalls == 1
        r = push(radixshmem.COMPONENT_MASK_FULL)
        assert r.committed == 8 and r.local_evicted == 0
        assert real.query(h, mask=AUX_MASK, local_only=True).common_hit == 8
        assert real.check_health() == 0


@pytest.mark.parametrize("failure, committed, counts, recycled", [
    ("full_write", 3, (0, 2, 1), (0, 1, 2)), ("swa_write", 8, (5, 0, 1), (1,)),
    ("commit_error", 0, (5, 2, 1), ()), ("commit_lost", 0, (5, 2, 1), ())])
def test_push_failure_preserves_slot_ownership_and_source(cfg, failure, committed, counts, recycled):
    with push_client(cfg, aux=True) as c:
        real, h = c.index, hashes(8)
        seed_aux(c, h)
        grants = {0: list(range(200, 205)), 1: [900, 901], 2: [950]}
        idx = push_index(real, hit=3, pool=grants[0], swa_pool=grants[1], mamba_pool=grants[2],
                         insert_error=radixshmem.InsertError.FULL_PATH_MISSING if failure == "commit_error" else OK,
                         commit_status=radixshmem.RemoteStatus.TIMEOUT if failure == "commit_lost" else 0)
        transport = _Transport()
        if failure.endswith("write"):
            transport.fail_items[1] = {0 if failure == "full_write" else 1}
        c._index, c._t = idx, transport

        r = c.push_async(h, AUX_MASK, rank=7, evict_local=True).wait(10)
        assert (r.committed, r.remote_blocks, r.local_evicted) == (committed, sum(counts), 0)
        assert r.remote_bytes == sum(n * c.store.pool(radixshmem.ComponentType(k)).slot_bytes
                                     for k, n in enumerate(counts))
        assert sorted((v[3], v[2]) for v in idx.calls if v[0] == "recycle") == [
            (k, grants[k]) for k in recycled]
        assert [k for k, _, _ in commit_entries(idx)] == (
            [] if failure == "full_write" else [0, 2] if failure == "swa_write" else [0, 1, 2])
        assert c.push_transfer_failures == int(failure.endswith("write"))
        assert c.push_commit_shortfalls == int(failure.startswith("commit"))
        assert real.query(h, mask=AUX_MASK, local_only=True).common_hit == 8
        assert real.check_health() == 0


def test_push_prepare_failure_releases_the_source_pin(cfg):
    with push_client(cfg, aux=True) as c:
        real, h = c.index, hashes(8)
        seed_aux(c, h)
        idx = push_index(real, hit=3, pool=[], prepare_status=radixshmem.RemoteStatus.TIMEOUT)
        prepare = idx.remote_prepare

        def checked_prepare(*args):
            assert real.evict_prefix(h) == 0
            return prepare(*args)

        idx.remote_prepare = checked_prepare
        transport = _Transport()
        c._index, c._t = idx, transport
        r = c.push_async(h, AUX_MASK, rank=7, evict_local=True).wait(10)
        assert (r.committed, r.remote_blocks, r.local_evicted) == (0, 0, 0)
        assert transport.submitted == []
        assert real.query(h, mask=AUX_MASK, local_only=True).common_hit == 8
        assert real.evict_prefix(h) == 8 and real.check_health() == 0


def test_push_one_failed_fragment_keeps_the_source(cfg):
    with push_client(cfg, aux=True) as c:
        real, h = c.index, hashes(8)
        seed_aux_path(c, h)
        idx = push_index(real, hit=3, pool=list(range(200, 205)), swa_pool=list(range(900, 906)),
                         mamba_pool=[950, 951], aux_errors={(SWAC, 5): radixshmem.InsertError.BAD_REQUEST})
        c._index, c._t = idx, _Transport()

        r = c.push_async(h, AUX_MASK, rank=7, evict_local=True).wait(10)
        assert (r.committed, r.remote_blocks, r.local_evicted) == (8, 13, 0)
        assert c.push_commit_shortfalls == 1
        assert not any(v[0] == "recycle" for v in idx.calls)
        assert real.query(h, mask=AUX_MASK, local_only=True).common_hit == 8
        assert real.check_health() == 0


class _Xfer:
    """Completes on poll 3; `failed` reports failure, `hang` keeps release busy."""

    local_segment = "me"

    def __init__(self, data):
        self.d, self.polls, self.n, self.hang, self.reject = data, {}, 0, set(), False
        self.failed = set()
        self.release_blocked = threading.Event()
        self.ops = []

    def read(self, seg, remote, local, kind):
        return self._submit("read")

    def write(self, seg, local, remote, kind):
        return self._submit("write")

    def _submit(self, op):
        if self.reject:
            return self.d.INVALID_BATCH, 0
        self.n += 1
        self.polls[self.n] = 0
        self.ops.append(op)
        return self.n, 1

    def poll(self, h):
        self.polls[h] += 1
        if h in self.failed:
            return self.d.XferState.FAILED, 0
        if h in self.hang or self.polls[h] < 3:
            return self.d.XferState.PENDING, 0
        return self.d.XferState.COMPLETED, 4096

    def release(self, h):
        if h in self.hang:
            self.release_blocked.set()
            return self.d.XFER_EBUSY
        return self.d.XFER_OK


@pytest.mark.parametrize("op", [radixshmem.OP_READ, radixshmem.OP_WRITE], ids=["read", "write"])
@pytest.mark.parametrize("failure", ["TIMEOUT", "FAILED"])
def test_server_defers_failed_completion_until_batch_drains(cfg, op, failure):
    from radixshmem import _data

    srv = up(cfg(**data(data_bytes=1 << 20)), geometry())
    x = _Xfer(_data)
    try:
        srv._xfer, srv._peers = x, {7: "peer"}
        x.hang.add(1)
        if failure == "FAILED":
            x.failed.add(1)
        item = (7, 0, [1, 2], [3, 4], op)
        ticket = srv.submit([item], 0 if failure == "TIMEOUT" else 5000)
        assert x.release_blocked.wait(2)

        assert srv.wait_any([ticket], 100) == {"done": {}, "unknown": []}
        assert srv._jobs[ticket].replies == [None]
        assert srv.stats()["pending_jobs"] == 1 and srv.stats()["transfers"] == 0

        x.hang.clear()
        done, deadline = {}, time.monotonic() + 3
        while not done and time.monotonic() < deadline:
            done.update(srv.wait_any([ticket], 500)["done"])
        reply = done[ticket][0]
        assert reply[0] is False and failure in reply[3]
        assert srv._zombies == []
        assert srv.stats()["pending_jobs"] == 0 and srv.stats()["transfers"] == 1
        assert srv.stats()["transfer_failures"] == 1
        srv._reap()
        assert srv.stats()["transfers"] == 1
    finally:
        x.hang.clear()
        srv._xfer = None            # the fake must not be released as a real SlotTransfer
        srv.close()


def test_server_jobs_complete_busy_and_expire(cfg):
    from radixshmem import _data

    c = cfg(**data(data_bytes=1 << 20, max_inflight=2, max_pending_jobs=3, job_ttl_s=0.2))
    srv = up(c, geometry())
    try:
        x = _Xfer(_data)
        srv._xfer, srv._peers = x, {7: "peer"}
        item = (7, 0, [1, 2], [3, 4], radixshmem.OP_READ)
        witem = (7, 0, [1, 2], [3, 4], radixshmem.OP_WRITE)

        def collect(tickets, timeout=2.0):
            done, deadline = {}, time.monotonic() + timeout
            while len(done) < len(tickets) and time.monotonic() < deadline:
                done.update(srv.wait_any([t for t in tickets if t not in done], 500)["done"])
            return done

        t1, t2 = srv.submit([item, witem], 5000), srv.submit([item], 5000)
        done = collect([t1, t2])
        assert done[t1] == [(True, 4096, 1, "")] * 2 and done[t2] == [(True, 4096, 1, "")]
        assert srv.stats()["transfers"] == 3 and srv.stats()["pending_jobs"] == 0
        assert sorted(x.ops) == ["read", "read", "write"]

        x.reject = True
        t4 = srv.submit([item], 1000)
        assert collect([t4])[t4][0][0] is False
        x.reject = False

        ts = [srv.submit([item], 5000) for _ in range(3)]
        with pytest.raises(radixshmem.ServerBusy):
            srv.submit([item], 5000)
        assert srv.wait_any([999], 10) == {"done": {}, "unknown": [999]}
        time.sleep(1.5)
        assert srv.wait_any(ts, 10)["unknown"] == ts
        srv.submit([item], 5000)
    finally:
        srv._xfer = None            # the fake must not be released as a real SlotTransfer
        srv.close()
