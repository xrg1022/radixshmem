# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RadixClient: index maintenance plus ``pull_async`` and the staged push
(``stage_push`` / ``submit_push``), the calls that also move bytes. It attaches the index (and, in data mode, the
SlotStore) its RadixServer published; the server runs the transfers, this side
stages the slots, submits, and publishes the result once the server reports
completion.

A client can be built before its server is ready. ``RadixClient(name,
geometry)`` hands the server the geometry (the Configure RPC) and returns at
once; ``status()`` asks the server where it stands without waiting;
``wait_ready()`` is the one call that blocks. The index and the SlotStore are
attached on the first index or data call, which raises ``ServerNotReady``
while the server is still waiting or configuring."""

from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from radixshmem import _core, _geometry, rpc
from radixshmem._core import COMPONENT_MASK_FULL, MAX_REMOTE_OP_BLOCKS, ComponentType, InsertError
from radixshmem.rpc import OP_READ, OP_WRITE, DataPlaneUnavailable, ServerNotReady

FULL, SWA, MAMBA = ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA

# Server lifecycle. Whether there is a data plane is NodeInfo.data_plane.
MODE_STARTING = "starting"          # start() has not finished its preflight
MODE_WAITING = "waiting"            # up, no geometry yet: a client's configure or a peer's publish
MODE_CONFIGURING = "configuring"    # geometry accepted, regions being created / cluster joining
MODE_READY = "ready"
MODE_CLOSED = "closed"

_ATTACH_TIMEOUT_MS = 30000          # a ready server's SlotStore is complete; attach never spins long

# (peer_rank, kind, remote_slots, local_slots, op) -> (ok, bytes, descriptors, error)
TransferItem = Tuple[int, int, List[int], List[int], int]
TransferReply = Tuple[bool, int, int, str]


@dataclass
class NodeInfo:
    mode: str
    data_plane: bool = False
    index_name: str = ""
    data_name: str = ""
    hugepage_path: str = ""
    geometry: Optional[Dict[str, Any]] = None
    rank: int = -1
    world_size: int = 0
    node_name: str = ""
    segment: str = ""
    data_bytes: int = 0
    last_error: str = ""            # why the last configuration failed; "" when none did
    swa_ratio: float = 0.0
    mamba_ratio: float = 0.0


@dataclass
class PullResult:
    """Everything in [0, common_hit) is in the local SlotStore when this exists.
    ``finalize()`` drops the local pin taken with ``lock=True`` (no-op otherwise)."""

    common_hit: int
    full_slots: np.ndarray
    swa_start: int
    swa_slots: np.ndarray
    mamba_slot: int
    remote_blocks: int
    remote_bytes: int
    source_rank: int
    finalize: Callable[[], None]


@dataclass
class PushResult:
    """Prefix the target holds at job end (0 after a failed insert); remote_* as in PullResult;
    target_rank is -1 when no target was chosen."""

    committed: int
    remote_blocks: int
    remote_bytes: int
    target_rank: int


@dataclass
class StagedPush:
    """One push staged by ``stage_push()``: ``rank`` holds ``hashes[:start]`` and
    reserved ``remote_slots`` for ``hashes[start:end]``; ``local_slots`` are
    borrowed from this node's FULL pool and never published here. The caller
    writes block ``start + i`` into ``local_slots[i]``, then hands this to
    ``submit_push()`` or ``abort_push()`` exactly once. ``rank`` is -1 when no
    target was found; ``end == start`` means there is nothing to write.

    ``swa`` holds ``(offset, local, remote)`` per staged SWA run: write the SWA
    state of block ``offset + i`` into ``local[i]``. ``mamba`` holds
    ``(end, local, remote)`` per staged checkpoint: write the state at ``end``
    into slot ``local``. Both cover only ``[0, end)`` and are empty when the
    caller asked for none or either side's pool could not supply the lane."""

    rank: int
    hashes: np.ndarray
    start: int
    end: int
    local_slots: np.ndarray
    remote_slots: np.ndarray
    swa: List[Tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)
    mamba: List[Tuple[int, int, int]] = field(default_factory=list)
    _used: bool = field(default=False, init=False, repr=False)

    def _lanes(self) -> List[Tuple[ComponentType, np.ndarray, np.ndarray]]:
        """(kind, local, remote) per staged lane, FULL first."""
        lanes = [(FULL, self.local_slots, self.remote_slots)]
        if self.swa:
            lanes.append((SWA, np.concatenate([l for _, l, _ in self.swa]),
                          np.concatenate([r for _, _, r in self.swa])))
        if self.mamba:
            lanes.append((MAMBA, np.asarray([l for _, l, _ in self.mamba], dtype=np.int32),
                          np.asarray([r for _, _, r in self.mamba], dtype=np.int32)))
        return lanes


class PullJob:
    """Handle of one pull_async(). ``wait()`` blocks until the job completed and
    returns its PullResult; a failed or refused transfer completes the job with
    the local hit, it never raises."""

    def __init__(self, job_id: int, client: "RadixClient"):
        self.job_id = job_id
        self._client = client
        self._f: concurrent.futures.Future = concurrent.futures.Future()
        self._q = None
        self._hashes: np.ndarray = np.empty(0, dtype=np.uint64)
        self._mask = 0
        self._lock = True
        self._hit = 0
        self._items: List[TransferItem] = []
        self._staged: Dict[ComponentType, Tuple[int, np.ndarray]] = {}
        self._src = -1
        self._ticket = -1
        self._cancelled = False
        # Known when pull_async() returns: blocks the local tree already held, and
        # that plus the remote run being pulled (equal when nothing is pulled).
        self.local_hit = 0
        self.planned_hit = 0

    def done(self) -> bool:
        return self._f.done()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def wait(self, timeout: Optional[float] = None) -> PullResult:
        if self._cancelled:
            raise RuntimeError(f"job {self.job_id} was cancelled")
        try:
            r = self._f.result(timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"job {self.job_id} still running") from None
        self._client._jobs.pop(self.job_id, None)
        return r

    def cancel(self) -> None:
        """Give up on the result: the job leaves the client's table now. A
        transfer still in flight finishes in the background (the RDMA READ into
        the staged slots cannot be aborted), its blocks are published as usual,
        and no pin is left behind. ``wait()`` afterwards raises."""
        self._client.cancel(self)


class PushJob:
    """Asynchronous push handle; cancellation is unsupported."""

    def __init__(self, job_id: int, client: "RadixClient"):
        self.job_id = job_id
        self._client = client
        self._f: concurrent.futures.Future = concurrent.futures.Future()
        self._q = None                    # no local pin: the source is borrowed slots
        self._items: List[TransferItem] = []
        self._ticket = -1
        self._sp: Optional[StagedPush] = None
        self.remote_hit = 0
        self.planned_push = 0

    def done(self) -> bool:
        return self._f.done()

    def wait(self, timeout: Optional[float] = None) -> PushResult:
        try:
            r = self._f.result(timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"job {self.job_id} still running") from None
        self._client._jobs.pop(self.job_id, None)
        return r


def _empty() -> np.ndarray:
    return np.empty(0, dtype=np.int32)


class _Rpc:
    def __init__(self, endpoint: str):
        self._stub = rpc.Stub(endpoint)

    def describe(self) -> NodeInfo:
        return NodeInfo(**self._stub.call("Describe", timeout=30))

    def wait_ready(self, timeout_s) -> NodeInfo:
        return NodeInfo(**self._stub.call("WaitReady", {"timeout_ms": int(timeout_s * 1000)},
                                          timeout=timeout_s + 5))

    def configure(self, spec: Dict[str, Any]) -> NodeInfo:
        return NodeInfo(**self._stub.call("Configure", {"geometry": spec}, timeout=30))

    def submit(self, items, timeout_ms) -> int:
        req = {"items": [[int(r), int(k), [int(x) for x in rs], [int(x) for x in ls], int(op)]
                         for r, k, rs, ls, op in items], "timeout_ms": int(timeout_ms)}
        return int(self._stub.call("Submit", req, timeout=30)["ticket"])

    def wait_any(self, tickets, timeout_ms) -> Dict[str, Any]:
        r = self._stub.call("WaitAny", {"tickets": list(tickets), "timeout_ms": int(timeout_ms)},
                            timeout=timeout_ms / 1000 + 5)
        return {"done": {int(t): [tuple(x) for x in v] for t, v in r["done"].items()},
                "unknown": [int(t) for t in r["unknown"]]}

    def close(self) -> None:
        self._stub.close()


class RadixClient:
    """``RadixClient(name)`` connects to the server started as ``--name <name>``
    on this node (socket ``unix:///dev/shm/<name>.sock``; ``endpoint`` overrides
    it for TCP or another path). The server need not be ready: the constructor
    only talks gRPC and returns at once. With ``geometry`` (a ``Geometry`` or
    its dict) it hands the server the geometry first (idempotent; a different
    one raises ``GeometryMismatch``, one this server cannot take
    ``ValueError``). ``status()`` refreshes ``info`` without waiting,
    ``wait_ready(timeout_s)`` blocks until the server is ready, ``configure()``
    sends a geometry later. The index and, when the server has a data plane,
    the SlotStore are attached on the first index or data call; before the
    server is ready that call raises ``ServerNotReady``.

    ``geometry`` is what the server published; callers compare their own page
    size and bytes against it. Index methods (query, insert, allocate_slots,
    recycle_slots, flush, reset, check_health, stats) are those of the
    underlying index client. ``pull_async`` brings a peer's blocks into local
    slots; ``stage_push`` / ``submit_push`` write blocks the caller put into
    borrowed local slots into a peer's slots and publish them there, never on
    this node. ``max_outstanding`` bounds the jobs in flight; both block while
    it is reached.

    ``local_read_only=True`` is the light form for processes that only read: it
    attaches the index and the SlotStore, never joins the cluster (no etcd,
    RDMA or RHT plane) and does not start a completer thread. ``query()`` is
    always local, ``is_distributed()`` is False, and ``pull_async`` /
    ``stage_push`` / ``insert`` / ``allocate_slots`` / ``recycle_slots`` /
    ``reset`` raise ``RuntimeError``; ``info`` still carries the server's rank
    and world size."""

    def __init__(self, name: str, geometry=None, *, endpoint: Optional[str] = None,
                 max_outstanding: int = 256, local_read_only: bool = False):
        self.name = name
        self.local_read_only = local_read_only
        self._index = None
        self._store = None
        self._attach_mu = threading.Lock()
        self._t = _Rpc(endpoint or rpc.endpoint_for(name))
        try:
            self.info = (self._t.configure(_geometry.as_spec(geometry)) if geometry is not None
                         else self._t.describe())
        except Exception:
            self._t.close()
            raise
        self.pull_transfer_failures = 0
        self.pull_staging_shortfalls = 0
        self.pull_promotion_shortfalls = 0
        self.push_transfer_failures = 0
        self.push_staging_shortfalls = 0
        self.push_commit_shortfalls = 0
        self.saturated = 0
        self._mu = threading.Lock()
        self._jobs: Dict[int, Union[PullJob, PushJob]] = {}
        self._pending: Dict[int, Union[PullJob, PushJob]] = {}
        self._next_id = 1
        self._slots = threading.Semaphore(max_outstanding)
        self._work = threading.Event()
        self._completer: Optional[threading.Thread] = None
        self._closed = False

    # ------------------------------------------------------------- control

    def status(self) -> NodeInfo:
        """The server's state now: one RPC, no waiting. Refreshes ``info``."""
        self.info = self._t.describe()
        return self.info

    def configure(self, geometry) -> NodeInfo:
        """Hand the server a geometry; returns as soon as it is validated and
        planned (``mode`` configuring, or ready when it already served it)."""
        self.info = self._t.configure(_geometry.as_spec(geometry))
        return self.info

    def wait_ready(self, timeout_s: float = 300.0) -> NodeInfo:
        """Block until the server is ready. ``TimeoutError`` while it is still
        waiting for a geometry or configuring, ``RuntimeError`` (with the
        server's ``last_error``) when its configuration failed."""
        self.info = self._t.wait_ready(timeout_s)
        return self.info

    def _attach(self):
        """The index client, attached on first use once the server is ready."""
        index = self._index
        if index is not None:
            return index
        with self._attach_mu:
            if self._index is not None:
                return self._index
            if self._closed:
                raise RuntimeError("client is closed")
            info = self.info if self.info.mode == MODE_READY else self.status()
            if info.mode != MODE_READY:
                raise ServerNotReady(info.mode, info.last_error)
            index = _core.RadixClient(info.index_name, local_read_only=self.local_read_only)
            store = None
            if info.data_plane:
                from radixshmem import _data

                store = _data.SlotStore.attach(info.data_name, info.hugepage_path, _ATTACH_TIMEOUT_MS)
            _geometry.check_attached(info.geometry, index, store)
            self._store, self._index = store, index
            return index

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._attach(), name)

    @property
    def index(self):
        return self._attach()

    @property
    def geometry(self) -> Optional[Dict[str, Any]]:
        return self.info.geometry

    @property
    def store(self):
        self._attach()
        if self._store is None:
            raise DataPlaneUnavailable("index-only server: no SlotStore to read")
        return self._store

    def slot_view(self, slot: int, kind: ComponentType = FULL):
        return self.store.slot_view(int(slot), kind)

    def close(self) -> None:
        self._closed = True
        self._work.set()
        if self._completer is not None:
            self._completer.join(2)
        self._store = None
        self._index = None
        self._t.close()

    # ------------------------------------------------------------ pull_async

    def pull_async(self, hashes, mask: int = COMPONENT_MASK_FULL, *, lock: bool = True,
                   timeout_ms: int = 30000, block: bool = True) -> PullJob:
        """query, stage local slots for every remote piece, hand the transfer to
        the server and return; the completer publishes the blocks when the
        server reports them and completes the job with local slots for
        [0, common_hit). With ``block=False`` a client at ``max_outstanding``
        does not wait: the job completes at once with the local hit."""
        if self.local_read_only:
            raise RuntimeError("local_read_only client: pull_async needs the cluster plane; "
                               "use query() for the local hit")
        self.store
        with self._mu:
            job = PullJob(self._next_id, self)
            self._next_id += 1
            self._jobs[job.job_id] = job
        job._hashes = np.ascontiguousarray(hashes, dtype=np.uint64)
        job._mask, job._lock = mask, lock
        job._q = q = self._index.query(job._hashes, mask=mask, local_only=False, lock=True,
                                       update_meta=True)
        job._hit = self._pull_stage(job, int(q.common_hit) if int(q.status) == 0 else 0)
        job.planned_hit = job._hit
        if not job._items:
            self._pull_complete(job, [])
            return job
        self._submit_job(job, timeout_ms, block, self._pull_complete)
        return job

    def wait(self, job: Union[PullJob, PushJob, int],
             timeout: Optional[float] = None) -> Union[PullResult, PushResult]:
        return (job if isinstance(job, (PullJob, PushJob)) else self._jobs[job]).wait(timeout)

    def cancel(self, job: Union[PullJob, int]) -> None:
        """See ``PullJob.cancel``. Under ``_mu`` so that a result set by the
        completer at the same moment still gets its pin released."""
        job = self._jobs[job] if isinstance(job, int) else job
        if not isinstance(job, PullJob):
            raise TypeError("only pull jobs can be cancelled")
        with self._mu:
            job._cancelled = True
            self._jobs.pop(job.job_id, None)
            r = (job._f.result() if job._f.done() and job._f.exception() is None
                 else None)
        if r is not None:
            r.finalize()

    def wait_any(self, jobs: Sequence[Union[PullJob, PushJob]],
                 timeout: Optional[float] = None) -> List[Union[PullJob, PushJob]]:
        concurrent.futures.wait([j._f for j in jobs], timeout,
                                return_when=concurrent.futures.FIRST_COMPLETED)
        return [j for j in jobs if j.done()]

    # ------------------------------------------------------------ staged push

    def stage_push(self, hashes, *, swa: Sequence[Tuple[int, int]] = (),
                   mamba: Sequence[int] = (), rank: Optional[int] = None,
                   fallback_rank: Optional[int] = None) -> StagedPush:
        """Reserve on a peer the FULL blocks of ``hashes`` it lacks and borrow as
        many local slots to stage them in; nothing is published on this node.
        ``swa`` lists the ``[a, b)`` block runs and ``mamba`` the checkpoint
        ends the caller can supply; the same prepare reserves those within the
        staged ``[0, end)``, including runs below ``start`` (the peer may hold
        that FULL prefix without the SWA window or checkpoint that ends the new
        one). The target is ``rank``, else the RHT's deepest holder, else
        ``fallback_rank``. One remote_prepare both reads the peer's hit and
        reserves the rest, so the staged range cannot drift from the grant."""
        if self.local_read_only:
            raise RuntimeError("local_read_only client: stage_push needs the cluster plane")
        self.store
        h = np.ascontiguousarray(hashes, dtype=np.uint64)[:MAX_REMOTE_OP_BLOCKS]
        n = len(h)
        # Clip to the truncated h; remote_prepare rejects any other bad entry.
        runs = [(int(a), min(int(b), n)) for a, b in swa if int(a) < min(int(b), n)]
        ends = [int(e) for e in mamba if int(e) <= n]
        if rank is None and n:
            best = self._index.rht_probe(h)
            rank = int(best[0]) if best is not None else fallback_rank
        if rank is None or n == 0:
            return StagedPush(-1, h, 0, 0, _empty(), _empty())
        rank = int(rank)
        if rank == int(self._index.rank()):
            raise ValueError("push target is this node")
        entries = [(SWA, b, b - a) for a, b in runs] + [(MAMBA, e, 1) for e in ends]
        rep = self._index.remote_prepare(rank, h, n, entries)
        if int(rep.status) != 0:
            return StagedPush(rank, h, 0, 0, _empty(), _empty())
        hr = int(rep.remote_hit)
        remote = np.asarray(rep.full_slots, dtype=np.int32)
        want = min(len(remote), int(self._index.mempool_total()))
        local = self._index.allocate_slots(want, FULL) if want > 0 else _empty()
        if len(local) < len(remote):      # stage what fits; the peer keeps no surplus
            self.push_staging_shortfalls += 1
            self._index.remote_recycle(rank, remote[len(local):], component=FULL)
        sp = StagedPush(rank, h, hr, hr + len(local), local, remote[:len(local)])
        self._stage_aux(sp, hr + len(remote), runs, ends, rep)
        return sp

    def _stage_aux(self, sp: StagedPush, grant: int, runs: List[Tuple[int, int]],
                   ends: List[int], rep) -> None:
        """Borrow local slots for the aux slots the peer granted; give the peer
        back the rest. The peer granted, in request order, each SWA run's head
        below ``grant`` and each checkpoint at or below it. A lane is
        all-or-none on either side, and all are dropped when the FULL range
        was cut below ``grant`` (they would not end at ``sp.end``)."""
        swa = np.asarray(rep.swa_slots, dtype=np.int32)
        mam = np.asarray(rep.mamba_slots, dtype=np.int32)
        pieces, pos = [], 0
        for a, b in runs:
            k = max(0, min(b, grant) - a)
            pieces.append((a, swa[pos:pos + k]))
            pos += k
        cps = list(zip([e for e in ends if e <= grant], mam.tolist()))
        pieces = [(a, r) for a, r in pieces if len(r)]
        if sp.end < grant:
            pieces, cps = [], []
        for kind, granted, kept in ((SWA, swa, np.concatenate([r for _, r in pieces])
                                     if pieces else _empty()),
                                    (MAMBA, mam, np.asarray([r for _, r in cps], dtype=np.int32))):
            # Pools may differ per node: a grant above this pool must not raise here.
            total = self._index.swa_mempool_total() if kind == SWA else self._index.mamba_mempool_total()
            local = (self._index.allocate_slots(len(kept), kind)
                     if 0 < len(kept) <= int(total) else _empty())
            if len(local) != len(kept):
                self.push_staging_shortfalls += 1
                kept, local = _empty(), _empty()
            extra = granted[~np.isin(granted, kept)]
            if len(extra):
                self._index.remote_recycle(sp.rank, extra, component=kind)
            if not len(kept):
                continue
            if kind == SWA:
                pos = 0
                for a, r in pieces:
                    sp.swa.append((a, local[pos:pos + len(r)], r))
                    pos += len(r)
            else:
                sp.mamba = [(e, int(l), r) for (e, r), l in zip(cps, local.tolist())]

    def submit_push(self, sp: StagedPush, *, timeout_ms: int = 30000,
                    block: bool = True) -> PushJob:
        """WRITE the staged slots to the peer and publish them there. Call only
        after every block is in place. The borrowed slots return to this
        node's pool when the WRITE ends, whatever its outcome."""
        self._consume(sp)
        with self._mu:
            job = PushJob(self._next_id, self)
            self._next_id += 1
            self._jobs[job.job_id] = job
        job._sp = sp
        job.remote_hit = sp.start
        job.planned_push = sp.end - sp.start
        if sp.end <= sp.start:
            job._f.set_result(PushResult(sp.start, 0, 0, sp.rank))
            return job
        job._items = [(sp.rank, int(kind), remote.tolist(), local.tolist(), OP_WRITE)
                      for kind, local, remote in sp._lanes()]
        self._submit_job(job, timeout_ms, block, self._push_complete)
        return job

    def abort_push(self, sp: StagedPush) -> None:
        """Give back both sides of a staged push that will not be submitted."""
        self._consume(sp)
        for kind, local, remote in sp._lanes():
            if len(remote):
                self._index.remote_recycle(sp.rank, remote, component=kind)
            self._recycle(local, kind)

    def _consume(self, sp: StagedPush) -> None:
        # A second submit/abort would recycle the same slots twice.
        with self._mu:
            if sp._used:
                raise RuntimeError("staged push was already submitted or aborted")
            sp._used = True

    def _push_complete(self, job: PushJob, replies: Sequence[TransferReply]) -> None:
        sp = job._sp
        lanes = sp._lanes()
        ok = {kind: bool(rep[0]) for (kind, _, _), rep in zip(lanes, replies)}
        self.push_transfer_failures += sum(not v for v in ok.values())
        entries, pushed, nbytes = [], 0, 0
        for kind, local, remote in lanes:
            self._recycle(local, kind)    # the WRITE has drained: the source is free
            if not (ok[FULL] and ok[kind]):
                self._index.remote_recycle(sp.rank, remote, component=kind)
                continue
            pushed += len(remote)
            nbytes += len(remote) * int(self._store.pool(kind).slot_bytes)
            if kind == FULL:
                entries.append((FULL, sp.end, remote))
            elif kind == SWA:
                entries.extend((SWA, a + len(r), r) for a, _, r in sp.swa)
            else:
                entries.extend((MAMBA, e, np.asarray([r], dtype=np.int32)) for e, _, r in sp.mamba)
        if not ok[FULL]:
            job._f.set_result(PushResult(sp.start, 0, 0, sp.rank))
            return
        res = self._index.remote_commit(sp.rank, sp.hashes[:sp.end], entries)
        # The peer recycled a failed insert itself; a lost reply may hide publication.
        # Either way the reserved slots are no longer ours to recycle.
        if int(res.status) != 0 or res.errors[0] != InsertError.OK or int(res.remote_hit) < sp.end:
            self.push_commit_shortfalls += 1
            committed = 0
        else:
            committed = sp.end
            self.push_commit_shortfalls += sum(e != InsertError.OK for e in res.errors[1:])
        job._f.set_result(PushResult(committed, pushed, nbytes, sp.rank))

    def _submit_job(self, job: Union[PullJob, PushJob], timeout_ms: int,
                    block: bool, complete: Callable) -> None:
        if block:
            self._slots.acquire()
        elif not self._slots.acquire(blocking=False):
            self.saturated += 1
            complete(job, [(False, 0, 0, "client saturated")] * len(job._items))
            return
        try:
            job._ticket = self._t.submit(job._items, timeout_ms)
        except Exception as e:  # noqa: BLE001
            self._slots.release()
            complete(job, [(False, 0, 0, str(e))] * len(job._items))
            return
        with self._mu:
            self._pending[job._ticket] = job
            if self._completer is None:
                self._completer = threading.Thread(target=self._run, name="radixshmem-completer",
                                                   daemon=True)
                self._completer.start()
        self._work.set()

    def _run(self) -> None:
        while not self._closed:
            with self._mu:
                tickets = list(self._pending)
            if not tickets:
                self._work.wait()
                self._work.clear()
                continue
            try:
                rep = self._t.wait_any(tickets, 1000)
                lost = "lost by server"
            except Exception as e:  # noqa: BLE001 - server gone: pending jobs degrade
                rep, lost = {"done": {}, "unknown": tickets}, str(e)
            for ticket in list(rep["done"]) + list(rep["unknown"]):
                with self._mu:
                    job = self._pending.pop(ticket, None)
                if job is None:
                    continue
                replies = rep["done"].get(ticket) or [(False, 0, 0, lost)] * len(job._items)
                complete = self._push_complete if isinstance(job, PushJob) else self._pull_complete
                try:
                    complete(job, replies)
                except Exception as e:  # noqa: BLE001
                    if job._q is not None:
                        job._q.finalize()
                    job._f.set_exception(e)
                self._slots.release()

    def _pull_stage(self, job: PullJob, hit: int) -> int:
        q, me = job._q, int(self._index.rank())
        remote = [(int(r), int(o), np.asarray(s, dtype=np.int32))
                  for r, o, s in q.full_fragments if r >= 0 and r != me]
        job.local_hit = int(remote[0][1]) if remote else hit
        if remote:
            job._src, off, rs = remote[0]
            st = self._index.allocate_slots(len(rs), FULL)
            if len(st) < len(rs):
                self._recycle(st, FULL)
                self.pull_staging_shortfalls += 1
                return off
            job._items.append((job._src, int(FULL), rs.tolist(), st.tolist(), OP_READ))
            job._staged[FULL] = (off, st)
        swa_rank, swa = int(q.swa_source_rank), np.asarray(q.swa_slots, dtype=np.int32)
        if len(swa) and swa_rank >= 0 and swa_rank != me:
            st = self._index.allocate_slots(len(swa), SWA)
            if len(st) == len(swa):
                job._items.append((swa_rank, int(SWA), swa.tolist(), st.tolist(), OP_READ))
                job._staged[SWA] = (0, st)
            else:
                self._recycle(st, SWA)
                self.pull_staging_shortfalls += 1
            job._src = swa_rank if job._src < 0 else job._src
        mamba = q.mamba
        if mamba is not None and int(mamba[0]) >= 0 and int(mamba[0]) != me:
            st = self._index.allocate_slots(1, MAMBA)
            if len(st) == 1:
                job._items.append((int(mamba[0]), int(MAMBA), [int(mamba[1])], st.tolist(),
                                   OP_READ))
                job._staged[MAMBA] = (0, st)
            else:
                self._recycle(st, MAMBA)
                self.pull_staging_shortfalls += 1
            job._src = int(mamba[0]) if job._src < 0 else job._src
        return hit

    def _pull_complete(self, job: PullJob, replies: Sequence[TransferReply]) -> None:
        ok = {ComponentType(k): rep[0] for (_, k, _, _, _), rep in zip(job._items, replies)}
        self.pull_transfer_failures += sum(not v for v in ok.values())
        job._q.finalize()   # unpinned parent absorbs the promoted leaf; an evicted prefix fails the insert (slots auto-recycled)
        hit, pulled, nbytes = job._hit, 0, 0
        if FULL in job._staged:
            off, st = job._staged[FULL]
            if ok[FULL]:
                self._index.insert(job._hashes[:off + len(st)], st, start=off)
                pulled += len(st)
                nbytes += len(st) * int(self._store.pool(FULL).slot_bytes)
            else:
                self._recycle(st, FULL)
                hit = off
        for kind in (SWA, MAMBA):
            if kind not in job._staged:
                continue
            st = job._staged[kind][1]
            if ok.get(FULL, True) and ok[kind]:
                self._index.insert(job._hashes[:hit], st, component=kind)
                pulled += len(st)
                nbytes += len(st) * int(self._store.pool(kind).slot_bytes)
            else:
                self._recycle(st, kind)
        self._pull_finish(job, hit, pulled, nbytes)

    def _pull_finish(self, job: PullJob, hit: int, pulled: int = 0, nbytes: int = 0) -> None:
        lock = job._lock and not job._cancelled
        r = (self._index.query(job._hashes[:hit], mask=job._mask, local_only=True,
                               lock=lock, update_meta=False) if hit > 0 else None)
        common = int(r.common_hit) if r is not None and int(r.status) == 0 else 0
        if common < hit:
            self.pull_promotion_shortfalls += 1
        if common == 0:
            if r is not None:
                r.finalize()
            res = PullResult(0, _empty(), 0, _empty(), -1, pulled, nbytes, job._src, lambda: None)
        else:
            runs = [np.asarray(s, dtype=np.int32) for _, _, s in r.full_fragments]
            full = np.concatenate(runs)[:common] if runs else _empty()
            res = PullResult(common, full, int(r.swa_start),
                            np.asarray(r.swa_slots, dtype=np.int32),
                            int(r.mamba[1]) if r.mamba else -1, pulled, nbytes, job._src,
                            r.finalize)
        with self._mu:
            job._f.set_result(res)
            if job._cancelled:      # cancelled between the pin above and here
                res.finalize()

    def _recycle(self, slots, kind: ComponentType) -> None:
        if len(slots):
            self._index.recycle_slots(np.asarray(slots, dtype=np.int32), kind)
