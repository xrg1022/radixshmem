#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Roles of the two-machine push_async smoke; two_node_push_smoke.sh drives them.

server  one RadixServer per host (Mooncake data plane on mlx5_bond_0, remote ops on),
        prints READY, writes /tmp/tn_<cluster>.json with the attach name, exits on SIGTERM
push    seed a byte-patterned prefix locally, push_async(rank=None[, evict_local]) it
verify  the prefix is in the local tree and every slot's bytes match the pattern
pull    pull_async the prefix from the peer and verify the bytes
health  check_health of the local tree
probe   the Mooncake backend comes up on mlx5_bond_0 (build check)
"""

import argparse
import json
import os
import signal
import sys
import time

import numpy as np

import radixshmem
from radixshmem import ClusterConfig, Geometry, RadixClient, RadixServer, ServerConfig

FULL, SWA, MAMBA = radixshmem.ComponentType.FULL, radixshmem.ComponentType.SWA, radixshmem.ComponentType.MAMBA
DEV, GID = "mlx5_bond_0", 3
BLOCKS, MAMBA_SLOTS, WINDOW = 200_000, 10_000, 4
SLOT_BYTES = {FULL: 128, SWA: 64, MAMBA: 256}
OK = radixshmem.InsertError.OK


def mask_of(aux):
    return radixshmem.COMPONENT_MASK_FULL | ({"swa": radixshmem.COMPONENT_MASK_SWA,
                                            "mamba": radixshmem.COMPONENT_MASK_MAMBA}.get(aux, 0))


def hashes_of(n, seed):
    return np.random.default_rng(seed).integers(1, 2**63 - 1, size=n, dtype=np.uint64)


def pattern(kind, h, words):
    """(k, words) uint64 words per block: FULL counts up from the hash, SWA from its complement, Mamba down."""
    h = np.asarray(h, dtype=np.uint64)
    if kind == SWA:
        h = ~h
    step = np.arange(words, dtype=np.uint64)
    return h[:, None] - step if kind == MAMBA else h[:, None] + step


def pools(c):
    st, out = c.store, {}
    for kind in (FULL, SWA, MAMBA):
        if st.has_pool(kind):
            p = st.pool(kind)
            out[kind] = np.frombuffer(st.pool_view(kind), dtype=np.uint64).reshape(
                int(p.num_slots), int(p.slot_bytes) // 8)
    return out


def words(kind):
    return SLOT_BYTES[kind] // 8


def info_path(a):   # one server per host normally; --node-name keeps two apart on one host
    return f"/tmp/tn_{a.cluster_id}{'_' + a.node_name if a.node_name else ''}.json"


def client_of(a):
    with open(info_path(a)) as f:
        return RadixClient(json.load(f)["name"])


def emit(d, ok):
    print(json.dumps(d), flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------- roles

def role_server(a):
    swa, mamba = a.aux == "swa", a.aux == "mamba"
    node = a.node_name or "n" + a.node_ip.replace(".", "_")
    # The budget is exactly the pools' bytes, so the ratios plan BLOCKS / MAMBA_SLOTS slots.
    swa_bytes = BLOCKS * SLOT_BYTES[SWA] if swa else 0
    mamba_bytes = MAMBA_SLOTS * SLOT_BYTES[MAMBA] if mamba else 0
    data_bytes = BLOCKS * SLOT_BYTES[FULL] + swa_bytes + mamba_bytes
    cfg = ServerConfig(   # per-node shm name: the gRPC socket derives from it (two servers on one host)
        name=f"/tn_{a.cluster_id}_{node}", data_bytes=data_bytes,
        swa_ratio=swa_bytes / data_bytes, mamba_ratio=mamba_bytes / data_bytes, slot_align=64,
        transfer_devices=[DEV], transfer_ip=a.node_ip, max_nodes=100_000, background_evict_ratio=0.0,
        cluster=ClusterConfig(expected_min_nodes=2, registry=a.registry, cluster_id=a.cluster_id,
                              node_name=node, rpc_address=a.node_ip, num_rht_shards=2,
                              rht_shard_holders=[0, 1], index_dev=DEV, gid_idx=GID,
                              bootstrap_timeout_sec=180, rht_slots_per_bucket=4,
                              enable_remote_insert=True, enable_remote_query=True,
                              remote_op_transport=a.remote_op))
    geo = Geometry(block_size=16, full_slot_bytes=SLOT_BYTES[FULL],
                   swa_slot_bytes=SLOT_BYTES[SWA] if swa else 0, swa_window_blocks=WINDOW if swa else 0,
                   mamba_slot_bytes=SLOT_BYTES[MAMBA] if mamba else 0, slot_align=64)
    srv = RadixServer(cfg).start()
    c = RadixClient(cfg.name, geo)       # hands the server its geometry; the cluster bootstraps behind it
    c.wait_ready(cfg.cluster.bootstrap_timeout_sec + 120)
    with open(info_path(a), "w") as f:   # clients attach by the configured name (its socket), not the resolved one
        json.dump({"name": cfg.name, "rank": int(c.rank())}, f)
    print(f"READY rank={c.rank()} world={c.world_size()} name={c.info.index_name} node={node}", flush=True)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    while not stop:
        time.sleep(0.2)
    h = int(c.check_health())
    print(f"EXIT rank={c.rank()} health={h} {c.health_report().describe().strip()}", flush=True)
    c.close()
    srv.close()
    return 0 if h == 0 else 3


def seed_local(c, h, aux):
    """Land hashes[:n] locally on top of whatever prefix the tree already holds."""
    n, P = len(h), pools(c)
    have = int(c.query(h, mask=radixshmem.COMPONENT_MASK_FULL, local_only=True, lock=False,
                       update_meta=False).common_hit)
    if have < n:
        s = c.allocate_slots(n - have)
        assert len(s) == n - have, "FULL pool short"
        P[FULL][s, :words(FULL)] = pattern(FULL, h[have:], words(FULL))
        assert c.insert(h, s, start=have).error == OK
    if aux == "swa":
        k = min(WINDOW, n)
        s2 = c.allocate_slots(k, SWA)
        assert len(s2) == k, "SWA pool short"
        P[SWA][s2, :words(SWA)] = pattern(SWA, h[n - k:], words(SWA))
        assert c.insert(h, s2, start=0, component=SWA).error == OK
    elif aux == "mamba":
        m = c.allocate_slots(1, MAMBA)
        assert len(m) == 1, "MAMBA pool short"
        P[MAMBA][m, :words(MAMBA)] = pattern(MAMBA, h[n - 1:], words(MAMBA))
        assert c.insert(h, m, component=MAMBA).error == OK


def bad_rows(view, slots, kind, h):
    w = words(kind)
    slots = np.asarray(slots, dtype=np.int32)
    if slots.size == 0:
        return 0
    return int(np.count_nonzero((view[slots, :w] != pattern(kind, h, w)).any(axis=1)))


def verify_local(c, h, aux):
    q = c.query(h, mask=mask_of(aux), local_only=True, lock=False, update_meta=False)
    hit, P = int(q.common_hit), pools(c)
    out = {"hit": hit, "bad_full": 0, "bad_swa": 0, "bad_mamba": 0, "swa_cells": 0, "mamba_slot": -1}
    if hit:
        full = np.concatenate([np.asarray(f[2], dtype=np.int32) for f in q.full_fragments])[:hit]
        out["bad_full"] = bad_rows(P[FULL], full, FULL, h[:hit])
    if aux == "swa" and hit:
        sw = np.asarray(q.swa_slots, dtype=np.int32)
        out["swa_cells"] = int(sw.size)
        out["bad_swa"] = bad_rows(P[SWA], sw, SWA, h[int(q.swa_start):hit])
    if aux == "mamba" and hit:
        mb = q.mamba                      # None or (source_rank, slot_id) at common_hit
        slot = int(mb[1]) if mb is not None else -1
        out["mamba_slot"] = slot
        if slot >= 0:
            out["bad_mamba"] = bad_rows(P[MAMBA], [slot], MAMBA, h[hit - 1:hit])
    return out


def role_push(a):
    c = client_of(a)
    h = hashes_of(a.n, a.seed)
    seed_local(c, h, a.aux)
    # A fresh prefix has no holder anywhere, so the RHT cannot name a target: `peer` pushes to
    # the other rank explicitly; `auto` (rank=None) is for a prefix the peer already partly holds.
    rank = None if a.target == "auto" else (int(c.rank()) + 1) % int(c.world_size())
    t0 = time.perf_counter()
    r = c.push_async(h, mask_of(a.aux), rank=rank, evict_local=a.evict).wait(300)
    dt = time.perf_counter() - t0
    after = verify_local(c, h, a.aux)
    d = {"role": "push", "target": a.target, "own_rank": int(c.rank()), "target_rank": r.target_rank, "committed": r.committed,
         "remote_blocks": r.remote_blocks, "remote_bytes": r.remote_bytes, "local_evicted": r.local_evicted,
         "push_s": round(dt, 4), "local_hit_after": after["hit"],
         "push_transfer_failures": c.push_transfer_failures,
         "push_staging_shortfalls": c.push_staging_shortfalls,
         "push_commit_shortfalls": c.push_commit_shortfalls, "health": int(c.check_health())}
    ok = (r.committed == a.n and r.target_rank >= 0 and r.target_rank != c.rank() and d["health"] == 0
          and (r.local_evicted == a.n and after["hit"] == 0 if a.evict else after["hit"] == a.n))
    c.close()
    return emit(d, ok)


def role_verify(a):
    c = client_of(a)
    h = hashes_of(a.n, a.seed)
    res = verify_local(c, h, a.aux)
    expect = a.n if a.expect_hit is None else a.expect_hit
    d = {"role": "verify", "own_rank": int(c.rank()), "expect_hit": expect, **res, "health": int(c.check_health())}
    ok = (res["hit"] == expect and res["bad_full"] == res["bad_swa"] == res["bad_mamba"] == 0
          and d["health"] == 0 and (expect == 0 or a.aux != "swa" or res["swa_cells"] == min(WINDOW, a.n))
          and (expect == 0 or a.aux != "mamba" or res["mamba_slot"] >= 0))
    c.close()
    return emit(d, ok)


def role_pull(a):
    c = client_of(a)
    h = hashes_of(a.n, a.seed)
    P = pools(c)
    t0 = time.perf_counter()
    r = c.pull_async(h, mask_of(a.aux), lock=True).wait(300)
    dt = time.perf_counter() - t0
    n = int(r.common_hit)
    bad = bad_rows(P[FULL], np.asarray(r.full_slots, dtype=np.int32)[:n], FULL, h[:n]) if n else 0
    if a.aux == "swa" and len(r.swa_slots):
        bad += bad_rows(P[SWA], r.swa_slots, SWA, h[int(r.swa_start):n])
    if a.aux == "mamba" and r.mamba_slot >= 0:
        bad += bad_rows(P[MAMBA], [int(r.mamba_slot)], MAMBA, h[n - 1:n])
    r.finalize()
    d = {"role": "pull", "own_rank": int(c.rank()), "common_hit": n, "remote_blocks": r.remote_blocks,
         "remote_bytes": r.remote_bytes, "source_rank": r.source_rank, "bad": bad, "pull_s": round(dt, 4),
         "health": int(c.check_health())}
    ok = n == a.n and r.remote_blocks > 0 and bad == 0 and d["health"] == 0
    c.close()
    return emit(d, ok)


def role_health(a):
    c = client_of(a)
    hlt = int(c.check_health())
    d = {"role": "health", "own_rank": int(c.rank()), "health": hlt,
         "report": c.health_report().describe().strip(), "blocks": int(c.total_blocks()),
         "mempool_used": int(c.mempool_used())}
    c.close()
    return emit(d, hlt == 0)


def role_probe(a):
    from radixshmem import _data
    b = _data.create_transfer_backend(_data.TransferConfig(
        backend="mooncake", metadata="P2PHANDSHAKE", ip_or_host="127.0.0.1", rpc_port=0,
        protocol="rdma", device_filter=[DEV], publish=False))
    print("PROBE_OK" if b is not None else "PROBE_FAIL", flush=True)
    return 0 if b is not None else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("role", choices=["server", "push", "verify", "pull", "health", "probe"])
    p.add_argument("--node-ip", default="127.0.0.1")
    p.add_argument("--node-name", default="")
    p.add_argument("--registry", default="etcd://127.0.0.1:2379")
    p.add_argument("--cluster-id", default="tn")
    p.add_argument("--aux", choices=["swa", "mamba", "none"], default="swa")
    p.add_argument("--remote-op", choices=["zmq", "dc"], default="zmq")
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--evict", action="store_true")
    p.add_argument("--target", choices=["peer", "auto"], default="peer",
                   help="push: explicit other rank, or rank=None (RHT) for a prefix the peer already holds")
    p.add_argument("--expect-hit", type=int, default=None)
    a = p.parse_args()
    return globals()["role_" + a.role](a)


if __name__ == "__main__":
    sys.exit(main())
