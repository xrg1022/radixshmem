# RadixShmem

RadixShmem is a KV cache index built on a distributed radix tree. The architecture is decentralized: each node
keeps its own radix tree in POSIX shared memory. Each tree is both a local and a shared index. It tracks the
KV cache blocks held in the node's DRAM, and any process on the node can attach to it and insert or look up
blocks by prefix hash. Other nodes can also query it: a sharded distributed hash table records which node
holds each prefix, and the querying node reads the peer's tree with one-sided RDMA, so the KV cache is indexed
globally across the cluster. An inference engine integrates with RadixShmem through one client interface,
which handles both local and cross-node lookups. The recommended setup uses RadixShmem as the index together
with an existing transfer engine, such as NIXL or Mooncake, to enable peer-to-peer KV cache sharing.

## Framework support

- sglang: HiCache L3 storage backend, in progress.
- vLLM: KV offload backend, in progress.

## Using RadixShmem

### Build

```bash
cd python && pip install -e .
```

The optional data-plane backends are built separately; see [Transfer backends](#transfer-backends).

### Start the server

Each node runs one `radix-server`, which manages `--data-bytes` of host memory; the slot size is configured
by the client. In a distributed setup, `--index-dev` and `--transfer-dev` select the devices for the control
plane and the data plane.

Single node:

```bash
radix-server --name /my_kv_cache --data-bytes 1024G
```

4 nodes, coordinated through etcd:

```bash
# Same command on every node except --rpc-address
radix-server --name /my_kv_cache --data-bytes 1024G \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.10 --index-dev mlx5_bond_0 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

Enable the SWA pool:

```bash
radix-server --name /my_kv_cache --data-bytes 1024G --swa-ratio 0.2 \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.10 --index-dev mlx5_bond_0 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

Enable the Mamba pool:

```bash
radix-server --name /my_kv_cache --data-bytes 1024G --mamba-ratio 0.2 \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.10 --index-dev mlx5_bond_0 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

### Client API

A client connects to the local server by name and provides `query`, `pull_async`, `insert`, and the staged
push `stage_push` / `submit_push` / `abort_push`.

1. **Initialize.** Construct the client with the server name and a `Geometry` (tokens per block and bytes per
   slot), then wait until the server is ready.

```python
import radixshmem

geometry = radixshmem.Geometry(block_size=16, full_slot_bytes=1179648)
client = radixshmem.RadixClient("/my_kv_cache", geometry)
client.wait_ready(timeout_s=600)
```

2. **Query.** `query()` matches the prefix in the local radix tree and, in a cluster, in peer radix trees.
   `common_hit` is the number of blocks hit; `full_fragments` lists the hit as `(source_rank, offset, slots)`
   runs, and the slot ids are valid in the SlotStore of `source_rank`.

```python
hashes = radixshmem.compute_hashes(token_ids, tokens_per_block=16)
r = client.query(hashes)
# 8 blocks, the first 3 cached locally (rank 0) and the first 6 on rank 1:
# r.common_hit == 6
# r.full_fragments == [(0, 0, [12, 40, 41]), (1, 3, [7, 8, 9])]
```

3. **Pull.** `pull_async()` is the data-plane operation: it transfers the KV cache hit on a peer into local
   slots and publishes those blocks to the local radix tree.

```python
r = client.pull_async(hashes).wait()   # [0, r.common_hit) is now in local slots
for slot in r.full_slots:
    client.slot_view(slot)             # memoryview of the slot
r.finalize()                           # release the pin on the hit blocks
```

4. **Insert.** To add new KV cache, query the local hit, allocate slots for the remaining blocks, copy the KV
   from GPU straight into the slots through their memoryviews (zero-copy D2H), then insert.

```python
matched = client.query(hashes, local_only=True).common_hit
slots = client.allocate_slots(len(hashes) - matched)
for i, slot in enumerate(slots):
    dst = torch.frombuffer(client.slot_view(slot), dtype=torch.uint8)
    dst.copy_(gpu_blocks[matched + i])  # gpu_blocks[k]: KV of block k as a flat uint8 tensor
client.insert(hashes, slots, start=matched)
```

5. **Push.** A staged push writes blocks a node computed into a peer's SlotStore and publishes them only in
   the peer's index. `stage_push()` asks the peer for its hit and reserves slots for the blocks it lacks (one
   `remote_prepare`), and borrows as many local slots, which this node never publishes. The caller writes block
   `start + i` into `local_slots[i]` (for example a D2H copy), then `submit_push()` has the server WRITE them and
   commit them on the peer; the borrowed slots return to the pool when the WRITE ends. `abort_push()` gives both
   sides back. The target is `rank`, else the peer the RHT routes the deepest prefix to, else `fallback_rank`.
   The target needs `--enable-remote-insert`; RHT target selection needs `--rht-slots` of 2 or more.
   With SWA or Mamba, pass the window (`swa=[(end - W, end)]`) or the checkpoint (`mamba=[end]`) that ends the
   prefix; they are staged in `sp.swa` / `sp.mamba` and committed with the FULL blocks.

```python
sp = client.stage_push(hashes, fallback_rank=p_rank)   # peer already holds hashes[:sp.start]
for i, slot in enumerate(sp.local_slots):
    dst = torch.frombuffer(client.slot_view(slot), dtype=torch.uint8)
    dst.copy_(gpu_blocks[sp.start + i])
r = client.submit_push(sp).wait()   # [0, r.committed) is now on rank r.target_rank
```

`python tests/agentic_stress.py --nodes 4 --push --block-size 128 --turns 3,6` exercises it (data mode, more
than one node); a turn longer than `MAX_REMOTE_OP_BLOCKS` hashes is not pushed and is counted as
`push.capped`, so keep the sessions under the cap with `--block-size` / `--turns`.

The full API (methods, parameters, result fields, error codes) is in [`docs/api.md`](docs/api.md).

## Transfer backends

### 1. NIXL

Build NIXL:

```bash
scripts/build_nixl.sh
cd python && pip install -e .
```

Start the server with `--transfer-backend nixl`. NIXL cannot pick its own port, so `--transfer-port` is
required:

```bash
radix-server --name /my_kv_cache --data-bytes 1024G \
    --transfer-backend nixl --transfer-port 17000 \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.10 --index-dev mlx5_bond_0 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

### 2. Mooncake

Build Mooncake:

```bash
scripts/build_mooncake.sh
cd python && pip install -e .
```

Start the server with `--transfer-backend mooncake`:

```bash
radix-server --name /my_kv_cache --data-bytes 1024G \
    --transfer-backend mooncake \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.10 --index-dev mlx5_bond_0 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

To check which backends were built in:

```bash
python -c "import radixshmem; print(radixshmem._data.TRANSFER_BACKENDS)"
```

## License

RadixShmem is released under the Apache License 2.0; the full text is in [LICENSE](LICENSE). Source files
carry an SPDX Apache-2.0 identifier and the NVIDIA copyright notice.
