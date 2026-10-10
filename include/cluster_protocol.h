// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// TCP bootstrap wire format between RadixClient and RadixServer (Hello/Reply
// per peer, then pure RDMA). Fixed-size, native endian; version mismatch is fatal.

#pragma once

#include <cstddef>
#include <cstdint>
#include <infiniband/verbs.h>

#include "cluster_manifest.h"   // CLUSTER_MAX_SHARDS
#include "types.h"              // RemoteStatus (shared with QueryResult)

namespace radixshmem {

constexpr uint32_t CLUSTER_PROTOCOL_VERSION = 11;  // lockstep with manifest v11
constexpr uint32_t CLUSTER_HELLO_MAGIC = 0x52584348; // "RXCH"
constexpr uint32_t CLUSTER_REPLY_MAGIC = 0x52584352; // "RXCR"
constexpr uint32_t CLUSTER_SHARDMAP_MAGIC = 0x52584d50; // "RXMP"

// rank 0 → worker: authoritative shard map, sent after rank 0 resolves it.
struct ShardMapMsg {
    uint32_t magic;          // CLUSTER_SHARDMAP_MAGIC
    uint32_t version;        // CLUSTER_PROTOCOL_VERSION
    uint32_t num_shards;     // M
    uint32_t rht_capacity;   // per-shard bucket count
    int32_t  rdma_sl;        // authoritative QoS (rank 0's env), see manifest
    int32_t  rdma_tc;
    uint32_t rht_transport;         // authoritative RHT-plane transport (ClusterTransport)
    uint32_t peer_index_transport;  // authoritative peer-index-plane transport
    uint32_t rht_slots;      // authoritative W (RHT slots per bucket)
    uint64_t dc_access_key;  // cluster-wide DCT key (when either plane is DC)
    uint32_t remote_op_transport;   // authoritative remote-op transport (RemoteOpTransport)
    uint32_t register_chunk_size;   // cs; node-boundary / probe stride alignment
    uint32_t holder_rank[CLUSTER_MAX_SHARDS];  // shard s → holder rank
};

// Client → Server, sent after TCP connect: client's RESET XRC SEND QP asks for
// a paired RECV QP on the server's shared SRQ.
struct ClusterHelloMsg {
    uint32_t      magic;
    uint32_t      version;
    int32_t       client_rank;          // RadixClient rank (informational)
    uint32_t      client_send_qp_num;   // XRC SEND QP the client just created

    uint8_t       link_layer;
    uint8_t       _pad0[3];
    uint16_t      lid;
    uint16_t      _pad1;
    int32_t       gid_index;
    union ibv_gid gid;
};

// Server → Client, after the server brings up the per-client XRC RECV QP.
struct ClusterReplyMsg {
    uint32_t      magic;
    uint32_t      version;
    int32_t       server_rank;          // peer's rank (informational; client also has it from manifest)
    uint32_t      server_recv_qp_num;   // client SEND QP RTR-targets this
    uint32_t      server_srq_num;       // wr.qp_type.xrc.remote_srqn

    uint8_t       link_layer;
    uint8_t       _pad0[3];
    uint16_t      lid;
    uint16_t      _pad1;
    int32_t       gid_index;
    union ibv_gid gid;
};

constexpr uint32_t REMOTE_OP_PROTOCOL_VERSION = 10;
constexpr uint32_t REMOTE_OP_REQ_MAGIC   = 0x52494E53; // "RINS"
constexpr uint32_t REMOTE_OP_REPLY_MAGIC = 0x52494E52; // "RINR"
constexpr uint16_t REMOTE_OP_INSERT_RESERVE = 1;   // allocate n slots (n=count, no hashes, no dedup)
constexpr uint16_t REMOTE_OP_INSERT         = 2;   // phase 2: the real tree insert + RHT publish
constexpr uint16_t REMOTE_OP_QUERY          = 3;   // target queries its own tree (lock = pin)
constexpr uint16_t REMOTE_OP_QUERY_RELEASE  = 4;   // unpin a query lock (hash-path dec_ref)
constexpr uint16_t REMOTE_OP_INSERT_RELEASE = 5;   // explicitly recycle reserved slots
constexpr uint16_t REMOTE_OP_COMMIT         = 6;
constexpr uint16_t REMOTE_OP_PREPARE        = 7;

// INSERT flags. 0 = target auto-recycles unused reservation; set = caller
// recycles explicitly via REMOTE_OP_INSERT_RELEASE.
constexpr uint16_t REMOTE_OP_FLAG_NO_AUTO_RECYCLE = 0x1;

// Caps root-based prefixes and sizes transport buffers.
constexpr int      MAX_REMOTE_OP_BLOCKS = 8192;

struct RemoteOpReqHeader {
    uint32_t magic;                // REMOTE_OP_REQ_MAGIC
    uint16_t version;              // REMOTE_OP_PROTOCOL_VERSION
    uint16_t op;                   // discriminant: REMOTE_OP_* above
    uint16_t flags;                // REMOTE_OP_FLAG_* (INSERT: recycle control)
    uint8_t  component;            // ComponentType pool for INSERT*; FULL for QUERY*
    uint8_t  _pad;
    int32_t  n;                    // number of hashes (<= MAX); payload length
    uint64_t req_id;               // echoed back for matching / late-reply discard
    union {
        struct { } reserve;                                  // empty
        struct { int32_t start; int32_t length; } insert;    // range; INSERT/RELEASE: length = slot count; SWA/MAMBA: start = 0
        struct { uint8_t local_only; uint8_t lock; uint8_t update_meta; uint8_t _pad; } query;   // lock=1 → two-phase, A must RELEASE; update_meta=1 → touch LRU
        struct { int32_t num_entries; int32_t num_slots; } commit;
        struct { int32_t full_hit; int32_t num_entries; } prepare;
    } u;
};
// followed by uint64_t hashes[n] (RESERVE: none, n = alloc count); INSERT/INSERT_RELEASE also int32_t slots[length]
static_assert(sizeof(RemoteOpReqHeader) == 32,
              "RemoteOpReqHeader layout changed — bump REMOTE_OP_PROTOCOL_VERSION");
static_assert(alignof(RemoteOpReqHeader) == 8, "RemoteOpReqHeader alignment changed");

// COMMIT/PREPARE payload: hashes[n], entries[num_entries]; COMMIT appends slots[num_slots].
struct RemoteCommitEntry {
    uint8_t component;
    uint8_t _pad;
    uint16_t num_slots;
    int32_t at;
};
static_assert(sizeof(RemoteCommitEntry) == 8, "RemoteCommitEntry layout changed");

inline bool valid_remote_entry(ComponentType component, int at, size_t num_slots,
                               int limit, bool require_full, int (&totals)[3]) {
    if (!component_type_valid(component) || at <= 0 || at > limit ||
        num_slots == 0 || num_slots > (size_t)at ||
        (require_full ? component != ComponentType::FULL || at != limit
                      : component == ComponentType::FULL) ||
        (component == ComponentType::MAMBA && num_slots != 1)) return false;
    auto& total = totals[(int)component];
    total += (int)num_slots;
    return total <= limit;
}

// Reply: target → initiator. INSERT_RESERVE / INSERT / QUERY replies = header +
// num_slots int32 payload (reserved / consumed / matched slots respectively).
struct RemoteOpReplyHeader {
    uint32_t magic;                // REMOTE_OP_REPLY_MAGIC
    uint16_t version;              // REMOTE_OP_PROTOCOL_VERSION
    uint16_t op;                   // echoes the request op
    uint16_t status;               // RemoteStatus (request-level)
    uint16_t error;                // InsertError, meaningful only when status==OK; INSERT: 0=OK
    uint64_t req_id;               // echoes the request req_id
    union {
        struct {                   // ALLOCATE (RESERVE): pure slot alloc
            int32_t matched_prefix;   // always 0 (no server-side dedup)
            int32_t inserted_count;   // slots allocated = num_slots
            int32_t num_slots;        // == inserted_count
        } reserve;
        struct {                   // INSERT publish ack (also *_RELEASE: fields 0)
            int32_t num_unused;       // unused slots (redundant + unlanded); payload follows
        } insert;
        struct {                   // QUERY
            int32_t hit_length;       // B local-tree total matched blocks
            int32_t num_slots;        // == hit_length; payload follows
        } query;
        struct { int32_t hit_length; int32_t num_entries; } commit; // uint32_t errors follow
        struct { int32_t hit_length; int32_t num_full; int32_t num_swa; int32_t num_mamba; } prepare;
    } u;
};

static_assert(sizeof(RemoteOpReplyHeader) == 40,
              "RemoteOpReplyHeader layout changed — bump REMOTE_OP_PROTOCOL_VERSION");
static_assert(alignof(RemoteOpReplyHeader) == 8, "RemoteOpReplyHeader alignment changed");

// Max wire sizes derived from the caps above (single source of truth for buffers).
constexpr uint32_t REMOTE_OP_MAX_REQ_BYTES =
    (uint32_t)sizeof(RemoteOpReqHeader) +
    (uint32_t)MAX_REMOTE_OP_BLOCKS *
        (uint32_t)(sizeof(uint64_t) + 3 * sizeof(int32_t)) +
    (uint32_t)(1 + 2 * MAX_REMOTE_OP_BLOCKS) * sizeof(RemoteCommitEntry);
constexpr uint32_t REMOTE_OP_MAX_REPLY_BYTES =
    (uint32_t)sizeof(RemoteOpReplyHeader) +
    (uint32_t)(3 * MAX_REMOTE_OP_BLOCKS) * (uint32_t)sizeof(int32_t);

} // namespace radixshmem
