// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Client half of the remote-op plane: RemotePeer, a request/reply session to a
// peer's local tree. Server half lives in remote_op_service.cpp.
#include "radix_client.h"          // RemotePeer decl + reply structs

#ifndef RADIXSHMEM_NO_RDMA

#include "cluster_protocol.h"       // remote-op wire structs
#include "bilateral_transport.h"    // IRemoteOpClientTransport

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace radixshmem {

namespace {

std::vector<char> build_req(uint16_t op, const uint64_t* hashes, int n,
                            uint64_t req_id, ComponentType component,
                            RemoteOpReqHeader*& hdr) {
    hdr = nullptr;
    if (n < 0 || n > MAX_REMOTE_OP_BLOCKS) return {};
    std::vector<char> req(sizeof(RemoteOpReqHeader) + (size_t)n * sizeof(uint64_t));
    hdr = reinterpret_cast<RemoteOpReqHeader*>(req.data());
    *hdr = {};
    hdr->magic     = REMOTE_OP_REQ_MAGIC;
    hdr->version   = REMOTE_OP_PROTOCOL_VERSION;
    hdr->op        = op;
    hdr->component = static_cast<uint8_t>(component);
    hdr->n         = n;
    hdr->req_id    = req_id;
    if (n > 0)
        std::memcpy(hdr + 1, hashes, (size_t)n * sizeof(uint64_t));
    return req;
}

// Guard against a reply that under-delivers the trailing slots[] array.
bool read_slots(const std::vector<char>& reply, int num_slots,
                std::vector<int32_t>& out) {
    size_t need = sizeof(RemoteOpReplyHeader) + (size_t)num_slots * sizeof(int32_t);
    if (num_slots < 0 || reply.size() < need) return false;
    auto* slots = reinterpret_cast<const int32_t*>(
        reinterpret_cast<const RemoteOpReplyHeader*>(reply.data()) + 1);
    out.assign(slots, slots + num_slots);
    return true;
}

}  // namespace

RemotePeer::RemotePeer(
    std::unique_ptr<IRemoteOpClientTransport> transport, std::string endpoint)
    : transport_(std::move(transport)), endpoint_(std::move(endpoint)) {}

RemotePeer::~RemotePeer() {
    auto g = guard();
    transport_.reset();
}
RemotePeer::RemotePeer(RemotePeer&&) noexcept = default;
RemotePeer& RemotePeer::operator=(RemotePeer&&) noexcept = default;

RemoteStatus RemotePeer::request(const void* req, size_t req_len,
                                 std::vector<char>& reply_out, int max_attempts) {
    // next_req_id_ was just stamped into the request header by the caller.
    return transport_->send_request(req, req_len, next_req_id_, reply_out, max_attempts);
}

AllocateReply RemotePeer::allocate(int n, ComponentType component) {
    auto g = guard();
    AllocateReply out;
    // Header-only: n counts slots in the component pool, not hashes.
    if (n < 0 || n > MAX_REMOTE_OP_BLOCKS) {
        out.status = RemoteStatus::BAD_REQUEST; return out;
    }
    std::vector<char> req(sizeof(RemoteOpReqHeader));
    auto* h = reinterpret_cast<RemoteOpReqHeader*>(req.data());
    *h = {};
    h->magic     = REMOTE_OP_REQ_MAGIC;
    h->version   = REMOTE_OP_PROTOCOL_VERSION;
    h->op        = REMOTE_OP_INSERT_RESERVE;
    h->component = static_cast<uint8_t>(component);
    h->n         = n;
    h->req_id    = next_req_id_ + 1;
    ++next_req_id_;
    // flags stays 0: bit0 is reserved.

    std::vector<char> reply;
    RemoteStatus st = request(req.data(), req.size(), reply);
    if (st != RemoteStatus::OK) { out.status = st; return out; }

    auto* r = reinterpret_cast<RemoteOpReplyHeader*>(reply.data());
    out.status = static_cast<RemoteStatus>(r->status);
    if (out.status != RemoteStatus::OK) return out;   // BAD_REQUEST / UNKNOWN_OP
    if (!read_slots(reply, r->u.reserve.num_slots, out.slots))
        out.status = RemoteStatus::BAD_REPLY;
    return out;
}

RemoteStatus RemotePeer::send_ack_op(uint16_t op, const uint64_t* hashes, int n,
                                 int start, int length,
                                 const int32_t* slots,
                                 uint16_t flags, ComponentType component) {
    auto g = guard();
    // Only the two RELEASE ops route here now. INSERT_RELEASE carries a slot
    // count; QUERY_RELEASE length is a hash range (<=n).
    const bool slot_op = (op == REMOTE_OP_INSERT_RELEASE);
    if (length < 0 || (slot_op ? length > MAX_REMOTE_OP_BLOCKS
                               : length > n)) return RemoteStatus::BAD_REQUEST;
    RemoteOpReqHeader* h = nullptr;
    auto req = build_req(op, hashes, n, next_req_id_ + 1, component, h);
    if (!h) return RemoteStatus::BAD_REQUEST; // over the cap: nothing sent
    if (slots && length > 0) {
        size_t off = req.size();
        req.resize(off + (size_t)length * sizeof(int32_t));
        h = reinterpret_cast<RemoteOpReqHeader*>(req.data());
        std::memcpy(req.data() + off, slots, (size_t)length * sizeof(int32_t));
    }
    ++next_req_id_;
    h->flags           = flags;
    h->u.insert.start  = start;
    h->u.insert.length = length;

    std::vector<char> reply;
    RemoteStatus st = request(req.data(), req.size(), reply);
    if (st != RemoteStatus::OK) return st;
    auto* r = reinterpret_cast<RemoteOpReplyHeader*>(reply.data());
    RemoteStatus rst = static_cast<RemoteStatus>(r->status);
    if (rst != RemoteStatus::OK) return rst;   // target-side reject
    return r->error == 0 ? RemoteStatus::OK : RemoteStatus::INTERNAL;
}

InsertResult RemotePeer::insert(const uint64_t* hashes, int n,
                                int start, int length, const int32_t* slots,
                                bool auto_recycle, ComponentType component) {
    auto g = guard();
    InsertResult out;
    if (length < 0 || length > MAX_REMOTE_OP_BLOCKS ||
        (length > 0 && !slots)) {              // slots are the payload now
        out.error = InsertError::REMOTE_FAILED;
        return out;
    }
    RemoteOpReqHeader* h = nullptr;
    auto req = build_req(REMOTE_OP_INSERT, hashes, n, next_req_id_ + 1, component, h);
    if (!h) { out.error = InsertError::REMOTE_FAILED; return out; }
    if (slots && length > 0) {
        size_t off = req.size();
        req.resize(off + (size_t)length * sizeof(int32_t));
        h = reinterpret_cast<RemoteOpReqHeader*>(req.data());
        std::memcpy(req.data() + off, slots, (size_t)length * sizeof(int32_t));
    }
    ++next_req_id_;
    h->flags           = auto_recycle ? (uint16_t)0
                                      : REMOTE_OP_FLAG_NO_AUTO_RECYCLE;
    h->u.insert.start  = start;
    h->u.insert.length = length;

    std::vector<char> reply;
    RemoteStatus st = request(req.data(), req.size(), reply);
    if (st != RemoteStatus::OK) { out.error = InsertError::REMOTE_FAILED; return out; }

    auto* r = reinterpret_cast<RemoteOpReplyHeader*>(reply.data());
    if (static_cast<RemoteStatus>(r->status) != RemoteStatus::OK) {
        out.error = InsertError::REMOTE_FAILED;   // request-level reject (BAD_REQUEST / UNKNOWN_OP)
        return out;
    }
    out.error = static_cast<InsertError>(r->error);
    read_slots(reply, r->u.insert.num_unused, out.unused_slots);
    return out;
}

CommitReply RemotePeer::commit(const uint64_t* hashes, int n,
                               const std::vector<CommitEntry>& entries) {
    auto g = guard();
    CommitReply out;
    out.status = RemoteStatus::BAD_REQUEST;
    if (n <= 0 || n > MAX_REMOTE_OP_BLOCKS || !hashes || entries.empty()) return out;
    int totals[3] = {};
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        if (!valid_remote_entry(e.component, e.at, e.slots.size(), n, i == 0, totals))
            return out;
    }
    const int num_slots = totals[0] + totals[1] + totals[2];
    RemoteOpReqHeader* h = nullptr;
    auto req = build_req(REMOTE_OP_COMMIT, hashes, n, next_req_id_ + 1,
                         ComponentType::FULL, h);
    const size_t desc_off = req.size();
    req.resize(desc_off + entries.size() * sizeof(RemoteCommitEntry) +
               (size_t)num_slots * sizeof(int32_t));
    h = reinterpret_cast<RemoteOpReqHeader*>(req.data());
    h->u.commit.num_entries = (int32_t)entries.size();
    h->u.commit.num_slots = num_slots;
    auto* desc = reinterpret_cast<RemoteCommitEntry*>(req.data() + desc_off);
    auto* slots = reinterpret_cast<int32_t*>(desc + entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        desc[i] = {(uint8_t)e.component, 0, (uint16_t)e.slots.size(), e.at};
        std::memcpy(slots, e.slots.data(), e.slots.size() * sizeof(int32_t));
        slots += e.slots.size();
    }
    ++next_req_id_;
    std::vector<char> reply;
    out.status = request(req.data(), req.size(), reply, 1);
    if (out.status != RemoteStatus::OK) return out;
    out.status = RemoteStatus::BAD_REPLY;
    if (reply.size() < sizeof(RemoteOpReplyHeader)) return out;
    const auto* r = reinterpret_cast<const RemoteOpReplyHeader*>(reply.data());
    if (r->op != REMOTE_OP_COMMIT) return out;
    if (r->status != (uint16_t)RemoteStatus::OK) {
        out.status = static_cast<RemoteStatus>(r->status);
        return out;
    }
    if (r->u.commit.num_entries != (int32_t)entries.size() ||
        r->u.commit.hit_length < 0 || r->u.commit.hit_length > n ||
        reply.size() != sizeof(*r) + entries.size() * sizeof(uint32_t)) return out;
    const auto* errors = reinterpret_cast<const uint32_t*>(r + 1);
    for (size_t i = 0; i < entries.size(); ++i)
        out.errors.push_back(static_cast<InsertError>(errors[i]));
    out.remote_hit = r->u.commit.hit_length;
    out.status = RemoteStatus::OK;
    return out;
}

PrepareReply RemotePeer::prepare(const uint64_t* hashes, int n, int full_hit,
                                 const std::vector<PrepareEntry>& entries) {
    auto g = guard();
    PrepareReply out;
    out.status = RemoteStatus::BAD_REQUEST;
    if (n <= 0 || n > MAX_REMOTE_OP_BLOCKS || !hashes || full_hit < 0 ||
        full_hit > n) return out;
    int totals[3] = {};
    for (const auto& e : entries) {
        if (!valid_remote_entry(e.component, e.at, e.num_slots, full_hit, false, totals))
            return out;
    }
    RemoteOpReqHeader* h = nullptr;
    auto req = build_req(REMOTE_OP_PREPARE, hashes, n, next_req_id_ + 1,
                         ComponentType::FULL, h);
    const size_t desc_off = req.size();
    req.resize(desc_off + entries.size() * sizeof(RemoteCommitEntry));
    h = reinterpret_cast<RemoteOpReqHeader*>(req.data());
    h->u.prepare.full_hit = full_hit;
    h->u.prepare.num_entries = (int32_t)entries.size();
    auto* desc = reinterpret_cast<RemoteCommitEntry*>(req.data() + desc_off);
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        desc[i] = {(uint8_t)e.component, 0, (uint16_t)e.num_slots, e.at};
    }
    ++next_req_id_;
    std::vector<char> reply;
    out.status = request(req.data(), req.size(), reply, 1);
    if (out.status != RemoteStatus::OK) return out;
    out.status = RemoteStatus::BAD_REPLY;
    if (reply.size() < sizeof(RemoteOpReplyHeader)) return out;
    const auto* r = reinterpret_cast<const RemoteOpReplyHeader*>(reply.data());
    if (r->op != REMOTE_OP_PREPARE) return out;
    if (r->status != (uint16_t)RemoteStatus::OK) {
        out.status = static_cast<RemoteStatus>(r->status);
        return out;
    }
    const auto& p = r->u.prepare;
    if (p.hit_length < 0 || p.hit_length > n || p.num_full < 0 ||
        p.num_full > std::max(0, full_hit - p.hit_length) ||
        p.num_swa < 0 || p.num_mamba < 0) return out;
    int expected[3] = {};
    if (p.num_full > 0) {
        const int end = p.hit_length + p.num_full;
        for (const auto& e : entries)
            expected[(int)e.component] += e.component == ComponentType::SWA
                ? std::max(0, std::min(e.at, end) - (e.at - e.num_slots))
                : (e.at <= end ? 1 : 0);
    }
    if ((p.num_swa != 0 && p.num_swa != expected[(int)ComponentType::SWA]) ||
        (p.num_mamba != 0 && p.num_mamba != expected[(int)ComponentType::MAMBA]) ||
        reply.size() != sizeof(*r) + (size_t)(p.num_full + p.num_swa + p.num_mamba) *
                                    sizeof(int32_t)) return out;
    const auto* slots = reinterpret_cast<const int32_t*>(r + 1);
    out.full_slots.assign(slots, slots + p.num_full);
    slots += p.num_full;
    out.swa_slots.assign(slots, slots + p.num_swa);
    slots += p.num_swa;
    out.mamba_slots.assign(slots, slots + p.num_mamba);
    out.remote_hit = p.hit_length;
    out.status = RemoteStatus::OK;
    return out;
}

RemoteStatus RemotePeer::release(const uint64_t* hashes, int n,
                             int start, int length, const int32_t* slots,
                             ComponentType component) {
    return send_ack_op(slots ? REMOTE_OP_INSERT_RELEASE
                             : REMOTE_OP_QUERY_RELEASE,
                       hashes, n, start, length, slots, /*flags=*/0, component);
}

QueryReply RemotePeer::query(const uint64_t* hashes, int n, bool lock,
                             bool update_meta) {
    auto g = guard();
    QueryReply out;
    RemoteOpReqHeader* h = nullptr;
    auto req = build_req(REMOTE_OP_QUERY, hashes, n, next_req_id_ + 1, ComponentType::FULL, h);
    if (!h) { out.status = RemoteStatus::BAD_REQUEST; return out; }
    ++next_req_id_;
    h->u.query.local_only  = 1;
    h->u.query.lock        = lock ? 1 : 0;
    h->u.query.update_meta = update_meta ? 1 : 0;

    std::vector<char> reply;
    RemoteStatus st = request(req.data(), req.size(), reply);
    if (st != RemoteStatus::OK) { out.status = st; return out; }

    auto* r = reinterpret_cast<RemoteOpReplyHeader*>(reply.data());
    out.status = static_cast<RemoteStatus>(r->status);
    if (out.status != RemoteStatus::OK) return out;   // BAD_REQUEST / UNKNOWN_OP
    out.hit_length = r->u.query.hit_length;
    if (!read_slots(reply, r->u.query.num_slots, out.slots))
        out.status = RemoteStatus::BAD_REPLY;
    return out;
}

}  // namespace radixshmem

#endif  // !RADIXSHMEM_NO_RDMA
