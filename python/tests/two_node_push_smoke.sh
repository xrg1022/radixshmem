#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Two-machine push_async smoke: A (this host) and B share one etcd, each runs a
# RadixServer with the Mooncake data plane on mlx5_bond_0, and the driver pushes
# a byte-patterned prefix across, verifies it on the far side, migrates one back
# (evict_local) and pulls one. Roles live in two_node_push_smoke.py.
#
#   two_node_push_smoke.sh sync            # rsync repo + mooncake-te to B
#   two_node_push_smoke.sh build           # build B's container (A is assumed built)
#   two_node_push_smoke.sh run [swa|mamba|none] [zmq|dc]
#   two_node_push_smoke.sh stop            # kill leftovers on both hosts
set -euo pipefail

A_IP=${A_IP:-10.6.131.12}
B_IP=${B_IP:-10.6.131.11}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519_ige}
REPO=${REPO:-/raid/xrg/radixshmem}          # same path on both hosts
TE=${TE:-/raid/xrg/mooncake-te}
CREPO=/workspace/RadixShmem                  # the repo inside radixbench_ige
CTE=/workspace/mooncake-te
# `env` so the assignments survive `nohup $PY ...` (nohup takes its first word as the command).
PY="env LD_LIBRARY_PATH=$CREPO/build/lib PYTHONPATH=$CREPO/python python3 $CREPO/python/tests/two_node_push_smoke.py"
LOG=${LOG:-/tmp/two_node_smoke}

B_SSH=(ssh -i "$SSH_KEY" -o ConnectTimeout=10 "root@$B_IP")
docker_a() { docker exec radixbench_ige bash -lc "$1"; }
docker_b() { "${B_SSH[@]}" "docker exec radixbench_ige bash -lc '$1'"; }

sync() {
    rsync -a --delete --exclude build --exclude build-local --exclude build.container-stale --exclude .git \
          -e "ssh -i $SSH_KEY" "$REPO/" "root@$B_IP:$REPO/"
    rsync -a -e "ssh -i $SSH_KEY" "$TE/" "root@$B_IP:$TE/"
    echo SYNC_OK
}

build() {
    "${B_SSH[@]}" "docker start radixbench_ige >/dev/null; docker ps --filter name=radixbench_ige --format '{{.Names}} {{.Status}}'"
    docker_b "cd $CREPO && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -UMOONCAKE_INCLUDE_DIR -UMOONCAKE_LIBRARY -UMOONCAKE_BASE_LIBRARY -UMOONCAKE_COMMON_LIBRARY -UMOONCAKE_ASIO_LIBRARY \
        -DMOONCAKE_ROOT=$CTE -DMOONCAKE_BUILD_DIR=$CTE/build-te-c > /tmp/tn_cfg.log 2>&1 \
        && grep -E \"Mooncake (found|not)\" /tmp/tn_cfg.log \
        && cmake --build build -j\$(nproc) --target radixshmem radix_server radixshmem_data_py > /tmp/tn_build.log 2>&1 && echo B_BUILD_OK"
    docker_b "cd $CREPO/python && pip install -e . --no-build-isolation > /tmp/tn_pyext.log 2>&1 && echo B_PYEXT_OK"
    docker_b "cd $CREPO && $PY probe"
}

wait_ready() {   # $1 = a|b, $2 = log path inside the container
    local who=$1 log=$2 i
    for i in $(seq 1 180); do
        if [ "$who" = a ]; then docker_a "grep -q '^READY' $log 2>/dev/null" && return 0
        else docker_b "grep -q ^READY $log 2>/dev/null" && return 0; fi
        sleep 1
    done
    echo "server $who never became ready; log tail:"; [ "$who" = a ] && docker_a "tail -20 $log" || docker_b "tail -20 $log"
    return 1
}

stop() {
    docker_a "pkill -f '[t]wo_node_push_smoke.py' || true; pkill -f '[e]tcd --name two_node' || true; rm -f /dev/shm/tn_*"
    docker_b "pkill -f \"[t]wo_node_push_smoke.py\" || true; rm -f /dev/shm/tn_*" || true
}

run() {
    local aux=${1:-swa} rot=${2:-zmq} cid="tn$(date +%s | tail -c 6)"
    local etcd_port=${ETCD_PORT:-23790} n=${N:-2048}
    stop
    mkdir -p "$LOG"
    echo "== etcd on $A_IP:$etcd_port =="
    docker_a "rm -rf /tmp/tn_etcd && nohup etcd --name two_node --data-dir /tmp/tn_etcd \
        --listen-client-urls http://0.0.0.0:$etcd_port --advertise-client-urls http://$A_IP:$etcd_port \
        --listen-peer-urls http://127.0.0.1:23791 --initial-advertise-peer-urls http://127.0.0.1:23791 \
        --initial-cluster two_node=http://127.0.0.1:23791 > /tmp/tn_etcd.log 2>&1 &"
    sleep 2
    local common="--registry etcd://$A_IP:$etcd_port --cluster-id $cid --aux $aux --remote-op $rot"
    echo "== servers =="
    docker_a "cd $CREPO && nohup $PY server --node-ip $A_IP $common > /tmp/tn_server.log 2>&1 &"
    docker_b "cd $CREPO && nohup $PY server --node-ip $B_IP $common > /tmp/tn_server.log 2>&1 &"
    wait_ready a /tmp/tn_server.log; wait_ready b /tmp/tn_server.log
    docker_a "grep ^READY /tmp/tn_server.log"; docker_b "grep ^READY /tmp/tn_server.log"

    local ext=$((n + 512))
    echo "== A pushes seed 1 to B (explicit peer), B verifies =="
    docker_a "cd $CREPO && $PY push $common --n $n --seed 1"
    docker_b "cd $CREPO && $PY verify $common --n $n --seed 1"
    echo "== A extends seed 1 by 512 and pushes with rank=None (RHT must pick B) =="
    docker_a "cd $CREPO && $PY push $common --n $ext --seed 1 --target auto"
    docker_b "cd $CREPO && $PY verify $common --n $ext --seed 1"
    echo "== B pushes seed 2 to A with evict_local, A verifies, B confirms it is gone =="
    docker_b "cd $CREPO && $PY push $common --n $n --seed 2 --evict"
    docker_a "cd $CREPO && $PY verify $common --n $n --seed 2"
    docker_b "cd $CREPO && $PY verify $common --n $n --seed 2 --expect-hit 0"
    echo "== B pulls seed 2 back (cross-host READ) =="
    docker_b "cd $CREPO && $PY pull $common --n $n --seed 2"
    echo "== health =="
    docker_a "cd $CREPO && $PY health $common"; docker_b "cd $CREPO && $PY health $common"
    docker_a "pkill -TERM -f '[t]wo_node_push_smoke.py server' || true"
    docker_b "pkill -TERM -f \"[t]wo_node_push_smoke.py server\" || true"
    sleep 2
    docker_a "tail -3 /tmp/tn_server.log"; docker_b "tail -3 /tmp/tn_server.log"
    docker_a "pkill -f '[e]tcd --name two_node' || true"
    echo RUN_OK
}

case "${1:-}" in
    sync) sync ;;
    build) build ;;
    run) shift; run "$@" ;;
    stop) stop ;;
    *) sed -n 2,11p "$0"; exit 2 ;;
esac
