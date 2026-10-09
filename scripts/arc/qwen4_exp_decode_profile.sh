#!/bin/bash
# GPU runbook: the DFlash K4 decode cycle of TP2 FP8 under the real RL harness load, broken down to kernels
# (scripts/arc/qwen4_exp_extend_profile.py capture / decode).
#
#   ALLOWED=2,3 GPUS=2,3 RUN=r1 bash scripts/arc/qwen4_exp_decode_profile.sh d1
#
# One arm with the H1 B flags (BCG, ARC_MAMBA_TRACK_HOST, ARC_MROPE_EXTEND_PINNED, extend telemetry, delayer
# 0.06, stream/recv interval 8) and the RL ledger (LEDGER=1: session replay, ledger all/routing/logprobs,
# routed experts), plus ARC_SPEC_PHASE_TIMING (CUDA events per decode phase, no profiler). arc-forge's
# run_arms.sh starts it, runs run_load.sh (NGROUPS groups x 8 sessions, MINUTES), waits, stops the arm and
# deletes its ledger; this script takes torch-profiler windows of CAPTURE_S seconds at CAPTURE_AT seconds
# into the load (steady state), then analyzes them (CPU).
#
# Rules: GPUs only from ALLOWED; port 30620-30629; servers only through serve_arm.sh (systemd unit
# b200perf-p<PORT>), stopped only with `systemctl --user stop`; the load unit b200perf-load-<NAME> likewise.
set -euo pipefail
ARC=/data/home/ycc/work/arc-forge/.claude/worktrees/bridge-cse_015gLJpSy4kASpr5VtocQrnY
FORK=$HOME/work/sglang-arc-2-b200perf-ext
R=$HOME/work/runs/qwen38fn_b200_perf
PY=$HOME/work/dsv4/venv-0520/bin/python
TOOL=$FORK/scripts/arc/qwen4_exp_extend_profile.py
NAME=dprof-${RUN:?RUN=<tag>}
DAY=$(date -u +%Y%m%d)
ARM=$R/$DAY-$NAME-a0
OUT=$R/loads/$NAME
PORT=${PORT:-30620}
NGROUPS=${NGROUPS:-16}
MINUTES=${MINUTES:-12}
CAPTURE_AT=${CAPTURE_AT:-330 510}
CAPTURE_S=${CAPTURE_S:-2}
BUCKETS="16 32 64 128 256 384 512 640 768 896 1024 1280 1536 1792 2048 2560 3072 4096 5120 6144 8192"

case $PORT in 3062[0-9]) ;; *) echo "✗ port $PORT is outside 30620-30629" >&2; exit 1 ;; esac
for g in ${GPUS//,/ }; do
    case ",${ALLOWED:?ALLOWED=<the GPUs handed over>}," in *",$g,"*) ;; *) echo "✗ GPU $g is not in ALLOWED=$ALLOWED" >&2; exit 1 ;; esac
done
[ ! -e "$ARM" ] || { echo "✗ $ARM exists (RUN tag used)" >&2; exit 1; }
[ ! -e "$OUT" ] || { echo "✗ $OUT exists (RUN tag used)" >&2; exit 1; }

gpu_report() {
    echo "· $(date -u +%T) nvidia-smi (only $ALLOWED may change because of this script):"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/    /'
}

case ${1:?step} in
    d1)
        gpu_report
        # run_arms.sh: arm -> wait -> run_load.sh -> (PROF_AT=0: no py-spy) -> load ends -> report -> stop arm, rm ledger
        (cd "$ARC" && NAME=$NAME NGROUPS=$NGROUPS MINUTES=$MINUTES PROF_AT=0 \
            ARMS="PORT=$PORT GPUS=$GPUS TP=2 QUANT=fp8 SPEC=dflash DFLASH_K=4 FORK=$FORK" \
            COMMON_ARM_ENV="ARC_QWEN4_EXP_PREFILL_BCG=1 ARC_MAMBA_TRACK_HOST=1 ARC_MROPE_EXTEND_PINNED=1 ARC_EXTEND_TELEMETRY=1 ARC_SPEC_PHASE_TIMING=$ARM/phase_timing.json" \
            COMMON_SERVER_ARGS="--stream-interval 8 --scheduler-recv-interval 8 --enable-prefill-delayer --prefill-delayer-max-delay-ms 1000 --prefill-delayer-max-delay-passes 20 --prefill-delayer-queue-min-ratio 0.06 --cuda-graph-bs-prefill $BUCKETS" \
            bash work/qwen38fn_b200_perf/run_arms.sh) > "$R/$DAY-$NAME.arms.out" 2>&1 &
        arms=$!
        until [ -f "$OUT/meta.json" ]; do
            kill -0 $arms 2>/dev/null || { echo "✗ run_arms.sh ended before the load started:"; tail -20 "$R/$DAY-$NAME.arms.out"; exit 1; }
            sleep 5
        done
        t0=$(python3 -c 'import json,sys;print(int(json.load(open(sys.argv[1]))["started"]))' "$OUT/meta.json")
        echo "· load started $(date -u -d @"$t0" +%T); captures at +${CAPTURE_AT// /s, +}s, ${CAPTURE_S}s each"
        i=0
        for at in $CAPTURE_AT; do
            while [ $(($(date +%s) - t0)) -lt "$at" ]; do sleep 2; done
            i=$((i + 1))
            echo "== capture d$i at +$(($(date +%s) - t0))s"
            "$PY" "$TOOL" capture --url "http://127.0.0.1:$PORT" --log "$ARM/serve/server.log" \
                --out-dir "$ARM/dprof" --prefix "d$i" --seconds "$CAPTURE_S"
        done
        wait $arms || { echo "✗ run_arms.sh failed:"; tail -20 "$R/$DAY-$NAME.arms.out"; exit 1; }
        tail -40 "$R/$DAY-$NAME.arms.out"
        [ ! -d "$ARM/ledger" ] || { echo "✗ $ARM/ledger is still there" >&2; exit 1; }
        gpu_report
        ;;
    *)
        echo "✗ unknown step $1" >&2
        exit 1
        ;;
esac
