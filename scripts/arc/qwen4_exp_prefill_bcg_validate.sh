#!/bin/bash
# GPU runbook for ARC_QWEN4_EXP_PREFILL_BCG (ARC.md, scripts/arc/qwen4_exp_prefill_bcg_check.py).
# One step per invocation, in order; read each step's output before the next:
#
#   bash scripts/arc/qwen4_exp_prefill_bcg_validate.sh s1          # TP1 NVFP4: B1 (GPU2) + A1 (GPU3) in parallel
#   bash scripts/arc/qwen4_exp_prefill_bcg_validate.sh s2          # TP1 NVFP4: A2 (GPU2) + B2 = B1 + task-2 flags (GPU3)
#   bash scripts/arc/qwen4_exp_prefill_bcg_validate.sh s3          # compare only (no GPU)
#   bash scripts/arc/qwen4_exp_prefill_bcg_validate.sh s4          # TP2 FP8 SPEC=none: A1 -> B1 -> A2, then compare
#   bash scripts/arc/qwen4_exp_prefill_bcg_validate.sh s4d         # TP2 FP8 SPEC=dflash K4, --random-seed 1234: A -> B, then compare
#   bash scripts/arc/qwen4_exp_prefill_bcg_validate.sh s4dc        # s4d's two arms again (A2, B2): seeded reproducibility, A2 vs B2
#   bash scripts/arc/qwen4_exp_prefill_bcg_validate.sh s4dv        # media (ViT) cost per extend, eager and graph DFlash, twice each
#   bash scripts/arc/qwen4_exp_prefill_bcg_validate.sh s5 off|on   # real harness, 16 groups x 15 min (run_arms.sh)
#
# RUN=<tag> names a fresh set of arm directories ($DAY-bcg-<tag>-<arm>) for a rerun of the steps.
# s3/s4/s4d also run `isolation` on each arm: gated (--expect-bitwise) on the graph arms. Collects from
# before the iso_*_pad2/fill2 cases (RUN=r3) no longer pass `isolation`; their s3 output is in the run dirs.
#
# Rules this script enforces: GPUs 2,3 only; ports 30620-30629; servers only through arc-forge's serve_arm.sh
# (systemd user unit b200perf-p<PORT>), stopped only with `systemctl --user stop` of that unit; every ARM_DIR new
# (serve_arm.sh refuses a used one). It never signals a process. serve_qwen38fn_0520.sh itself refuses to start
# on a GPU that has someone else's process.
set -euo pipefail
ARC=/data/home/ycc/work/arc-forge/.claude/worktrees/bridge-cse_015gLJpSy4kASpr5VtocQrnY
FORK=$HOME/work/sglang-arc-2-b200perf-ext
R=$HOME/work/runs/qwen38fn_b200_perf
PY=$HOME/work/dsv4/venv-0520/bin/python
CHECK=$FORK/scripts/arc/qwen4_exp_prefill_bcg_check.py
DAY=${DAY:-$(date -u +%Y%m%d)}
BASE=$R/$DAY-bcg-${RUN:+$RUN-}
# memory qsa-fast-topk-nondeterminism: without all three a restart alone moves logprobs
DET="ARC_QSA_TOPK_EXACT=1 SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 ARC_HC_MIX_DETERMINISTIC=1"
# the production list (G1, 2026-10-08): up to chunked_prefill_size 8192
BUCKETS="16 32 64 128 256 384 512 640 768 896 1024 1280 1536 1792 2048 2560 3072 4096 5120 6144 8192"
ON="ARC_QWEN4_EXP_PREFILL_BCG=1"
TASK2="ARC_MAMBA_TRACK_HOST=1 ARC_MROPE_EXTEND_PINNED=1"
ROUTED="--enable-return-routed-experts"

STARTED=()
cleanup() {
    for unit in "${STARTED[@]}"; do
        if systemctl --user is-active --quiet "$unit"; then
            echo "· cleanup: systemctl --user stop $unit" >&2
            systemctl --user stop "$unit"
        fi
    done
}
trap cleanup EXIT

json_field() { python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
gpu_report() {
    echo "· $(date -u +%T) nvidia-smi (all GPUs; only 2,3 may change because of this script):"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/    /'
}

# start <name> <port> <gpus> <tp> <quant> <spec> <arm_env> <server_args>
start() {
    local name=$1 port=$2 gpus=$3 tp=$4 quant=$5 spec=$6 envs=$7 sargs=$8
    case $port in 3062[0-9]) ;; *) echo "✗ port $port is outside 30620-30629" >&2; exit 1 ;; esac
    for g in ${gpus//,/ }; do
        case $g in 2 | 3) ;; *) echo "✗ GPU $g was not handed to this run" >&2; exit 1 ;; esac
    done
    echo "== start $name: port $port GPU $gpus TP$tp $quant SPEC=$spec"
    echo "   ARM_ENV=[$envs]"
    echo "   SERVER_ARGS=[$sargs]"
    gpu_report
    (cd "$ARC" && env ARM_DIR="$BASE$name" PORT="$port" GPUS="$gpus" TP="$tp" QUANT="$quant" SPEC="$spec" \
        LEDGER=0 FORK="$FORK" ARM_ENV="$envs" SERVER_ARGS="$sargs" bash work/qwen38fn_b200_perf/serve_arm.sh)
    STARTED+=("b200perf-p$port")
}
ready() { (cd "$ARC" && WAIT_SECONDS=${WAIT_SECONDS:-2400} bash work/qwen38fn_b200_perf/serve_arm.sh wait "$BASE$1"); }
collect() {
    local dir=$BASE$1
    "$PY" "$CHECK" collect --url "http://127.0.0.1:$(json_field "$dir/arm.json" port)" --out "$dir/collect.json" \
        --routed-experts
    "$PY" "$CHECK" log "$dir/serve/server.log" | tee "$dir/log_summary.txt"
    echo "· error lines in server.log: $(awk '/Traceback|CUDA error|illegal memory|RuntimeError/{n++} END{print n+0}' \
        "$dir/serve/server.log")"
}
stop() {
    local unit
    unit=$(json_field "$BASE$1/arm.json" unit)
    echo "== stop $1 ($unit)"
    systemctl --user stop "$unit"
    for _ in $(seq 60); do
        systemctl --user is-active --quiet "$unit" || break
        sleep 2
    done
    sleep 10
    gpu_report
}
compare() {
    echo "== compare $1 vs $2"
    "$PY" "$CHECK" compare "$BASE$1/collect.json" "$BASE$2/collect.json" | tee "$BASE$2/compare-vs-$1.txt"
}
# vit <arm>: media embedding (ViT + merge) vs language model GPU time per extend (torch profiler)
vit() {
    echo "== vitcost $1"
    "$PY" "$CHECK" vitcost --url "http://127.0.0.1:$(json_field "$BASE$1/arm.json" port)" --out-dir "$BASE$1/vitcost" \
        | tee "$BASE$1/vitcost.txt"
}
# vittel <arm>: the same media cost without a profiler, from the extend telemetry (spec arms only)
TELEMETRY="ARC_EXTEND_TELEMETRY=1 ARC_EXTEND_TELEMETRY_INTERVAL_S=10"
vittel() {
    echo "== vittel $1 ${2:-}"
    "$PY" "$CHECK" vittel --url "http://127.0.0.1:$(json_field "$BASE$1/arm.json" port)" --log "$BASE$1/serve/server.log" \
        --interval 10 | tee "$BASE$1/vittel${2:-}.txt"
}
# iso <arm> [--expect-bitwise [--gate-single]]: padding isolation within one arm (gate it on graph
# arms; --gate-single only where FlashInfer's GDN prefill never takes its CP kernel, i.e. TP1)
iso() {
    echo "== isolation $1 ${*:2}"
    "$PY" "$CHECK" isolation "$BASE$1/collect.json" "${@:2}" | tee "$BASE$1/isolation.txt"
}

case ${1:?step} in
    s1)
        start tp1-B1 30620 2 1 nvfp4 none "$DET $ON" "$ROUTED --cuda-graph-bs-prefill $BUCKETS"
        start tp1-A1 30621 3 1 nvfp4 none "$DET" "$ROUTED"
        ready tp1-B1
        ready tp1-A1
        collect tp1-B1
        collect tp1-A1
        stop tp1-B1
        stop tp1-A1
        ;;
    s2)
        # A2 on the other GPU than A1: the control then covers a restart and a device change.
        start tp1-A2 30622 2 1 nvfp4 none "$DET" "$ROUTED"
        start tp1-B2 30623 3 1 nvfp4 none "$DET $ON $TASK2" "$ROUTED --cuda-graph-bs-prefill $BUCKETS"
        ready tp1-A2
        ready tp1-B2
        collect tp1-A2
        collect tp1-B2
        stop tp1-A2
        stop tp1-B2
        ;;
    s3)
        rc=0
        compare tp1-A1 tp1-A2 || { rc=1; echo "✗ CONTROL FAILED: A vs A is not bitwise; A vs B below is no gate"; }
        compare tp1-A1 tp1-B1 || rc=1
        compare tp1-A1 tp1-B2 || rc=1
        iso tp1-A1
        iso tp1-B1 --expect-bitwise --gate-single || rc=1
        iso tp1-B2 --expect-bitwise --gate-single || rc=1
        exit $rc
        ;;
    s4)
        for name in tp2-A1:30624: tp2-B1:30625:on tp2-A2:30626:; do
            IFS=: read -r n port on <<< "$name"
            if [ "$on" = on ]; then
                start "$n" "$port" 2,3 2 fp8 none "$DET $ON" "$ROUTED --cuda-graph-bs-prefill $BUCKETS"
            else
                start "$n" "$port" 2,3 2 fp8 none "$DET" "$ROUTED"
            fi
            ready "$n"
            collect "$n"
            if [ "$n" = tp2-A1 ]; then
                vit "$n" || echo "✗ vitcost failed (above); the gate run continues"
            fi
            stop "$n"
        done
        rc=0
        compare tp2-A1 tp2-A2 || { rc=1; echo "✗ CONTROL FAILED: A vs A is not bitwise; A vs B below is no gate"; }
        compare tp2-A1 tp2-B1 || rc=1
        iso tp2-A1
        iso tp2-B1 --expect-bitwise || rc=1
        exit $rc
        ;;
    s4d)
        # A fixed --random-seed on both arms: DFlash's verify draws from the server RNG, so unseeded arms
        # do not reproduce the decode even eager vs eager (ARC.md, 2026-10-08 r5 dA/dA2)
        start tp2-dA 30627 2,3 2 fp8 dflash "$DET $TELEMETRY" "$ROUTED --random-seed 1234"
        ready tp2-dA
        collect tp2-dA
        vittel tp2-dA || echo "✗ vittel failed (above); the gate run continues"
        stop tp2-dA
        start tp2-dB 30628 2,3 2 fp8 dflash "$DET $TELEMETRY $ON" "$ROUTED --random-seed 1234 --cuda-graph-bs-prefill $BUCKETS"
        ready tp2-dB
        collect tp2-dB
        vittel tp2-dB || echo "✗ vittel failed (above); the gate run continues"
        stop tp2-dB
        rc=0
        compare tp2-dA tp2-dB || rc=1
        iso tp2-dA
        iso tp2-dB --expect-bitwise || rc=1
        exit $rc
        ;;
    s4dc)
        # s4d's seeded arms once more: does seeded DFlash reproduce its decode eager vs eager / graph vs graph?
        start tp2-dA2 30629 2,3 2 fp8 dflash "$DET $TELEMETRY" "$ROUTED --random-seed 1234"
        ready tp2-dA2
        collect tp2-dA2
        vittel tp2-dA2 || echo "✗ vittel failed (above); the gate run continues"
        stop tp2-dA2
        start tp2-dB2 30620 2,3 2 fp8 dflash "$DET $TELEMETRY $ON" "$ROUTED --random-seed 1234 --cuda-graph-bs-prefill $BUCKETS"
        ready tp2-dB2
        collect tp2-dB2
        vittel tp2-dB2 || echo "✗ vittel failed (above); the gate run continues"
        stop tp2-dB2
        rc=0
        compare tp2-dA tp2-dA2 || { rc=1; echo "✗ seeded eager DFlash does not reproduce"; }
        compare tp2-dB tp2-dB2 || { rc=1; echo "✗ seeded graph DFlash does not reproduce"; }
        compare tp2-dA2 tp2-dB2 || rc=1
        iso tp2-dB2 --expect-bitwise || rc=1
        exit $rc
        ;;
    s4dv)
        # the media cost alone, twice per arm (no collect): eager and graph DFlash
        start tp2-dAv 30621 2,3 2 fp8 dflash "$DET $TELEMETRY" "$ROUTED --random-seed 1234"
        ready tp2-dAv
        vittel tp2-dAv 1
        vittel tp2-dAv 2
        stop tp2-dAv
        start tp2-dBv 30622 2,3 2 fp8 dflash "$DET $TELEMETRY $ON" "$ROUTED --random-seed 1234 --cuda-graph-bs-prefill $BUCKETS"
        ready tp2-dBv
        vittel tp2-dBv 1
        vittel tp2-dBv 2
        stop tp2-dBv
        ;;
    s5)
        mode=${2:?s5 off|on}
        envs="ARC_EXTEND_TELEMETRY=1"
        sargs="--stream-interval 8 --scheduler-recv-interval 8"
        case $mode in
            off) ;;
            on) envs="$envs $ON" sargs="$sargs --cuda-graph-bs-prefill $BUCKETS" ;;
            *) echo "✗ s5 off|on" >&2; exit 1 ;;
        esac
        gpu_report
        # run_arms.sh: start -> wait -> run_load.sh -> py-spy at 390 s -> stop (systemctl) -> delete the ledger
        (cd "$ARC" && NAME=bcg-s5-$mode NGROUPS=16 MINUTES=15 \
            ARMS="PORT=30620 GPUS=2,3 TP=2 QUANT=fp8 SPEC=dflash DFLASH_K=4 FORK=$FORK" \
            COMMON_ARM_ENV="$envs" COMMON_SERVER_ARGS="$sargs" bash work/qwen38fn_b200_perf/run_arms.sh)
        gpu_report
        arm=$(ls -d "$R"/*-bcg-s5-"$mode"-a0 | tail -1)
        "$PY" "$CHECK" log "$arm/serve/server.log" | tee "$arm/log_summary.txt"
        python3 "$ARC/work/qwen38fn_b200_perf/server_log_stats.py" "$arm/serve/server.log" | tee "$arm/log_stats.txt"
        grep "ARC extend telemetry" "$arm/serve/server.log" | tail -3
        ;;
    *)
        echo "✗ unknown step $1" >&2
        exit 1
        ;;
esac
