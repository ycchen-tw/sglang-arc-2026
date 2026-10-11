#!/bin/bash
# GPU runbook: kernel breakdown of a large Qwen4-Exp extend batch, TP2 FP8 DFlash K4 (the RL deployment
# shape), eager prefill vs the breakable prefill CUDA graph (scripts/arc/qwen4_exp_extend_profile.py).
#
#   ALLOWED=0,1,2,3 GPUS_A=0,1 GPUS_B=2,3 RUN=r1 bash scripts/arc/qwen4_exp_extend_profile.sh x1
#       three arms: eA (eager prefill), eB (BCG), eC (BCG + ARC_DFLASH_EXACT_PREFILL=1
#       ARC_DFLASH_REFRESH_ASYNC=1: the drafter refresh without its host reads). eA then eC on GPUS_A,
#       eB on GPUS_B at the same time (GPUS_A = GPUS_B: all three one after the other). Per arm: start,
#       drive (telemetry + torch profiler per shape), stop. ARMS=eA,eB limits the arms.
#   RUN=r1 bash scripts/arc/qwen4_exp_extend_profile.sh xa      # analysis only (CPU), rerunnable
#
# All arms: the H1 B arm's deployment flags (ARC_MAMBA_TRACK_HOST=1 ARC_MROPE_EXTEND_PINNED=1, the prefill
# delayer at 0.06, stream/recv interval 8, routed experts) with LEDGER=0, and no determinism switches (the
# production QSA top-k and MoE finalize kernels); eB/eC add ARC_QWEN4_EXP_PREFILL_BCG=1 and the production
# bucket list. All add ARC_EXTEND_TELEMETRY=1 (10 s windows) and SGLANG_ENABLE_COLOCATED_BATCH_GEN=1 (a
# batched /generate of media prompts reaches the scheduler as one group; the cost is one CPU-group
# barrier poll per scheduler pass, outside the extend's spans). The driver's requests are not session
# turns, so under EXACT_PREFILL every one bootstraps its window from its own chunk (the drafter never saw
# the cached prefix): eC refreshes the same tokens as eB, and the difference is the host reads.
#
# Rules this script enforces: only the GPUs in ALLOWED (given by the coordinator); ports 30620-30629;
# servers only through arc-forge's serve_arm.sh (systemd user unit b200perf-p<PORT>), stopped only with
# `systemctl --user stop` of that unit; every ARM_DIR new. It never signals a process.
set -euo pipefail
ARC=/data/home/ycc/work/arc-forge/.claude/worktrees/bridge-cse_015gLJpSy4kASpr5VtocQrnY
FORK=$HOME/work/sglang-arc-2-b200perf-ext
R=$HOME/work/runs/qwen38fn_b200_perf
PY=$HOME/work/dsv4/venv-0520/bin/python
TOOL=$FORK/scripts/arc/qwen4_exp_extend_profile.py
CHECK=$FORK/scripts/arc/qwen4_exp_prefill_bcg_check.py
DAY=${DAY:-$(date -u +%Y%m%d)}
BASE=$R/$DAY-xprof-${RUN:?RUN=<tag>}-
SHAPES=${SHAPES:-p10k,p10k_img,cold,p10k_352,p10k_512,p10k_1408,pmix,p45k}
ARMS=${ARMS:-eA,eB,eC}
DFLASH_ASYNC="ARC_DFLASH_EXACT_PREFILL=1 ARC_DFLASH_REFRESH_ASYNC=1"
BUCKETS="16 32 64 128 256 384 512 640 768 896 1024 1280 1536 1792 2048 2560 3072 4096 5120 6144 8192"
COMMON_ENV="ARC_MAMBA_TRACK_HOST=1 ARC_MROPE_EXTEND_PINNED=1 ARC_EXTEND_TELEMETRY=1 ARC_EXTEND_TELEMETRY_INTERVAL_S=10 SGLANG_ENABLE_COLOCATED_BATCH_GEN=1"
COMMON_ARGS="--enable-return-routed-experts --stream-interval 8 --scheduler-recv-interval 8 --enable-prefill-delayer --prefill-delayer-max-delay-ms 1000 --prefill-delayer-max-delay-passes 20 --prefill-delayer-queue-min-ratio 0.06"

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
    echo "· $(date -u +%T) nvidia-smi (all GPUs; only ${ALLOWED:-?} may change because of this script):"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/    /'
}

# start <name> <port> <gpus> <arm_env> <server_args>
start() {
    local name=$1 port=$2 gpus=$3 envs=$4 sargs=$5
    case $port in 3062[0-9]) ;; *) echo "✗ port $port is outside 30620-30629" >&2; exit 1 ;; esac
    for g in ${gpus//,/ }; do
        case ",${ALLOWED:?ALLOWED=<the GPUs handed over>}," in *",$g,"*) ;; *) echo "✗ GPU $g is not in ALLOWED=$ALLOWED" >&2; exit 1 ;; esac
    done
    echo "== start $name: port $port GPU $gpus TP2 fp8 SPEC=dflash K4"
    echo "   ARM_ENV=[$envs]"
    echo "   SERVER_ARGS=[$sargs]"
    gpu_report
    (cd "$ARC" && env ARM_DIR="$BASE$name" PORT="$port" GPUS="$gpus" TP=2 QUANT=fp8 SPEC=dflash DFLASH_K=4 \
        LEDGER=0 FORK="$FORK" ARM_ENV="$envs" SERVER_ARGS="$sargs" bash work/qwen38fn_b200_perf/serve_arm.sh)
    STARTED+=("b200perf-p$port")
}
ready() { (cd "$ARC" && WAIT_SECONDS=${WAIT_SECONDS:-2400} bash work/qwen38fn_b200_perf/serve_arm.sh wait "$BASE$1"); }
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
# drive <name> <expect-graph true|false>
drive() {
    local dir=$BASE$1
    echo "== drive $1 ($SHAPES)"
    "$PY" "$TOOL" drive --url "http://127.0.0.1:$(json_field "$dir/arm.json" port)" --log "$dir/serve/server.log" \
        --out-dir "$dir/xprof" --shapes "$SHAPES" --expect-graph "$2" | tee "$dir/drive.txt"
    "$PY" "$CHECK" log "$dir/serve/server.log" | tee "$dir/log_summary.txt"
    echo "· error lines in server.log: $(awk '/Traceback|CUDA error|illegal memory|RuntimeError/{n++} END{print n+0}' \
        "$dir/serve/server.log")"
}
want() { case ",$ARMS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
arm() {  # arm <eA|eB|eC> <gpus>; skipped unless in ARMS
    local name=$1 gpus=$2
    want "$name" || { echo "· skip $name (ARMS=$ARMS)"; return 0; }
    case $name in
        eA) start "$name" 30620 "$gpus" "$COMMON_ENV" "$COMMON_ARGS" ;;
        eB) start "$name" 30621 "$gpus" "$COMMON_ENV ARC_QWEN4_EXP_PREFILL_BCG=1" "$COMMON_ARGS --cuda-graph-bs-prefill $BUCKETS" ;;
        eC) start "$name" 30622 "$gpus" "$COMMON_ENV ARC_QWEN4_EXP_PREFILL_BCG=1 $DFLASH_ASYNC" "$COMMON_ARGS --cuda-graph-bs-prefill $BUCKETS" ;;
        *) echo "✗ unknown arm $name" >&2; exit 1 ;;
    esac
    ready "$name"
    if [ "$name" = eC ]; then
        # the opt-ins must have reached the drafter (the worker logs its exact-prefill setup)
        grep -q "DFlash exact prefill" "$BASE$name/serve/server.log" \
            || { echo "✗ eC: no 'DFlash exact prefill' line in server.log; the opt-ins did not take" >&2; exit 1; }
    fi
    drive "$name" "$([ "$name" = eA ] && echo false || echo true)"
    stop "$name"
}
# analyze: eager arm first (its per-shape name map attributes the graph arm's replayed kernels)
analyze() {
    local shape tok seqs
    for shape in ${SHAPES//,/ }; do
        read -r seqs tok < <("$PY" -c "import sys; sys.path.insert(0, '$FORK/scripts/arc'); import qwen4_exp_extend_profile as x; s = x.SHAPES['$shape']; print(len(s), sum(r[1] for r in s))")
        for a in eA eB eC; do
            want "$a" || continue
            [ -d "$BASE$a/xprof/$shape" ] || { echo "✗ $a has no $shape traces" >&2; exit 1; }
            local d=$BASE$a/xprof/$shape
            local t0 t1
            t0=$(ls "$d"/"$shape"-*-TP-0-*.trace.json.gz)
            t1=$(ls "$d"/"$shape"-*-TP-1-*.trace.json.gz)
            local map=()
            [ "$a" = eA ] || map=(--map "$BASE"eA/xprof/$shape/analysis.json)
            "$PY" "$TOOL" analyze "$t0" --peer "$t1" --seqs "$seqs" --tokens "$tok" "${map[@]}" \
                --out "$d/analysis.json" | tee "$d/analysis.txt"
        done
    done
}

table() {
    local dirs=()
    for a in eA eB eC; do want "$a" && dirs+=("$BASE$a"); done
    "$PY" "$TOOL" table "${dirs[@]}" | tee "${BASE}table.txt"
}

case ${1:?step} in
    x1)
        : "${GPUS_A:?GPUS_A=<pair>}" "${GPUS_B:?GPUS_B=<pair>}"
        if [ "$GPUS_A" = "$GPUS_B" ]; then
            arm eA "$GPUS_A"
            arm eB "$GPUS_B"
            arm eC "$GPUS_A"
        else
            for g in ${GPUS_A//,/ }; do
                case ",$GPUS_B," in *",$g,"*) echo "✗ GPUS_A and GPUS_B overlap but differ" >&2; exit 1 ;; esac
            done
            # each background arm stops its own unit on failure (an EXIT trap is not inherited)
            (trap cleanup EXIT; arm eA "$GPUS_A"; arm eC "$GPUS_A") &
            pa=$!
            (trap cleanup EXIT; arm eB "$GPUS_B") &
            pb=$!
            rc=0
            wait $pa || rc=1
            wait $pb || rc=1
            [ $rc = 0 ] || { echo "✗ an arm failed (above)" >&2; exit 1; }
        fi
        analyze
        table
        ;;
    xa)
        analyze
        table
        ;;
    *)
        echo "✗ unknown step $1" >&2
        exit 1
        ;;
esac
