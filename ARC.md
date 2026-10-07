# sglang-arc-2：arc-forge 的 sglang 私有 fork

上游 [sgl-project/sglang](https://github.com/sgl-project/sglang) `v0.5.20`（`94602c9c`）加上 arc-forge 的修改。
每個 commit 就是一個改動。早期的改動是從 arc-forge `patches/` 一個個重建進來的，commit 訊息會寫明來源 patch 的檔名和 sha256。
之後所有新的 sglang 修改都直接 commit 在這裡；arc-forge 的 `patches/sglang-*` 已經凍結。

部署目標指向哪個 commit，由 arc-forge 的 `deploy/sglang.lock.json` 決定，維護工具在 arc-forge 的 `work/sglang_fork/`。遠端機器（Colab、Runpod、Kaggle）不 clone 這個私有 repo，而是用 `export_tree.py` 匯出的 tarball。

## 分支與 tag

| ref | 內容 |
|---|---|
| `main`（分支） | 所有平台共用的主線：上游 `v0.5.20` 加 0001–0032、HiSparse h01–h39（不含 h37）、h40–h43、DFlash（`speculative/arc_dflash/`），以及 2026-10-05 的 B200 修正（`c70c0178`–`616f4b07`）。新的修改都 commit 在這裡 |
| `v0.5.20`（tag） | 上游基底 `94602c9c`；升級上游就是把 `main` rebase 到新的上游 tag |
| `deploy/g4-hs55-v8`（tag） | AGI-3 Kaggle hs55 v8（RTX PRO 6000）：v6 加上 h40–h43，tree `3dcbde80` |
| `deploy/g4-hs55-v6`（tag） | 0001–0032 加上 h01–h09、h12–h22，tree `e93d999a` |
| `deploy/g4-dflash`（tag） | v6 加上 d01、d02（舊 G4 DFlash，drafter 程式當時放在 arc-forge） |
| `deploy/l4-h28`（tag） | ARC-AGI-2 Kaggle v13／v14（L4×4），tree `b9cbd753`（展開慣例） |
| `deploy/l4-h30`（tag） | ARC-AGI-2 Kaggle v15，NoMTP 快速樹 `df5847fd` |
| `deploy/l4-r9r2`（tag） | h28 加上 h29–h36、h38、h39（L4 native MTP），tree `1742410a` |
| `deploy/l4-dflash`（tag） | r9-r2 加上 l01（舊 L4 DFlash） |
| `exp/h37-swap-in-padding-counts`（tag） | r9-r2 加上 h37（r9-r2 刻意排除，從未部署） |
| `exp/kaggle-*-optin`（tag） | h28 加上三個互斥的 L4 通訊實驗（FlashInfer PCIe TP4、attention TP2 custom AR、FlashInfer attention TP2） |
| `exp/l4-dflash-opt`（tag） | `dfd2fd8f` 加上 drafter 目錄只檢查一次、`ARC_SHM_GATHER=1` 共用記憶體 all-gather（2026-10-04 L4 優化，未進 main） |
| `legacy/sglang-0.5.17`（tag） | 上游 v0.5.17 加 4 個 patch |
| `legacy/sglang-97c6978`（tag） | 上游 `97c6978` 加 13 個 patch |
| `legacy/sglang-qwen4-pr36497`（tag） | PR #36497 快照 `73a25520` 加 0001–0035（B200 venv-qwen4） |
| `legacy/sglang-qwen4-pr36497-sm89`（tag） | 同一個基底加 0001–0019、0036–0043（舊 Kaggle L4 runtime） |
| `legacy/sglang-qwen4-mtp3-73a2552`（tag） | pr36497 加 MTP3 native GR 的 0001–0006 |

`deploy/*`、`exp/*`、`legacy/*` 都是不再修改的快照，所以是 tag 不是分支（2026-10-04 前是同名分支，內容相同）。`deploy/*` 的 digest 都和部署當時的紀錄逐位相同。`exp/*` 和 `legacy/*` 沒有留下可比對的 digest 紀錄，只驗證過能完整、無衝突地套上，而且可以重現（tree 相同）。
其他 `exp/*` 分支是進行中的實驗，做完就合回 `main` 或轉成 tag。

**沒有帶進來的**：0.5.20 的 `experimental-qsa`（0018–0032），以及 pr36497 的 0044–0059（QSA offload 重建）。兩者都屬於已被 HiSparse 取代的舊 QSA offload 線，刻意丟棄，內容仍留在 arc-forge 的 git 歷史裡。

## 平台差異怎麼放

同一個 commit 跑在所有平台上，差異用旗標或 GPU 架構分派，預設維持原本行為：

- **HiSparse L4 路徑**（h23–h39：attention DP2、sm_89、分散式 spec、native MTP host KV）：
  - 都用 `ARC_HISPARSE_*`、`ARC_GDN_STATE_FMT`、`ARC_NATIVE_MTP_KV_HOST` 這類旗標控制，或只在 sm_89／DP／TP>1 才會走到。
  - G4（TP1）在 L4 旗標全關時，和 v8 逐位相同：12 條真實 prompt 的 prefill logprob 加 512 步 decode 都一樣，見 arc-forge `work/sglang_fork/`。
- **DFlash**（`ARC_DFLASH_DRAFTER=<export dir>`）：
  - 只有一個 worker：drafter 跑在這張卡的 attention-TP 群組上。
  - routed experts 預設依 compute capability 選後端：sm_100／sm_120 用 CUTLASS W4A4；sm_89 用 Marlin W4A16，並在群組內做 EP。
  - lm_head：單卡用 `TargetHead`，群組大小大於 1 時用 `PairHead`。
- **B200（sm_100）**（arc-forge `docs/research/sglang_fork_b200_20261005.md`）：
  - DFlash 群組大於 1 時預設 `ARC_DFLASH_COMPILE=det`：各 rank 的 torch.compile 各自 autotune 會選到不同 kernel，提案就會分歧。
  - NVFP4 KV 加 HiSparse 走 sm_100 的 trtllm-gen gather；上游對 hybrid GDN 只准 triton／trtllm_mha／fa4 這幾種 attention backend。
  - 0004 的 distributed MTP top-1 不支援 attention DP，起服就 raise（`SGLANG_QWEN4_MTP_DISTRIBUTED_TOPK1=0`）。
- **rejection sampling**：draft decode graph 的 `draft_probs` 在 verify 前 clone（`c70c0178`），所有平台都適用。
- **TITO session 帳本**（全部 opt-in，arc-forge `docs/plans/tito_session_ledger.md`）：`--session-replay` 讓 TokenizerManager 依 `session_id` 記下模型看過與取樣的精確 token，並把 client 重切的 id 換回取樣的 id；`--session-ledger-dir`（`-all`／`-logprobs`／`-routing`）落地。`--no-mm-literal-image-sentinel`：chat／Responses 流量裡字面 `<image>` 是文字。
- **routed experts**：`flashinfer_trtllm` 改由 kernel 的 `routing_replay_out` 寫（以前回全零）；int16、只在送 output 的 rank 建 host buffer；`meta_info` 帶 `routed_experts_meta`（dtype／shape／start），**client 一律依它解碼**。
- **帶影片的 token-id 請求不再重切**：上游 `SGLANG_MM_AVOID_RETOKENIZE`（#26555）只在純圖片時保留 client 的 id，帶影片就用處理器decode 後重新 tokenize 的結果，非標準切法會無聲漂移。qwen3-vl／qwen3.5／qwen4_exp 現在一律保留 client 的 id，只把處理器展開的媒體區塊（影片＝帶時間戳的分組）拼進 placeholder（`base_processor.splice_timestamped_media_blocks`）；對不上就報錯，不退回重切。
- **Qwen4 PLE 的媒體 id**（`--ple-mm-placeholder-ids`，opt-in，預設維持上游行為；arc-forge `docs/research/qwen38fn_ple_mm_clamp_20261007.md`）：`embed_mm_inputs` 把 radix 的 pad hash 原地 clamp 成 vocab_size−1，而 PLE 之後讀的就是同一個 `forward_batch.input_ids`，所以每個圖片／影片位置讀到 248319；HF、vLLM、llama.cpp 讀 `<|image_pad|>`／`<|video_pad|>`。開旗標時模型宣告 `mm_media_token_ids`，clamp 前用 scatter 同一份 mask 把媒體位置改寫成各自的 placeholder。chunk 之間、radix 與 HiSparse session cache 的 n-gram 前文、DP attention gather 的 n-gram id 都從這個張量來，跟著一起改。純 torch、各平台同一處。混合 precomputed／encoder 的 batch 與 EVS 會 raise。
- **DFlash**：只做 prefill 就結束的請求也釋放 draft state（`0d808d613`；以前閒置 server 上約 70 個 `max_new_tokens=1` 請求就會打掛 scheduler）。
- **執行期 VRAM**（arc-forge `docs/research/g4_runtime_vram_headroom_20261005.md`；G4 c110、0.92 真實遊戲的裝置最少剩餘 0.84 → 3.19 GiB）：
  - PLE short conv 的 extend 批次不再補成 (請求數, 最長列, 通道)，改用 `kernels/ops/qwen4_ple_varlen.py`，和補齊版逐位相同；PLE forward 的中間值用完即釋放、殘差與 mask 改 in-place（逐位相同）。不分平台、沒有旗標。
  - Qwen3-VL vision：每次 ViT forward 最多 `ARC_VIT_MAX_PATCHES`（預設 16384）個 patch、MLP 每段最多 `ARC_VIT_MLP_ROWS`（預設 8192）列；設 `0` 回到一次算完。數學相同，不保證逐位。
  - 影片解碼後的 frame 只在 `--mm-feature-transport` 為 `cuda_ipc`／`cuda_vmm` 時 pin；`cpu` 時 tokenizer process 不 pin，不再在 GPU 上多開一個 550 MiB 的 CUDA context。
  - DFlash 群組只有一個 rank 時，drafter graph 進 target 的 graph pool、在 target 的 capture stream 上 capture（`ARC_DFLASH_SHARED_GRAPH_POOL=0` 關閉）；L4 的 attention-TP pair 維持自己的 pool。
- **dense DFlash drafter**（`exp/dense-dflash` 併入）：drafter export 的 config `ffn_type: dense_shared` 時，routed experts 換成一個 dense SwiGLU（block FP8），不帶 keep／calib（`ARC_DFLASH_KEEP`／`CALIB`／`MOE` 必須不設）；L4 pair 上 attention head 切分、dense FFN 複製。
- **hs110（G4 c110、dense10240 DFlash K=8）的穩定性與速度**（arc-forge `docs/research/hs110_c110_dense10240_k8_20261006.md`）：
  - `ARC_MM_RELEASE_AFTER_PREFILL=1`（opt-in，預設 0）：請求的 prefill 一完成就釋放多模態特徵（CPU 上的 fp32 pixels）。codex 每回合重送整段歷史的截圖，c110 下 scheduler host RAM 約 9.6 → 1.3 GiB/h。被 retract 的請求會報錯（特徵已無法重新 prefill），所以會 retract 的部署維持預設。
  - `ARC_DFLASH_DRAFT_VOCAB=<{"ids": [...]}>`（單卡）：drafter head 只對這些 id 算 logits／top-k（`heads.SubsetHead`），只影響提案，輸出分布由 rejection sampling 保證；G4 c110 K=8 draft 16.3 → 12.6 ms、accept 不變。
  - 數值不變的同步移除：`embed_mm_inputs` 的 placeholder id、HiSparse prefill→decode batch、mamba pool alloc 改 pinned non-blocking；CUDA tensor 的純量賦值改 `fill_`／`index_fill_`（DFlash state pool、HiSparse coordinator／pool）；QSA extend metadata 對多模態 batch 也給 host max-position 上界（M-RoPE 位置不超過 token 序號）。吞吐量不出差異（噪音內），留著是因為少了逐請求 stream sync。
- **RL 權重同步**（`rl/v1`；arc-forge `work/qwen38fn_rl/SPEC.md` §6、§8）：`update_weights_from_disk` 的 `keep_identity`（opt-in）只把目錄裡的 tensor 寫進正在服務的模型，served name／model_path／`model_config.model_path` 不動；寫到一半失敗不做假 rollback，標成 poisoned（中止所有請求、generate 回 503，直到下一次 keep_identity 成功）；`flush_cache` 時必須全 rank 閒置否則原封拒絕，不 flush 時可在 `pause_generation(in_place)` 中途同步；speculative decoding 下拒絕。`/model_info` 多 `weights_source`、`weights_poisoned`。所有 `update_weights_*` 寫權重前後各一次 device sync（overlap 下 pause 前發出的 batch 可能還在 forward stream）；disk 路徑 flush 失敗回 `success=False`，不 assert。
- **routed experts 的 sentinel**（`rl/v1`）：host buffer 與每次 target forward 的 device 列初值 −1（`ROUTING_UNWRITTEN`），capture 沒寫到的列不再讀成 expert 0 或 KV slot 前一個主人的值；請求收尾時有 −1 或全 0 列 → 該請求不回 routing，`meta_info["routed_experts_error"]` 說明原因，server 不死（列數對不上仍 raise：那是 server 自己的帳錯）。沒有 capture site 的層（dense 層）回 −1，不再是 0。TITO 帳本 append 記 `weight_versions`（生成 token 的版本區段），server 拒收的 routing 記成 `routing_error`。
