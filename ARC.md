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
| `exp/prefill-work-aware`（tag） | main 加上 work-aware prefill delayer 草稿 `ARC_PREFILL_WORK_AWARE=1`（2026-10-08；四個門檻必填、沒有預設，只有 CPU 測試、沒上過 GPU，未進 main） |
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
- **TITO session 帳本**（全部 opt-in，arc-forge `docs/plans/tito_session_ledger.md`）：`--session-replay` 讓 TokenizerManager 依 `session_id` 記下模型看過與取樣的精確 token，並把 client 重切的 id 換回取樣的 id；`--session-ledger-dir`（`-all`／`-logprobs`／`-routing`）落地。`--session-replay-empty-think`：上一個 assistant 回合非標準的空 thinking 被 client 渲染成標準的 `<think>\n\n</think>\n\n` 時也換回取樣的 id（span `kind: empty_think`）。`--no-mm-literal-image-sentinel`：chat／Responses 流量裡字面 `<image>` 是文字。
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
  - `ARC_MM_RELEASE_AFTER_PREFILL=1`（opt-in，預設 0）：請求的 prefill 一完成就釋放多模態特徵（CPU 上的 fp32 pixels）。codex 每回合重送整段歷史的截圖，c110 下 scheduler host RAM 約 9.6 → 1.3 GiB/h。被 retract 的請求先停在 scheduler、由 tokenizer 重處理缺特徵的 item 後再排回（見下面 retract 補特徵）。
  - `ARC_DFLASH_DRAFT_VOCAB=<{"ids": [...]}>`（單卡）：drafter head 只對這些 id 算 logits／top-k（`heads.SubsetHead`），只影響提案，輸出分布由 rejection sampling 保證；G4 c110 K=8 draft 16.3 → 12.6 ms、accept 不變。
  - 數值不變的同步移除：`embed_mm_inputs` 的 placeholder id、HiSparse prefill→decode batch、mamba pool alloc 改 pinned non-blocking；CUDA tensor 的純量賦值改 `fill_`／`index_fill_`（DFlash state pool、HiSparse coordinator／pool）；QSA extend metadata 對多模態 batch 也給 host max-position 上界（M-RoPE 位置不超過 token 序號）。吞吐量不出差異（噪音內），留著是因為少了逐請求 stream sync。
  - `ARC_PLE_NVFP4=1`（opt-in，預設 0；只支援 `--ple-offload-backend pinned`＋FP8 PLE 表）：PLE 表在載入時逐 shard 於 GPU 量化成 NVFP4（e2m1＋每 16 個一個 e4m3 scale，沒有 global scale：來源已是 e4m3），每列 90 B 連續存放，host 47.7 → 26.8 GiB。gather 解碼成 bf16，相對 FP8 列 relRMSE 9.6%、cosine 0.9955；4090 上 gather 比 FP8 快 15%。所有列沒載滿就報錯。`models/arc_ple_nvfp4.py`。
  - `ARC_MM_LAZY_MEDIA=1`（opt-in，預設 0；Qwen3-VL 家族、單一 tokenizer worker）：看過的圖片／影片不再重做前處理、不送 pixel。前端以內容（h41 artifact key）記住每份媒體被 processor 展開成的 token 區塊、pad 段與 grid；請求只對沒看過的媒體跑 processor（一次呼叫、只含它們的 placeholder），其他媒體把快取區塊拼進 tokenize 結果，送 metadata-only 的 lazy item。scheduler 排 prefill 時若 lazy item 落在 extend 裡（前綴被逐出、或前綴停在它之前），就在配置任何東西前把請求退回 tokenizer（`ArcMmRematerializeReq`，帶這次 extend 內所有 item 的 hash），tokenizer 把它們補上 pixel 後重送；ids／hash 與帶不帶 pixel 無關，所以前綴照樣命中。影片（每組幀一段 offset）改走逐 item 的 embedding 路徑，只編碼與 extend 重疊的影片。被 retract 的請求補特徵後再排回（見下面 retract 補特徵）。`ARC_MM_LAZY_MEDIA_MIN_SEEN`（預設 1）：被幾個先前請求帶過才改送 lazy；`ARC_MM_LAZY_MEDIA_ENTRIES`（預設 16384）。4090 Qwen3.5-4B、3 影片＋3 圖 17 回合：prompt ids 與 greedy 輸出逐回合相同（含中途 flush），前端 CPU 54.2 → 7.9 s。`multimodal/arc_lazy_media.py`。
  - retract 補特徵（`65b0d354d`）：特徵已釋放（`ARC_MM_RELEASE_AFTER_PREFILL=1`）或是 lazy item 的請求被 retract 時，不再 raise（以前整台 server 會掛），改停在 `Scheduler.arc_mm_refill_parked`（不佔資源、保留已輸出的 token），送 `ArcMmRematerializeReq(refill=True)` 給 tokenizer；tokenizer 帶 pixel 重處理後回送 `arc_mm_refill=True` 的請求，scheduler 只拿缺特徵 item 的 feature（hash、modality、pad value、span 長度都要相同，否則以錯誤 abort 該請求），再以 retracted 身分排回。停靠中被 client abort 會照 queue 中請求的方式清掉。G4 hs110 `SGLANG_TEST_RETRACT` 20 分：17 次 retract 全部補回、0 Traceback。
- **DFlash／QSA／extend 的 2026-10-08 改動**（arc-forge `docs/research/sglang_five_opt_20261008.md`；G4 c110、dense10240 acc-v4 FP8 drafter、K=8，每臂真實遊戲 20 分鐘，各臂 0 Traceback）：
  - drafter 的 prefill ids 在 target forward 之前取（`DFlashDraftWorker.prefill_draft_ids`，預設生效、沒有旗標）。target 的 mm embed 會原地改寫媒體位置（改成 vocab_size−1，或 `--ple-mm-placeholder-ids` 的 placeholder）。部署路徑上 eager runner 先把 input_ids 複製進自己的 buffer，drafter 本來就讀到 `<|image_pad|>`，輸入逐位不變；`SGLANG_EAGER_INPUT_NO_COPY`、`--enable-pdmux`、full prefill CUDA graph 這三條路徑共用同一個張量，以前 drafter 在那裡會讀到 248319／`<|video_pad|>`。`_draft_extend_for_prefill` 的 `draft_ids` 必填，漏傳或 shape 不符就 raise。
  - `ARC_DFLASH_EXACT_PREFILL=1`（opt-in，預設 0）：drafter state 只依賴最後 `receptive`＝層數×(window＋3) 個位置（acc-v4 為 5,135），prefill 就只 refresh prompt 的最後 receptive 個位置：接續的 chunk 用 append，整段落在 tail 之前的 chunk 只留最後一格 GR，從 tail 起的新 window 用真的前一格 GR。結果等於整段一次 refresh（float64 oracle、各種切法差約 1e-12）。歷史從沒進過 drafter 的 resumed prefix 仍照舊從 chunk 尾端 bootstrap（計數 `bootstrap`）。
  - `ARC_DFLASH_SESSION_PARK=1`（opt-in，需 `ARC_DFLASH_EXACT_PREFILL=1`，否則起服 raise）：session 請求結束時 drafter slot 不釋放，掛在 session_id 下停車；同 session 下一個請求在 window 讀過的每個位置 ids 都相同（媒體位置比對 pad hash）時接手，只 append window 之後的部分。只有 prompt 的最後一個 chunk 處理完的請求才登記停車：chunk 之間結束（abort）的請求，window 可能從整段 prompt 的 tail 才開始、前面沒有歷史，所以直接釋放；沒有登記的 session 請求一進 decode 就 raise。slot 不夠時從最舊的停車 slot 收回，不另配 VRAM。每 200 個 extend batch 印一行 `DFlash exact prefill: {...}`（接手數與各種 miss 原因）。G4（登記規則修正前的版本，差別只在 prompt 中途結束的請求）：停車 2,494 次、接手 2,430 次（97.4%），miss 只有 gap 5、ids 1。限制：attention DP 下 session 換到別的 DP rank 就接不到（`park_none`）；`/close_session` 不立即釋放停車 slot，靠 LRU 收回。
  - `ARC_DFLASH_SPARSE_Q=1`（opt-in，預設 0；需 `--speculative-use-rejection-sampling`，否則起服 raise）：rejection sampling 的 q 不再 scatter 進 dense `[max batch, K−1, vocab]` FP32 buffer（c110 K=8 為 0.71 GiB，在 target 量 KV 預算之前配置），改成每個位置的 selector 候選 id 與機率 `[batch, K−1, 16]`（`eagle_info.SparseDraftProbs`），dense buffer 不配置；verify 走 `reject_sampling.chain_speculative_sampling_sparse_q_triton`，與 dense kernel 同一套 accept／residual／inverse-CDF 運算（含 residual 為零時回 V−1 的舊行為）；G4 上部署 shape（110×8×248,320）與交替 batch 的 graph replay 都和 dense kernel 逐位相同，L4 未測。G4：state pool 2.00 → 1.29 GiB，同 memfrac 下 KV device rows +3.4%。省下的位元組在同 memfrac 下會變成 KV；要換成 headroom 得同時把 memfrac 調低約 0.0075（0.71 GiB／95 GiB）。native MTP／EAGLE 的 dense 路徑不動。
  - QSA verify 工具（都 opt-in，沒設時行為不變）：`ARC_QSA_SELECTION_STATS=N`＋`ARC_QSA_SELECTION_STATS_LAYERS=3,23,47` 統計前 N 個 TARGET_VERIFY forward 裡同一請求 K 列的選取重疊，印一行 `ARC_QSA_SELECTION_STATS {...}`；stats op 會錄進 verify graph、之後一直執行，只用於除錯 run。G4 K=8：相鄰兩列共用的比例（g2 common_fraction）0.79／0.74／0.83（layer 3／23／47），沒到 0.85 門檻，所以不做共用歷史的 grouped kernel。`ARC_QSA_DIRECT_TILE_TABLE=<json>`：h22 direct decode 依精確 shape 查量測過的 tile（格式與檢查見 `qsa/direct_tiles.py`），沒設時 G4 與 sm_89 兩組內建 tile 不變；目前還沒有量完的表。
  - `ARC_EXTEND_TELEMETRY=1`（opt-in，`ARC_EXTEND_TELEMETRY_INTERVAL_S` 預設 60）：每個 extend batch 在 target 前後與 drafter refresh 後各記一個 CUDA event 和 host 時間（不 sync），每個 interval 印一行 `ARC extend telemetry`：依（請求數×新 token）分 bin 的 target／refresh GPU 與 CPU ms，以及 prefill delayer 的放行原因（ms 上限到期另記 `wait_ms_cap`）。MIXED batch 會 raise。只在 G4 TP1 驗過：attention DP 下 decode 模式的 batch 若因 `is_extend_in_batch` 進了 extend 分支，`batch_shape` 會 raise，L4 開之前先短跑。`managers/arc_extend_telemetry.py`。
- **RL 權重同步**（`rl/v1`；arc-forge `work/qwen38fn_rl/SPEC.md` §6、§8）：`update_weights_from_disk` 的 `keep_identity`（opt-in）只把目錄裡的 tensor 寫進正在服務的模型，served name／model_path／`model_config.model_path` 不動；寫到一半失敗不做假 rollback，標成 poisoned（中止所有請求、generate 回 503，直到下一次 keep_identity 成功）；`flush_cache` 時必須全 rank 閒置否則原封拒絕，不 flush 時可在 `pause_generation(in_place)` 中途同步；speculative decoding 下拒絕。`/model_info` 多 `weights_source`、`poisoned`（bool）、`weights_poisoned`（原因或 null）。所有 `update_weights_*` 寫權重前後各一次 device sync（overlap 下 pause 前發出的 batch 可能還在 forward stream）；disk 路徑 flush 失敗回 `success=False`，不 assert。
- **routed experts 的 sentinel**（`rl/v1`）：host buffer 與每次 target forward 的 device 列初值 −1（`ROUTING_UNWRITTEN`），capture 沒寫到的列不再讀成 expert 0 或 KV slot 前一個主人的值；請求收尾時有 −1 或全 0 列 → 該請求不回 routing，`meta_info["routed_experts_error"]` 說明原因，server 不死（列數對不上仍 raise：那是 server 自己的帳錯）。沒有 capture site 的層（dense 層）回 −1，不再是 0。TITO 帳本 append 記 `weight_versions`（生成 token 的版本區段），server 拒收的 routing 記成 `routing_error`。
- **G4 吞吐的 2026-10-08 改動**（arc-forge `docs/research/qwen38fn_dflash_g4_c110_throughput_20261008.md`；G4 c110、dense10240 acc-v4 FP8 drafter、K=4，每臂真實遊戲 20 分鐘，各臂 0 Traceback；三項都是 opt-in，沒設時行為不變）：
  - `ARC_GDN_HOST_CHUNK_INDICES=1`（`check` 會再用舊算法重算每個 plan、不同就 raise）：GDN backend 的 `init_forward_metadata` 用 `extend_seq_lens_cpu` 在 numpy 算出 chunk prefill 的 chunk indices 與 chunk offsets（chunk 64，以及 64 token 以下的 batch 給 `chunk_fwd_o` 的 BT），從 pinned memory 非阻塞送上 GPU，`fla/index.py` 對同一個 cu_seqlens 物件直接回傳。舊路徑在每個 extend 的第一個 GDN 層同步 stream（`prepare_chunk_indices` 的 `.tolist()`、`prepare_chunk_offsets` 的 pageable `new_tensor([0])`）。開旗標後，CUDA cu_seqlens 上沒規劃到的請求會 raise。
  - `ARC_MM_NO_SYNC=1`：CPU 上的 pixel 特徵先複製到 pinned memory 再送 GPU（新截圖 9.8 MB 的 pageable copy 會讓 host 等到 stream 走到它），Qwen3-VL vision encoder 的 attention max_seqlen 改用 host 的 cu_seqlens 算，不再對 device 那份做 `.item()`。
    兩個旗標一起開，target extend 裡就沒有任何阻塞呼叫；但吞吐只多約 0.3%（落在噪音內）：eager extend 受限於 host 發 kernel 的速度（CPU 與 GPU 每批都約 120–130 ms），拿掉同步只換到提早開始發 kernel 那段，等待移到 drafter refresh。價值在讓 target extend 可以 graph 化。
  - `ARC_DFLASH_REFRESH_ASYNC=1`（需 `ARC_DFLASH_EXACT_PREFILL=1`，否則起服 raise）：prefill refresh 的 position 檢查改成 device assert（`require`），window 的 cursor 由 prefill 寫的 host 鏡像提供（請求 decode 後丟棄），停車 slot 的 (cursor, pending, window end) 在停車時於 forward stream 複製到 pinned memory，`settle` 讀那份。refresh 不在 forward stream 上會 raise。G4：refresh 每批 GPU −4.2 ms（−17%）、CPU −10 ms（按請求數×token 分 bin 對齊）。
  - `ARC_RS_FUSED_SOFTMAX=1`（需 `ARC_DFLASH_SPARSE_Q=1`，否則 raise）：verify 直接從 target logits 做 sparse-q rejection sampling（`reject_sampling.chain_speculative_sampling_sparse_q_from_logits`），先逐列算 div_rn(logit, T) 的 online max／sum，讀到 target 機率時才算 exp(·−M)/S，不寫出 [bs×K, vocab] 的 softmax；有 top-k／top-p 的 batch 走原路徑。精度比 torch FP32 更接近 float64；2,200 次共用亂數的判定只有 4 次不同（亂數落在邊界上）；400k 次 Monte Carlo 分布檢驗通過。kernel K=4 1.95 → 0.64 ms、K=8 3.71 → 0.93 ms；G4 decode cycle −1.3～−1.6 ms（−1.7～−2.0%），accept 長度不變。
  - 三項全開（對同一棵樹全關，按請求數×token 分 bin）：target 每批 GPU −2.5 ms、refresh −5.7 ms（−24%），decode 每輪 −1.0～−1.4 ms；gen tok/s +1.4%（第 5–16 分鐘 3,196 對 3,153，小於 run 間噪音，看分項）。
- **RL extend 的 host 同步移除**（branch `exp/b200-perf-extend`，2026-10-08；兩個旗標都是 opt-in，沒設時行為逐位不變；GPU 只驗過和 `ARC_QWEN4_EXP_PREFILL_BCG=1` 同開的 TP1 NVFP4，對 eager 基準逐位相同，見下一項的 GPU 驗證結果；不開 BCG 的 eager 路徑與 TP2 還沒上 GPU）：
  - `ARC_MAMBA_TRACK_HOST=1`
    - 作用範圍：`extra_buffer` 下 plain EXTEND 的 mamba radix track。
    - 原本的問題：計算 track plan 時要從 device 把資料讀回 host，而這些讀取會同步 stream。這些讀取包括：
      - scheduler 每個請求一次的 `.item()`，以及三個 pageable 的 `torch.tensor(list, device=)`。
      - backend 的 `mask.any()`。
      - conv indices 的布林遮罩索引，以及 `clamp(0, qsl[-1] - 1)`。後者因為傳入 0 維 tensor，會綁到 Scalar overload 而觸發 `.item()`。
      - `_init_track_ssm_indices` 的 6 個 `.cpu()`。
      - GDN 的 `nonzero()`。
      - FlashInfer checkpoint plan 的 4 個 `.cpu()`。
    - 開旗標後的做法：
      - scheduler 記下每個請求這次要用的 ping-pong 位置。`mamba_track_indices` 改成在 schedule stream 上把各請求的 ping-pong buffer `stack` 起來再 `gather`，跟所有寫入這些 buffer 的操作在同一條 stream、照順序執行。
      - mask 與 seqlens 改用 pinned＋non_blocking 上傳，host 端的 list 同時保留在 `ScheduleBatch`／`ForwardBatch.mamba_extend_track_{mask,seqlens}_cpu`。
      - backend 用這兩份鏡像加上 `extend_{seq,prefix}_lens_cpu`，在 host 上算出所有列的選擇（`layers/attention/mamba/track_host.py`），一次 pinned 上傳。slot id 在 device 上用整數索引 `gather`。
      - 值、dtype、device 都和舊路徑相同。
    - 只有 plain EXTEND 走新路徑，decode、verify、draft-extend、mixed 都照舊。
    - 會 raise 的情況：plain EXTEND 帶了 track mask 卻沒有鏡像，或鏡像長度對不上。已知會碰到的有 DP attention 的 `convert_decode_to_extend`。prefill CUDA graph 的 capture batch 與 replay batch 現在都帶鏡像（見下一項的 runner 改動），不再 raise。
    - `check` 模式：每個 extend 同時跑舊路徑，dtype、shape、device、值任何一項不同就 raise。這個模式本身會同步，只用來驗證。每比對 100 個 extend 印一行 `ARC_MAMBA_TRACK_HOST=check: all comparisons matched so far: {...}`，列出比對過的 extend、列數、checkpoint plan、slot gather 次數，用來確認這條路徑真的有跑到。另外，check 模式會把被拿掉的同步加回來，所以拿掉同步後才可能出現的 race 在這個模式下看不到，要另外用 `=1` 跑一輪驗證。
    - 測試在 `test/registered/unit/layers/test_mamba_track_host.py`，涵蓋兩件事：
      - 用隨機 batch 對舊路徑逐值比對，GDN 與 Mamba2 都有。
      - device tensor 放在 meta 上，任何 host 讀取都會 raise。舊路徑當陽性對照，確實會 raise。
  - `ARC_MROPE_EXTEND_PINNED=1`：`_compute_mrope_positions_extend` 拼出 3×token 的 int64 positions 後，先 pin 再 non_blocking 上傳，數值相同。原本從 pageable 記憶體上傳時，driver 要經 staging buffer 複製，大小超過 staging buffer 就會等 stream。
- **Qwen4-Exp 純文字 extend 的 prefill CUDA graph**（branch `exp/b200-perf-extend`，2026-10-08；opt-in，沒設時行為不變；TP1 NVFP4、TP2 FP8 SPEC=none、TP2 FP8 DFlash K4 都已在 GPU 上對 eager 逐位驗過，結果見本項最後）：
  - `ARC_QWEN4_EXP_PREFILL_BCG=1`：讓 breakable（BCG）prefill runner capture Qwen4-Exp 的 transformer body（`models/qwen4_exp_prefill_bcg.py`）。
    - capture 的部分：HC mix／combine、所有投影與 norm、RoPE、MoE（routed experts 經 `routing_replay_out` 寫靜態 buffer）、attention TP 的 all-reduce。
    - 留 eager 的部分（BCG 斷點，replay 時從 forward context 讀當次的 batch）：
      - GDN 核心：沿用 `RadixLinearAttention` 既有的斷點（和 Qwen3.5 相同）。
      - QSA attention 核心：沿用 `RadixAttention` 既有的斷點，餵 `topk_indices`。
      - QSA indexer：每個 QSA 層一個斷點。它的壓縮寫入、compressed key gather、row chunk 都依 host 上的批次長度決定，不能 capture。結果寫進 captured segment 配好的 int32 [bucket, topk 寬]，padding 列填 -1。BCG 下 indexer 不在 alt stream 上重疊。
      - PLE：PLE 層一個斷點，裡面依當次 batch 做 prepare、跑 PLE 層、commit N-gram 歷史，輸出寫進 captured buffer。原本 commit 在最後一層之後；中間沒有任何東西讀寫 N-gram 歷史，所以模型只有一個 PLE 層時兩者等價（多於一層會 raise）。BCG 下 PLE embedding 的 prefetch 關掉（它的 buffer 依當次 token 數配置，replay 時會一直配新的）。
    - body 在 BCG 下回傳 (hidden, hc_hidden)，runner replay 後交給 `Qwen4ExpVLModel.unpack_breakable_body_output`，所以 `logits_output.hidden_states`（DFlash drafter 讀的 GR）是 replay 出來的 HC stream。
    - 帶媒體的 batch 也走 graph。eager tail 對靜態 batch 跑外層 forward：ViT＋merge 寫進 `input_embeds` slot；媒體位置的 ids 原地改寫在靜態 input_ids 上，PLE break 讀同一份（eager 讀的是改寫後的 live ids）；M-RoPE 從 `mrope_positions` slot 讀，padding 尾端是 0。原本這裡把帶媒體的 batch 一律退回 eager，但 `ForwardBatch.mm_inputs` 是請求層級的（含 cached prefix 裡的圖），agent 的請求歷史裡都有圖，G1（2026-10-08，codex RL 負載）因此 1,644 個 prefill batch 全部走 eager，連 extend 範圍內沒有新圖的也一樣。
    - 會 raise 的情況：沒設旗標卻在 BCG 下跑 Qwen4-Exp（例如明確指定 `--cuda-graph-backend-prefill breakable`）、draft worker（native MTP 的 draft prefill 沒驗證）、DP attention、PP、QSA HiSparse、SPEC 不是 none 也不是 DFlash drafter。
    - 開旗標時 Qwen4-Exp 加進 multimodal BCG allowlist，其他相容性規則照常適用；bucket 用 `--cuda-graph-bs-prefill` 或 `--cuda-graph-max-bs-prefill` 控制。
  - runner 的通用改動（所有模型、不看旗標；數值不變）：replay batch 帶上 `req_pool_indices_cpu` 與 mamba track 鏡像（之前 QSA 的 prefix extend 在 replay 內會 `.tolist()`）；capture batch 的 track 鏡像從它的靜態 slot 讀出來，`ARC_MAMBA_TRACK_HOST` 因此可以和 prefill CUDA graph 一起開。
  - CPU 測試 `test/registered/unit/models/test_qwen4_exp_prefill_bcg.py`：PLE 斷點對 eager 的三步（prepare、層、commit）在 bucket padding 下輸出與 N-gram 歷史逐位相同；indexer 斷點的列數、padding 與各種不符會 raise；VL body 的回傳契約與 unpack；runner 的 unpack、鏡像，帶媒體的 batch 不被擋。
  - GPU 驗證 `scripts/arc/qwen4_exp_prefill_bcg_check.py`：同一份 build、同樣旗標的兩台 server 先後起（A 不開、B 開 `ARC_QWEN4_EXP_PREFILL_BCG=1`），兩邊都要開三個確定性旗標 `ARC_QSA_TOPK_EXACT=1 SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 ARC_HC_MIX_DETERMINISTIC=1`，要比 routing 再加 `--enable-return-routed-experts`；DFlash 兩邊還要同一個 `--random-seed`（見下面的結果）。先把 A 起兩次（A1、A2）互比當對照，必須逐位相同才有資格比 A 對 B。`collect` 送 exact bucket、會 padding 的長度、兩條同批、padding 隔離（iso：同一個 prompt 單獨送〔replay 補 padding row〕對比旁邊配一條補滿 bucket 的 filler 請求，各自之後再送 prompt+256 的 follow-up，接在 cached KV 與 extend 尾端 track 的 mamba state 上）、iso 的兩序列版（pad2/fill2：P 後面多一條 8 token 的 companion，兩邊每次 extend 都 ≥2 條序列，原因見下面 TP2 的 GDN CP）、10k cached prefix 上 extend 1024（exact bucket）與 800（padding）的 case（graph 上全是 exact bucket 的 case 排在所有 padding case 之前，所以 A、B 在每個 gate case 之前送過的請求完全相同；spec decode 會受前面的請求影響）。每個請求記下 prompt token 的 logprob 與 top-5、16 個 greedy token 與 logprob、cached_tokens、routed experts；`compare` 要求每個 extend 都落在 exact bucket 的 case（exact、prefix、iso 的 fill／fill2 與其 64 對齊的 follow-up）逐位相同（padding 與同批 case 只報差異：eager 跑在 M=n、graph 跑在 M=bucket，GEMM 可能選不同 tile）；`isolation` 在 graph server 上要求 pad2 對 fill2（TP1 另加單序列的 pad 對 fill，`--gate-single`）：prompt 的 routed experts 與整個 follow-up 逐位相同，padding row 漏進真實 row、KV 或 GDN state 都會在這裡現形（prompt 自己的 logprob 走 eager LM head、M 不同，只報；spec decoding 時 follow-up 只 gate prompt 端，decode 只報）；`log` 印 prefill capture 的時間與記憶體、KV 大小、prefill batch 走 graph 的比例與前 40 個 batch 的組成。
  - GPU 驗證結果（2026-10-08，`scripts/arc/qwen4_exp_prefill_bcg_validate.sh`，arm 目錄 `runs/qwen38fn_b200_perf/20261008-bcg-<RUN>-*`；TP1 是 s1–s3 `RUN=r3`，單序列 isolation 那時是 gate、pad2/fill2 還沒有）：
    - TP1 NVFP4、SPEC=none、18 個 bucket（16…4096）：capture 8.6–9.0 s，prefill graph 佔 2.41 GB（可用 25.58→23.17 GB）。一輪 collect 的 27 個 prefill batch 裡 25 個走 graph；走 eager 的兩個是 VLM warmup（帶圖，走媒體退回）和 8192 token 的 chunk（超過最大 bucket）。
    - A1 對 A2（重啟＋換卡的對照）、A1 對 B1、A1 對 B2（B1 再加 `ARC_MAMBA_TRACK_HOST=1 ARC_MROPE_EXTEND_PINNED=1`）：每個 gate case 都逐位相同（prompt 與 output logprob、top-5、16 個 greedy token、48 層 routed experts、cached_tokens）。B1、B2 的 isolation 四組（17/722/1088/3328）全部逐位相同。
    - pad case 的 A/B 差（max |Δlp| 1.7–2.8，routing 從第 0 層就分岔）是 eager 自己隨 M 變的量：A1 裡同一個 prompt 跑在 M=n 與 M=bucket，17/722/1088 差 1.8–2.9，3328 與 4096 則逐位相同。serving 本來就不是 batch-invariant，這個差不是 BCG 造成的。
    - `ARC_MAMBA_TRACK_HOST=1` 本身不印 log；它和 BCG 旗標經同一個 `ARM_ENV` 進 unit，而 BCG 旗標確實生效（有 capture，batch 顯示 `cuda graph: True`）。
    - TP2 FP8 SPEC=none（s4，`RUN=r4`）：capture 9.3 s，prefill graph 每張卡 2.62 GB（可用 23.33→20.70 GB）；43 個 prefill batch 裡 41 個走 graph。A1 對 A2、A1 對 B1 的 27 個 gate case 全部逐位相同；B1 的 pad2/fill2 四組逐位相同。另外 B 的 pad2（graph，含 padding）和 A 的 pad2（eager，不含 padding）在 722 與 3328 逐位相同（這兩個大小 eager 的 M 變了也不換 GEMM kernel），等於直接比到「graph 加 padding」對「eager」。
    - TP2 的 GDN CP（不是 BCG 的問題，eager 一樣）：flashinfer 0.6.18 的 `chunk_gated_delta_rule` 預設 `use_cp="auto"`，在 SM100 上 num_seqs × 每卡 v-head × 4 < SM 數（B200 是 148）就改走 context-parallel kernel。TP2 每卡 24 個 v-head：單序列的 extend 走 CP，兩條以上不走；TP1 每卡 48 個 v-head，永遠不走。extend 要求 state checkpoint（mamba track 不落在最後一格）時，CP 會退回非 CP 並印 RuntimeWarning。CP 和非 CP 的數值不同，所以 TP2 下同一個 prompt 單獨 prefill 和跟別的請求同批 prefill，結果不同（r3 的單序列 isolation 因此在 17/1088/3328 不同；722 有 checkpoint，所以相同）。
    - TP2 FP8 DFlash K4（s4d，`RUN=r5`）：capture 9.1–9.35 s，每張卡 3.12 GB；43 個 batch 裡 41 個走 graph。兩邊都加 `--random-seed 1234`（`r5-tp2-dAs`／`r5-tp2-dBs`）時，27 個 gate case 全部逐位相同，連 DFlash decode 的 token、logprob、spec verify 次數與 accept length 都相同。但 r6 推翻了「固定 seed 就可重現」：見下面帶媒體的結果，r5 這次全對是運氣。
    - 沒固定 seed 時，DFlash 的 decode 連 eager 對 eager 都不可重現（`r5-tp2-dA`／`dA2`、`dB`／`dB2` 與 A 對 B，三組都是同三個 case 只差在 decode，target 端逐位相同）：server 每次起來的 `random_seed` 不同，DFlash 的 verify 會從 server 的 RNG 抽數。固定 seed 後，同一台 server 上前面的請求不同，RNG 的位置也不同，所以 spec 下 isolation 只 gate follow-up 的 prompt 端。accept length 在 eager 是 1.42–1.47、graph 是 1.41–1.45（隨機 token 的 prompt）。
    - 帶媒體（`RUN=r6`；bucket 加到 8192，和 G1 相同）：
      - TP2 FP8 SPEC=none（s4）：capture 10.5 s，每張卡 3.60 GB；56 個 prefill batch 全部走 graph。A1 對 A2、A1 對 B1 的 37 個 gate case 全部逐位相同，包括：extend 內有新圖（`mm_exact_1024`）；舊圖在 cached 1024 的 prefix、extend 是純文字（`mm_prefix_text_512`，G1 被擋的就是這種）；cached prefix 之後再來一張新圖（`mm_prefix_img_1024`）；一張圖被 8192 的 chunk 邊界切開（`mm_chunked_8704`，兩個 chunk 都走 graph）；一條帶圖請求和兩條文字請求同批補滿 768（`isomm_704_fill2` 及其 follow-up）。padding 的 `mm_pad_700`（只報）也逐位相同；isolation 的帶圖版逐位相同。`mm_pair_2048` 只報：帶圖的請求前端處理較慢，晚到 scheduler，三個 arm 都拆成 1000 與 1048 兩個 batch。
      - TP2 FP8 DFlash K4、兩邊 `--random-seed 1234`（s4d＋s4dc：`r6-tp2-dA`、`dB`、`dA2`、`dB2`）：capture 10.2 s，每張卡 4.13 GB；全部走 graph。dA 對 dB、dA 對 dA2、dB 對 dB2、dA2 對 dB2 的 target 端（prompt logprob、top-5、prompt 那幾列的 routing、cached、extend 給的第一個 token）35 個 gate 全部逐位相同。decode 每組都有一個 follow-up 不同，eager 對 eager、graph 對 graph 也一樣，所以固定 seed 也不能讓 DFlash 的 decode 重現；spec server 的 `compare` 因此只 gate target 端、decode 只報。
    - 媒體（ViT＋merge）的成本（`vittel`，DFlash arm 的 extend telemetry，CUDA event，沒有 profiler；2048-token 的 extend，每種 8 個；`r6-tp2-dA2`／`dB2`／`dAv`／`dBv`）：eager 純文字 83–84 ms；有新圖（1 或 4 張 512²、或 1 張 1024²）多 15–19 ms。graph 純文字 59 ms（少 24 ms，−29%）；有新圖多 17–22 ms，約佔帶新圖 extend 的 22–27%。張數和尺寸幾乎沒有影響（launch-bound）。同一張圖第二次出現時命中 embedding cache，只多 1–4 ms。ViT 的 graph 化還沒做。torch profiler（`vitcost`）下兩邊都被放大：mm embedding 36–45 ms，language model 157–178 ms。
- **大 extend batch 的 kernel 分解**（2026-10-08／09，`RUN=r3`；TP2 FP8 DFlash K4、H1 B 的部署旗標、LEDGER=0、不開確定性旗標；工具 `scripts/arc/qwen4_exp_extend_profile.py`＋`.sh`，結果 `qwen38fn_b200_perf/20261008-xprof-r3-{eA,eB,eC}/` 與 `20261008-xprof-r3-table.txt`）：
  - 做法：每個形狀一次 /generate 送 4 條請求，每條接在自己已 cache 的 prefix 後面（每批都檢查是一個 prefill batch、cached_tokens 等於 prefix）。先跑一批暖身不計，再 6 批量 extend telemetry（CUDA event，沒有 profiler），再 2 批開 torch profiler 拆 kernel。eager 的 kernel 依 launch 的 module stack 歸類，graph replay 的 kernel 用 eager 臂同形狀的「kernel 名稱 → 類別」比例分攤；all-reduce 把 TP0、TP1 第 k 次配對，取 min 當傳輸，其餘算等另一個 rank。三臂：eA eager、eB BCG、eC＝eB＋`ARC_DFLASH_EXACT_PREFILL=1 ARC_DFLASH_REFRESH_ASYNC=1`。
  - H1 B 的 k4+/<4096 batch 平均 4.9 條、2,885 new token、cached prefix **整批合計** 43.7k（每條約 9k），target 88.8 ms GPU；`p10k`（4×704 new，每條 prefix 10240）就是這個形狀。
  - 結論：p10k 在 BCG 下 target 78.2 ms GPU，其中 kernel 約 70.7 ms、空隙加等待 7.5 ms；roofline 只有約 17 ms。每條帶一張新圖時 99.2 ms。生產每條請求平均 1.37 張新圖，88.8 ms 落在兩者之間。
  - p10k、BCG（eB）的分項（每 rank、每次 extend；roofline 取 FLOPs／峰值與 bytes／HBM 的較大者）：

    | 項目 | 實測 ms | roofline ms | 偏離的原因 |
    |---|---|---|---|
    | MoE routed | 17.7 | 8.6 | FC1＋FC2 10.8，約是讀權重下限的 80%（256 個本地 expert 全被打到，每層 1.26 GB FP8）；activation 4.2、finalize 1.4、routing 1.1 是額外的 |
    | QSA sparse prefill（`_sparse_gqa_chunk_prefill`） | 13.4 | 0.38 | 每條 query 各自 gather 自己選中的 2051 個 key，query 之間不共用：每層 2.95 GB，實際約 2.6 TB/s。和 prefix 長度無關 |
    | bf16 dense GEMM（全部 nvjet） | 13.5 | 約 5.5 | checkpoint 的 `modules_to_not_convert` 讓非 expert 的線性層全是 bf16；cuBLAS 在 M≈3k 約半峰；BCG 墊到 3072 bucket |
    | HC＋norm（GEMM 以外） | 11.7 | 未融合約 5、融合約 2 | 每個 HC block 對 4 倍寬的 stream 讀寫約 9.5 遍；`hc_combine` 每次 53 µs |
    | all-reduce | 7.4 | 1.55 | 傳輸 3.7，另 3.7 是等另一個 rank（eager 臂的等待是 68 ms：CPU 下發慢的那個 rank 讓另一個空等） |
    | GDN core | 5.3 | 0.6 | chunk kernel 1.2、conv 1.0，其餘約 3 是 state 的 gather／copy |
    | QSA indexer | 4.8 | 0.15 | TileLang score kernel 的 grid 只有「列數／32」個 CTA；隨總 prefix 長大：pmix 15.2、p45k 23.4 |
    | 其他（PLE 等） | 約 2.5 | | |

  - target GPU ms（eA eager／eB BCG）：cold 79.9／62.9，p10k 76.5／78.2，pmix（總 prefix 107k）89.7／90.4，p45k（總 prefix 184k）99.2／100.0，4×352 81.6／53.2，4×512 82.4／59.3，4×1408 128.3／133.4，p10k 帶 4 張新圖 101.4／99.2。eager 在 1.4k–2.8k token 都是 76–82 ms：CPU 下發 kernel 本身就要 74–82 ms。BCG 只在小 batch 省 GPU 時間，到 p10k GPU 本身的工作量已經有 67–71 ms，BCG 沒有空間可省（H1／H2 吞吐 +2.3%／+4.0% 與此一致）。
  - drafter refresh：eB 的 refresh CPU 65.3 ms、GPU 16.5 ms，refresh 區間內 8 次 `cudaStreamSynchronize` 共 27.6 ms（p45k 42 ms、4×1408 77 ms），是在等 target 的 GPU 尾巴（BCG 下 target 的 CPU 28.5 ms 就結束，GPU 要 78 ms）。eC 的 refresh 區間 0 次 sync，target 不變（77.7 ms）；refresh GPU 16.5→15.1 ms、CPU 65.3→55.4 ms（各形狀 GPU 少 1.4–5.5 ms、CPU 少約 10 ms）。剩下的 CPU 等待不是 host 讀取，推測是 refresh 約 1500 次 eager launch 排在 target 的 GPU 工作後面、launch queue 滿了（未驗證）。driver 的請求不是 session，EXACT_PREFILL 下每條都從自己的 chunk 重建 window，refresh 的 token 數和 eB 相同，差只反映拿掉 host 讀取；吞吐要看 BCG 下的 A/B。
  - 候選（依收益÷風險）：(1) BCG 在 2048–4096 之間加 bucket（只改旗標；p10k 墊到 3072 時 BCG 的 kernel 時間比 eager 多 4 ms，落在 2048 bucket 時沒差）；(2) QSA indexer 的 TileLang kernel 也沿 key 切開成 2D grid、拿掉整張 `torch.zeros`、把除法併進 kernel（p10k 約 −2、pmix 約 −11、p45k 約 −18 ms）；(3) ViT／mm 的 CUDA graph（BCG 下每批 4 張圖多 21 ms，ViT 本身的 kernel 約 6 ms）；(4) QSA sparse prefill 重寫成相鄰 query 一起讀選中 block 的聯集（−8～11 ms）；(5) HC prefill 融合（−6～8 ms）；(6) MoE activation＋finalize（flashinfer 內部，約 −3 ms）；(7) drafter refresh 本身（sm80 cutlass SDPA 3.1 ms、copy／fill 約 725 次 launch 2.9 ms）；(8) EXACT＋ASYNC 的吞吐 A/B；(9) GDN core 周邊的 state copy（約 −2 ms）。
  - 不必做（量過了）：
    - MoE 的 autotune tactic：2,816 token 落在沒有快取的 3072 bucket、走 flashinfer 預設 tactic，但選到的 FC1／FC2 kernel 和 2048 調好的 tactic 同名（同一個 t128x64x128 tile）；eager 的 FC1 在 2048／2816／5632 token 是 5.28／6.56／11.54 ms，隨 token 數連續成長，沒有跳階。
    - QSA sparse prefill 的 Triton 配置：`_get_best_config` 在 B200 落到 L20 表（BLOCK_N=16、1 warp）；microbench（`20261008-xprof-r3-sparse_prefill_cfg_bench.py`，p10k 形狀）重現每層 1128 µs，掃 8 組 BLOCK_N／warps／stages 全部 1169–1744 µs，沒有一組更快。瓶頸是每條 query 各自 gather，不是 tile。
- **decode cycle 的分解與三個 decode opt-in**（2026-10-09，branch `exp/b200-perf-extend`；TP2 FP8 DFlash K4、H1 B 的部署旗標＋RL 帳本、真 harness 16 groups × 12–15 分鐘；工具 `scripts/arc/qwen4_exp_extend_profile.py`（`capture`／`decode`／`phase-compare`／`ledger-check`）、`scripts/arc/qwen4_exp_verify_parity.py`、`scripts/arc/qwen4_exp_decode_profile.sh`（`d1`／`par`）；結果 `qwen38fn_b200_perf/20261009-dprof-*`、`20261009-par-*`、`20261009-hsneg2`）：
  - 判準：`ARC_SPEC_PHASE_TIMING` 的逐 bs CUDA event（不開 profiler），兩對卡同時跑 A／B，比 bs 80–128 每個 bs 的 cycle ms 的中位差。同配置 A 對 A：同一對卡 −0.002／+0.001 ms，跨卡對 −0.03 ms（中位）。profiler 視窗那幾個 bs 會被墊高（平均數會被拉，用中位）。整輪吞吐（gen tok/s）在這個精度下是噪音。
  - cycle（bs 113–119、每條 context 約 10–20k）：40.7–40.9 ms，GPU-bound（cycle 間空隙中位 0.05–0.8 ms）。draft 5.5、verify 29.2、post 5.1–5.3（GDN commit kernel 3.1–3.3、RS 0.87、logprob 0.95）、commit 0.16。對 roofline 偏離最大的：QSA verify 6.0–6.3 ms 對 0.28（`_compact_kv` 2.9＋fmha 1.9），drafter propose 5.6 ms 對 0.41（653 個 kernel），dense GEMM 4.7 對 0.79，MoE 11.3 對 7.6。
  - `ARC_RS_FUSED_SOFTMAX=1 ARC_DFLASH_SPARSE_Q=1`：−0.42 ms／cycle（RS kernel 0.87 → 0.43 ms）；帳本每個 token 都有 logprob、無 NaN、均值與分位數在對照的範圍內。
  - `ARC_QSA_FP8_DIRECT=1`（h22 的 direct sparse 讀 FP8 pool，第一次用在 B200 的 target verify）：parity gate（`qwen4_exp_verify_parity.py`：DET 旗標、seed、64 條 8–14k prompt 的 greedy decode，batch 兩次＋solo 24 條；floor＝同一台的 batch 對 solo）全部在 floor 內，solo 對 solo 的 output 0（extend）逐位相同；−1.42 ms／cycle（QSA 約 5.4 → 3.9 ms：`_compact_kv`＋fmha 換成 `_qsa_direct_decode`＋combine）。
  - `ARC_GDN_FOLD_IN_VERIFY`（h20）放寬到預設 radix cache（`--mamba-radix-cache-strategy extra_buffer` 要明給；no_buffer、lazy、mixed chunk、streaming session、PD、hicache、int8 checkpoint、unified memory 拒絕；HiSparse 臂照舊、含它的 session radix cache）：
    - 找到並修掉一個 race：overlap scheduler 下，新請求在 `HybridReqToTokenPool.alloc` 把 row 的 spec cursor 歸零是在 schedule stream 上，而 DFlash 的 WAR barrier 只等 target verify 的 metadata prep，早於 advance；同一 row 的前主人若還在 in-flight batch，它的 advance 會在歸零之後把 `(base, n_acc, 1)` 寫回。一般模式的 finish kernel 會清掉，fold 模式沒有，新請求的第一個 verify 就把前主人的 ring 折進自己的 GDN state。修法：fresh row 在 forward stream 上、extend 之前再歸零一次（`replay_cursor_reset_rows`，和 `mamba_clear_indices` 同一條路）。負對照（`ARC_GDN_FOLD_IN_VERIFY=check`＋`ARC_GDN_FOLD_IN_VERIFY_SKIP_FWD_RESET=1`，10 分鐘）：3,034 個請求裡 88 列（約 2.9%）；修正打開 0／12,000+ commit。
    - QSA HiSparse 不受影響：`QSAHiSparseCoordinator.request_finished` 在釋放 row 前讓 schedule stream 等 forward stream。B200 上 hs110 形狀（reap320 NVFP4、acc-v4 DFlash K8、TP1、HiSparse＋session，修正關掉）同樣的負對照 986 個請求 0 列。native MTP 也不受影響（WAR barrier 等的是 advance 之後的 draft extend）。
    - `check`：每個 commit 斷言 row 上待折的視窗是現在這個請求自己留下的（owner 表；判決晚一個 commit 經 pinned copy 讀，不加 stream sync）。
    - 數值：B200 上不是逐位（h20 在 G4 fp16 是逐位）：solo 對 solo 在輸出 23% 處第一次分歧、|Δlp| p50 2.7e-5，比 batch 對 solo 的 floor 小；判讀為 ULP 級。
    - 吞吐：−0.45 ms／cycle（中位）。post 5.2 → 2.8 ms，但 verify 的 GDN kernel 1.54 → 2.76 ms，且 commit kernel 還有 0.87 ms：extra_buffer 下 `mamba_track_indices` 永遠是張量，commit 每個 cycle 都對全部 row 開滿 grid（約 17 萬個 CTA，幾乎都馬上 return）。只對 track row 開 grid 估計再 −0.8 ms（未做）。
  - 三項合併（換卡對：對照在 GPU 2,3、候選在 0,1，排除卡對效應）：−2.14 ms／cycle（中位；bs 60 −0.9、bs 125 −2.7，bs 104 37.9 → 35.8 ms，−5.6%），約等於三項各自之和；accept 3.152 對 3.144。帳本 logprob 均值四輪對照 −0.2554～−0.2599、候選 −0.2565～−0.2621，差都在 1σ（cluster SE 0.0021–0.0024）上下、方向不一致，判讀為噪音。
  - 候選（只評估）：HC 在 verify 尺寸的融合（combine 用 `hc_gate` 預算 gate＋串流 apply，逐位不變，約 −1.0～1.6 ms／verify；row-tiled 的 down／up GEMM 融合要原生 CuTe，改數值）；drafter propose（ring attention 在 B200 重調 tile 約 −0.8 ms、window commit 合併 launch、sampler 鏈融合、top-16 兩段式、TP2 的 `ARC_DFLASH_ATTN_TP=1`、`ARC_DFLASH_GRAPH_SIZES` 補 104／112／120），合計約 −2.3～2.8 ms。
