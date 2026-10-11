# sglang-arc-2：arc-forge 的 sglang 私有 fork

上游 [sgl-project/sglang](https://github.com/sgl-project/sglang) `v0.5.20`（`94602c9c`）加上 arc-forge 的修改。
每個 commit 就是一個改動。早期的改動是從 arc-forge `patches/` 一個個重建進來的，commit 訊息會寫明來源 patch 的檔名和 sha256。
之後所有新的 sglang 修改都直接 commit 在這裡；arc-forge 的 `patches/sglang-*` 已經凍結。

部署目標指向哪個 commit，由 arc-forge 的 `deploy/sglang.lock.json` 決定，維護工具在 arc-forge 的 `work/sglang_fork/`。遠端機器（Colab、Runpod、Kaggle）不 clone 這個私有 repo，而是用 `export_tree.py` 匯出的 tarball。

## 分支

| 分支 | 內容 |
|---|---|
| `main` | 上游 `v0.5.20` |
| `arc/main` | 所有平台共用的主線：0001–0032、HiSparse h01–h39（不含 h37）、h40–h43、DFlash（`speculative/arc_dflash/`） |
| `deploy/g4-hs55-v8` | AGI-3 Kaggle 正式版（RTX PRO 6000）：v6 加上 h40–h43，tree `3dcbde80` |
| `deploy/g4-hs55-v6` | 0001–0032 加上 h01–h09、h12–h22，tree `e93d999a` |
| `deploy/g4-dflash` | v6 加上 d01、d02（舊 G4 DFlash，drafter 程式當時放在 arc-forge） |
| `deploy/l4-h28` | ARC-AGI-2 Kaggle v13／v14（L4×4），tree `b9cbd753`（展開慣例） |
| `deploy/l4-h30` | ARC-AGI-2 Kaggle v15，NoMTP 快速樹 `df5847fd` |
| `deploy/l4-r9r2` | h28 加上 h29–h36、h38、h39（L4 native MTP），tree `1742410a` |
| `deploy/l4-dflash` | r9-r2 加上 l01（舊 L4 DFlash） |
| `exp/h37-swap-in-padding-counts` | r9-r2 加上 h37（r9-r2 刻意排除，從未部署） |
| `exp/kaggle-*-optin` | h28 加上三個互斥的 L4 通訊實驗（FlashInfer PCIe TP4、attention TP2 custom AR、FlashInfer attention TP2） |
| `legacy/sglang-0.5.17` | 上游 v0.5.17 加 4 個 patch |
| `legacy/sglang-97c6978` | 上游 `97c6978` 加 13 個 patch |
| `legacy/sglang-qwen4-pr36497` | PR #36497 快照 `73a25520` 加 0001–0035（B200 venv-qwen4） |
| `legacy/sglang-qwen4-pr36497-sm89` | 同一個基底加 0001–0019、0036–0043（舊 Kaggle L4 runtime） |
| `legacy/sglang-qwen4-mtp3-73a2552` | pr36497 加 MTP3 native GR 的 0001–0006 |

`deploy/*` 的 digest 都和部署當時的紀錄逐位相同。`exp/*` 和 `legacy/*` 沒有留下可比對的 digest 紀錄，只驗證過能完整、無衝突地套上，而且可以重現（tree 相同）。

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
- **PLE 表格式**（從 main 移植到 L4 樹）：
  - `ARC_PLE_NVFP4=1`（opt-in，預設 0；只支援 `--ple-offload-backend pinned`＋FP8 PLE 表）：PLE 表在載入時逐 shard 於 GPU 量化成 NVFP4（e2m1＋每 16 個一個 e4m3 scale，沒有 global scale：來源已是 e4m3），每列 90 B 連續存放，host 47.7 → 26.8 GiB。gather 解碼成 bf16，相對 FP8 列 relRMSE 9.6%、cosine 0.9955；4090 上 gather 比 FP8 快 15%。所有列沒載滿就報錯。`models/arc_ple_nvfp4.py`。
  - `ARC_PLE_INT5=1`（opt-in，預設 0；與 `ARC_PLE_NVFP4` 互斥，同時開起服 raise；同樣只支援 `--ple-offload-backend pinned`＋FP8 PLE 表、寬度須為 16 的倍數）：同一套載入時 GPU 量化，改存對稱 INT5（每 16 個一組、每組一個 e4m3 scale＝amax/15，碼 −15..15 以 code+16 存 5 bit，8 碼 5 byte 小端連續），一列 d·5/8 碼 byte 接 d/16 scale byte 連續存放（d=160 為 110 B），host 47.7 → 32.8 GiB。gather 解碼成 bf16（精確）。EAN320 layer 1 shard_7 前 40 萬列：相對 FP8 列 relRMSE 4.17%（NVFP4 9.59%）、cosine 0.99915；B200 pinned 20M 列隨機 gather 15.8k／131k／524k 列：FP8 150／1,158／4,638 us、NVFP4 117／890／3,536、INT5 128／996／4,074。`models/arc_ple_int5.py`，測試 `test/registered/kernel/embeddings/test_arc_ple_int5_codec.py`＋`test_qwen4_ple_offload.py`。
- **多模態（從 main 移植，全部 opt-in）**：
  - `ARC_MM_RELEASE_AFTER_PREFILL=1`（opt-in，預設 0）：請求的 prefill 一完成就釋放多模態特徵（CPU 上的 fp32 pixels）。codex 每回合重送整段歷史的截圖，G4 c110 下 scheduler host RAM 約 9.6 → 1.3 GiB/h。被 retract 的請求先停在 scheduler、由 tokenizer 重處理缺特徵的 item 後再排回（見下面 retract 補特徵）。
  - `ARC_MM_LAZY_MEDIA=1`（opt-in，預設 0；Qwen3-VL 家族、單一 tokenizer worker）：看過的圖片／影片不再重做前處理、不送 pixel。前端以內容（h41 artifact key）記住每份媒體被 processor 展開成的 token 區塊、pad 段與 grid；請求只對沒看過的媒體跑 processor（一次呼叫、只含它們的 placeholder），其他媒體把快取區塊拼進 tokenize 結果，送 metadata-only 的 lazy item。scheduler 排 prefill 時若 lazy item 落在 extend 裡（前綴被逐出、或前綴停在它之前），就在配置任何東西前把請求退回 tokenizer（`ArcMmRematerializeReq`，帶這次 extend 內所有 item 的 hash），tokenizer 把它們補上 pixel 後重送；ids／hash 與帶不帶 pixel 無關，所以前綴照樣命中。影片（每組幀一段 offset）改走逐 item 的 embedding 路徑，只編碼與 extend 重疊的影片。被 retract 的請求補特徵後再排回（見下面 retract 補特徵）。`ARC_MM_LAZY_MEDIA_MIN_SEEN`（預設 1）：被幾個先前請求帶過才改送 lazy；`ARC_MM_LAZY_MEDIA_ENTRIES`（預設 16384）。4090 Qwen3.5-4B、3 影片＋3 圖 17 回合：prompt ids 與 greedy 輸出逐回合相同（含中途 flush），前端 CPU 54.2 → 7.9 s。`multimodal/arc_lazy_media.py`。
  - retract 補特徵（`65b0d354d`）：特徵已釋放（`ARC_MM_RELEASE_AFTER_PREFILL=1`）或是 lazy item 的請求被 retract 時，不再 raise（以前整台 server 會掛），改停在 `Scheduler.arc_mm_refill_parked`（不佔資源、保留已輸出的 token），送 `ArcMmRematerializeReq(refill=True)` 給 tokenizer；tokenizer 帶 pixel 重處理後回送 `arc_mm_refill=True` 的請求，scheduler 只拿缺特徵 item 的 feature（hash、modality、pad value、span 長度都要相同，否則以錯誤 abort 該請求），再以 retracted 身分排回。停靠中被 client abort 會照 queue 中請求的方式清掉。G4 hs110 `SGLANG_TEST_RETRACT` 20 分：17 次 retract 全部補回、0 Traceback。
