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
