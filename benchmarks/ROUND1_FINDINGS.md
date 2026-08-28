# 第一轮实测结果 · 2026-08-25

机器：RTX 3080 20GB / 64GB RAM / Windows / torch 2.12.1+cu130 / ComfyUI 0.33.3
场景：单段 R2V，124f @24fps，一采 352×608，二采 768×1376，`memory_strategy=balanced_20gb`

## 时间分解（总 373s）

| 阶段 | 耗时 | 每步 |
|---|---|---|
| Conditioning + 模型 staging | 14.7s | |
| 一采 20 步 | 145.0s | 6.84 s/it |
| H3 Latent Upscale | 5.8s | |
| 二采 3 步 | 139.1s | **43.39 s/it** |
| Pre-refine VAE Decode | 13.1s | |
| Final VAE Decode | 49.2s | |

latent `22x38 → 48x86`，token 数 **4.94×**，每步耗时 **6.34×**。超线性，但远不到 attention 二次方该有的 24×——**二采的瓶颈是 FFN 与权重搬运，不是 attention。**

## 结论 1：`comfy_kitchen` 正常，基线可信

```
Found comfy_kitchen backend cuda: {'available': True, 'disabled': False, ...}
Native ops: convrot_w4a4, asym_w4a8_int8, int8_tensorwise
emulated ops: nvfp4, mxfp8, float8_e4m3fn, float8_e5m2
```

H3 的 int8-convrot 走原生 kernel。不存在"降级路径"问题。

## 结论 2：`AllocRetries` 全程为 0

torch 缓存分配器从未重试。压力**不在 torch 层**，而在 DynamicVRAM / comfy-aimdo 层。
→ 优化 torch 侧显存无意义，要减的是**同时常驻的模型总量**。

## 结论 3：假设"TE 挤占二采"—— 证伪

```
[Resident] Before Refine Sampling | MiniMaxH3 12801MiB; MiniMaxH3TEModel_ 4MiB; VideoVAE 0MiB
NVMLFree = 5620 MiB
```

二采起步时 TE 已被清掉，专用显存还有 5.6GB 空余。二采不缺显存。

## 结论 4：真正的病根 —— 整机 RAM 被抽干

| 时刻 | AvailRAM | GPUShared |
|---|---|---|
| Run Start | 48,699 MiB | 74 MiB |
| First Sampling (pre) | 31,444 MiB | **15,380 MiB** |
| After First Sampling | 8,555 MiB | **24,719 MiB** |
| After First VAE Decode | **3,134 MiB** | 25,658 MiB |

64GB 内存跑到只剩 3.1GB，Windows 全面换页 —— 这就是"卡"。

两个大户：GPUShared **24.7GB**（DynamicVRAM 溢出到系统内存）+ Pinned memory **26GB**（启动日志 `Enabled pinned memory 26157.0`）= 约 50GB。

**GPUShared 在第一次采样就冲到 15GB，不是二采才开始。** 二采感觉更卡，是因为那时 RAM 已被前面耗到只剩 8.5GB。

## 结论 5：15GB 的 Text Encoder 是溢出的起点

```
[Resident] First Sampling (pre) | MiniMaxH3TEModel_ 14980MiB @cuda:0; MiniMaxH3VideoVAE 512MiB @cuda:0
NVMLUsed = 18987 / 20480，GPUShared = 15380 MiB
```

一采开始时 TE 还整个压在显存里，紧接着要 stage 19995MB 的主模型。20GB 的卡装不下 35GB，于是 15GB 进了共享显存。

**而 conditioning 此时已经算完，TE 是纯粹的死重。**

单段路径下 `unload_models = seg_total > 1` = False，`balanced_20gb` 在 RAM 充足时也走 `unload_models=False`，所以它从未被释放。

## 结论 6：core 里确实有 `v = v.clone()`

```
! L168: v = v.clone()   comfy/ldm/minimax/model.py
```

spec 第 16 节的猜测成立。暂不动用户 core，仅记录。

---

# 实测 B：开启 SageAttention（已确认，采纳）

图里的 `Patch Sage Attention KJ` 长期是 `sage_attention = disabled`，等于挂着不干活。
改成 `auto`，其余一律不动（同 seed 666、同 20 步、同 `balanced_20gb`）：

| 指标 | disabled | **auto** | 变化 |
|---|---|---|---|
| 总时长 | 373.1s | **303.3s** | **−69.8s / −18.7%** |
| 一采 20 步 | 145.0s | 101.4s | −30.0% |
| 一采 s/it | 6.84 | **4.82** | −29.5% |
| 二采 3 步 | 139.1s | 116.5s | −16.3% |
| 二采 s/it | 43.39 | **37.25** | −14.1% |

日志确认：`Using sage attention mode: auto`。
**画质人工比对（同 seed）：无可见差异 → 采纳，设为新基线。**

## 实验缺陷（记录在案）

两次跑起始状态不同（B 接着上一次跑，未重启）：

| | 基线 | B |
|---|---|---|
| Run Start AvailRAM | 48,699 MiB | 10,411 MiB |
| Run Start GPUShared | 74 MiB | 22,518 MiB |

B 在更差的起跑线上仍快 19%，所以提升可信、甚至被低估。
但 **Final VAE 的 −17% 不算数**——VAE 解码不走 H3 attention patch，那是起始状态噪声。
往后跑对照**先重启 ComfyUI**。

## 由此得到的结构性结论

一采受益 −29.5%，二采只有 −14.1%。二采 token 数是一采的 4.94 倍，attention 占比更高，
按理应该受益更多，结果反而更少。

→ **二采剩下的 116s 主要不是 attention，是 FFN 与权重在 PCIe 上的搬运。**
往后优化二采要往这个方向走，不要再在 attention 上使劲。

## Sage 没解决的部分

```
After Final VAE Decode | AvailRAM=2167.7MiB | GPUShared=25380.1MiB
```

比基线的 3,134 MiB 还低。**Sage 省算力，不省内存。**"卡"的根因原封不动。

---

# 第二轮改动

## 新增：conditioning 后定向释放 Text Encoder

`memory_policy.release_text_encoder()`，在 conditioning 完成、一采开始之前调用。

- **只放 TE**，VAE 和 diffusion model 一律不动
- 不改变任何生成数值（conditioning 张量已在手）
- **仅 `balanced_20gb` 生效**，`standard` 行为与之前逐字节一致

预期：一采时 `GPUShared` 应显著低于 15,380 MiB，`AvailRAM` 曲线整体抬高。

代价：多段时每段要重新 stage 一次 TE（约 15GB）。单段无代价。**多段是否划算需要单独测。**

## 诊断修正

- `comfy_kitchen` 后端状态改为读 backend dict（上一轮显示 unknown）
- driver 版本加 `nvml.dll` ctypes 回退
- vram mode 改读 `model_management.vram_state`（不是 CLI flag）
- `[Tokens]` 行上一轮没打出来：token 估算改为按 rank 通用推导，不再只认 5 维
- 环境报告新增系统 RAM 余量与 pinned memory 提示

---

# 第二轮怎么跑

**同样的工作流、同样的 seed、同样的参数**，跑两次：

1. `memory_strategy = balanced_20gb`（带新的 TE 释放）
2. `memory_strategy = standard`（对照）

两次都开 `memory_debug`，日志发回。

## 重点看三个数

1. **`[Memory] First Sampling (pre)` 的 `GPUShared`** —— 是否从 15,380 MiB 掉下来
2. **`Run End` 的 `AvailRAM`** —— 是否高于 3,995 MiB
3. **`First Sampler End` 的 elapsed** —— 145s 有没有变化，以及主观上机器还卡不卡

## 顺带值得一试（独立变量，先别和上面混）

启动加 `--disable-pinned-memory` 再跑一次。26GB 的 pinned 预留在 64GB 机器上、配合 24.7GB 共享显存，很可能是净负担。这是 ComfyUI 启动参数，不是插件行为——插件只做建议，不自行修改。
