# MiniMax H3 Production Director

> **这是个人二开分支，不是官方版本。**
>
> 上游原仓库：**[AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)**
>
> **如果你只是想正常使用这个插件，请直接去上游安装。** 上游更新更勤、社区支持更好、功能更完整。
>
> 这个分支只为一类机器服务：**20GB 显存的 Ampere 卡（RTX 3080 20GB / 3090）+ Windows**。
> 改动带有很强的针对性，对别的配置不一定更好，也可能更差。

---

最新更新：**2026-09-11 — 二采改画幅引导修复、接缝光色对齐、参考图长边档位**。
查看[完整更新说明与使用方法](CHANGELOG.md)。

## 20GB Ampere 面对的是什么

本分支已合入上游截至 `ed3cdb0`（2026-09-11 检查）的更新：执行期音频缓存与按需解码、
一采缓存补齐、R2V 原声、SolAttn v5 布局兼容、混合时间轴、媒体缩略图、导演包导入导出。
保留本分支的 `balanced_20gb`、文本编码器释放、延迟 VAE 解码和内存诊断。

Refine 新增 latent 放大的时间分块、二采的空间分块，均默认关闭，按需开启；
分块可能改变生成效果和耗时，暂未实测显存节省幅度。实时采样预览改为循环 WebP，
不再发送成片整段 JPEG 回放；完整成片请从下游 CreateVideo / SaveVideo 查看。

开启「段间引导」后可选择「引导＋重绘」，重绘幅度为 0–0.95，新默认值为 0.1；
0 表示接缝硬锁定，越大越允许接缝重绘。旧工作流已保存的数值保持不变。
二采会按目标画幅重建前缀约束；一采优先使用匹配画幅的上一段一采 latent，
二采优先衔接匹配画幅的上一段二采尾部，避免尺寸不匹配和不必要的像素回编。
R2V 参考图可选长边 1024／1280／1536，介于原有 match 与 max 之间，仅缩小不放大。
自定义帧率现在参与视频段时长到帧数的换算，前后端统一使用 MiniMax 的 17k+5 对齐规则。
参考视频条件节点已改为关键字调用，兼容官方参数顺序调整。

「段间引导」新增默认开启的「保完整」：保留帧数对齐多生成的尾部画面及音频，
避免强制裁回设定时长时吞掉句尾。实际成片可能略长；需要严格时长时可关闭，
但会恢复尾部裁切。切换该选项会使相关旧缓存失效，建议重新运行全部段落；
已导出文件不会自动恢复被裁掉的音频。

性能区新增 **低内存分段导出**（API：`low_memory_segment_export`），默认关闭。
仅在「分段导出」且关闭「一采确认」时生效：上一段完成衔接、所需成片和一采 MP4
均成功落盘后，旧段仅保留内部封面，并从 `images` / `images_pre_refine` 输出中省略，
避免下游 Save Video 保存静帧视频。音频和原片同步选择保留段，`frame_count` 对应实际输出帧数。
最后一段保留完整帧。
导出或衔接裁剪后的重写失败时保留该段完整画面。下游需要完整帧时请保持关闭；
分段模式不再额外拼接整条时间轴。

工具栏支持 `*.mmxpack.zip` 导演包，包含时间轴及素材，不包含模型；导入会替换当前
节点时间轴。格式与目录说明见 [英文文档](README_EN.md#director-pack-script--media)。


原插件在 24GB 以上、Ada/Blackwell 架构上跑得很顺。**RTX 30 系 + 20GB 会同时撞上两堵墙**，
而这两堵墙都不是「调参数」能绕过去的。

### 墙一：显存装不下，Windows 会偷偷用内存顶上

H3 的一次生成要同时用到：

| 组件 | 规模 |
|---|---|
| Qwen3-VL 文本编码器 | ~15 GB |
| H3 扩散模型 | ~20 GB |
| Video VAE | ~5 GB |
| Audio VAE | ~0.6 GB |

**加起来 40GB，卡只有 20GB。**

装不下的部分会被 WDDM 挪进 **Shared GPU Memory**——那本质上是系统内存，走 PCIe 访问。
生成还能跑完，但**整台机器卡顿**。

更麻烦的是**你看不见它**：NVML 只报专用显存，`nvidia-smi` 看着一切正常。
本分支实测过一次跑完之后，64GB 内存只剩 3.1GB，共享显存吃掉了 24.7GB。

### 墙二：sm_86 没有 FP8 / FP4 硬件

| 权重格式 | sm_86（30 系） | 说明 |
|---|---|---|
| int8 / int4-convrot | **原生** | H3 的 convrot kernel 直接打张量核 |
| w4a8_int8 | **原生** | |
| bfloat16 | **原生** | |
| float8_e4m3fn / e5m2 | **模拟** | 需要 SM89+（40 系起） |
| nvfp4 / mxfp8 | **模拟** | 需要 SM100+ |

**模拟路径 = 在软件里反量化。** 一个 fp8 量化的权重虽然文件更小，
在 30 系上可能比更大的 int8 版本还慢。

这条直接决定了**你该选哪个 checkpoint**，而 ComfyUI 要等模型加载完才会打印
`Native ops / emulated ops`——那时候已经晚了。本分支在启动时就告诉你。

### 顺带：稀疏注意力这条路在 30 系是关着的

社区的 Sol-Attn 系列（sparse attention）需要 SM89+ 的 TMA 或专用 pointer kernel。
**sm_86 装了只会回退到 dense，白折腾。** 本分支不在这上面花力气，
诊断里也只在真能用的卡上才提这件事。

**30 系唯一确实有效的注意力加速是 SageAttention** —— 实测一采 −34.5%。

---

## 实测结果

固定场景：单段 R2V，124 帧 @24fps，一采 352×608，二采 768×1376，20 步，固定 seed，**冷启动**。

| | 起点 | 现在 | 变化 |
|---|---|---|---|
| **总时长** | 373.1s | **319.6s** | **−14.3%** |
| 一采 s/it | 6.84 | **4.48** | −34.5% |
| 二采 s/it | 43.39 | 41.64 | −4.0% |
| 一采时 GPUShared | 15,380 MiB | **< 1,000 MiB** | −93% |

**画质：人工同 seed 比对，无可见差异。**

### 收益从哪来

**1. SageAttention 之前根本没开。**
工作流里挂着 `Patch Sage Attention KJ`，但 `sage_attention = disabled`——节点在那儿什么也没做。
改成 `auto` 后一采从 6.84 降到 4.48 s/it。

这不是本分支的代码功劳，是**诊断查出来的配置问题**。但它恰好说明了为什么这个分支
要先做诊断再做优化：**跑了几个月的工作流里，最大的一块提速一直躺在一个下拉框里。**

**2. 文本编码器在一采期间白占 15GB 显存。**
conditioning 算完之后 Qwen3-VL 就没用了，但它一直留到运行结束。
一采要 stage 20GB 的扩散模型——20GB 的卡装不下 35GB，于是 15GB 进了共享显存。

本分支在 conditioning 完成后**定向释放文本编码器**（只放它，VAE 和扩散模型不动）。
conditioning 张量此时已经算完，所以**不改变任何生成数值**。

**3. 一采该用 int8_convrot，不是 W4A8。**
实测 `ref2va_pruned_int8_convrot` 比 `ref2va_pruned_w4a8_mixed` **快 21%**
（4.48 vs 5.69 s/it），而且画质更好——这正是「墙二」的直接后果。

**W4A8 当一采模型是双输。** 当二采模型仍然合理：低 denoise 精修，且省显存。

---

## 推荐配置（20GB Ampere）

| 项 | 值 | 理由 |
|---|---|---|
| 一采 UNET | `minimax_h3_ref2va_pruned_int8_convrot` | 原生 kernel，最快也最好 |
| 二采 UNET | `minimax_h3_ref2va_pruned_w4a8_mixed` | 低 denoise 精修，省显存 |
| Patch Sage Attention KJ | **`auto`**（一采、二采都要） | −34.5%，无画质代价 |
| `allow_compile` | `false` | 除非你同时用 TorchCompile，否则无意义 |
| `memory_strategy` | `balanced_20gb` | 释放 TE，共享显存降 93% |
| `memory_debug` | 平时 `false` | 排查时才开 |

> 两个 Sage 节点**都要改**。只改主路径的话二采吃不到，日志里 `Using sage attention mode: auto`
> 会只出现一次而不是两次。

---

## 相对上游的改动

全部是**加法**——新增文件 + 少量调用点，尽量不重写上游文件，方便持续合并上游更新。

| 文件 | 类型 | 说明 |
|---|---|---|
| `director/env_diagnostics.py` | **新增** | 环境体检（只读） |
| `director/memory_policy.py` | **新增** | `balanced_20gb` 内存策略 |
| `director/memory_debug.py` | **新增** | RAM / VRAM / 共享显存 / 耗时诊断 |
| `benchmarks/` | **新增** | 实测记录与跑测协议 |
| `nodes/director_common.py` | 改动 | 加 `memory_strategy`、`memory_debug` 两个控件 |
| `director/executor_core.py` | 改动 | 诊断探针 + TE 释放调用点 |
| `director/refine_sampling.py` | 改动 | 二采前一个探针 |

**没有改动的部分**：段间引导逻辑、采样数学、conditioning、导出、前端 UI、所有任务模式。
**上游的功能一个没少**，并持续合并上游更新。

### 新增控件

节点「性能」组里多了两个。

**`memory_strategy`**

| 值 | 行为 |
|---|---|
| `standard` | **与上游完全一致**，不做任何额外释放 |
| `balanced_20gb` | conditioning 后释放 TE、deferred pre-refine 解码、RAM 感知清理 |
| `aggressive_lowmem` | 预留位，当前等同 `standard` |

默认 `standard`——**上游行为是默认行为**，新策略必须显式开启，出问题随时切回来复现。

**`memory_debug`**（默认关）

开启后在控制台和节点 `report` 输出诊断。关闭时所有探针直接 return，零开销。

---

## 诊断能看到什么

### 启动时的环境体检

插件加载时自动输出一次。**只读**——不加载模型、不改 ComfyUI 状态、不写任何文件。

```
=== MiniMax H3 Director — Environment Report ===

--- Hardware / torch ---
GPU            : NVIDIA GeForce RTX 3080 (sm_86)
VRAM           : 20480 MiB
Driver         : 591.86
torch          : 2.12.1+cu130 / CUDA 13.0

--- comfy_kitchen (H3 int8-convrot kernels) ---
backend        : enabled
VERDICT        : OK — H3 quantised CUDA kernels are live

--- Weight formats this GPU runs in hardware ---
native         : int8, int4 / convrot_w4a4, w4a8_int8, bfloat16
EMULATED       : float8_e4m3fn / float8_e5m2 (needs SM89+)
                 nvfp4 / mxfp8 (needs SM100+)
note           : sm_86 有 INT8 张量核但没有 fp8/fp4 硬件，优先选 int8-convrot 权重

--- Attention ---
backend        : attention_pytorch
SageAttention  : available

--- H3 PackedLayout ownership (continuity) ---
owner          : stock

--- Core regression scan ---
! L168: v = v.clone()  <- full value-tensor clone in the H3 attention path

--- ComfyUI launch ---
vram mode      : NORMAL_VRAM
pinned memory  : enabled
system RAM     : 57153 / 65393 MiB available
```

它专门在查这几个坑：

- **`comfy_kitchen` CUDA 后端是否被静默禁用。** ComfyUI 在 `comfy/quant_ops.py` 里按
  `torch.version.cuda >= 13` 卡这个后端。不满足就只打一行 warning 然后走模拟路径——
  H3 的 int8-convrot 专用 kernel 全部失效，[实测慢 2.17 倍](https://note.com/tank_ai/n/nab26edd4ab96)。
  **ComfyUI 便携版自带的 PyTorch 不会随核心更新**，很容易长期处于这个状态而不自知。
- **这张卡哪些权重格式是原生的。** 见上面「墙二」。
- **`PackedLayout` 被谁占了。** 段间引导会 patch 它，遇到不认识的第三方 wrapper 会直接报错。
- **core 里的已知高显存写法。**

### 运行时探针（`memory_debug = true`）

```
[Baseline] COLD start — timings are comparable to other cold runs.
[Memory]   Run Start | RSS=... | AvailRAM=... | NVMLUsed=... | GPUShared=... | Δt=...
[Model]    First Sampling (pre) | MiniMaxH3Model | tensors=... | stored=...GiB
[Resident] First Sampling (pre) | MiniMaxH3TEModel_ 14980MiB @cuda:0; ...
[Tokens]   Before Refine Sampling | latent=(...) | video_tokens=... | ffn_intermediate~...MiB
```

- **`GPUShared`** —— Windows「GPU Process Memory」性能计数器。
  **NVML 看不到 WDDM 溢出，这个能看到。** 这是整套诊断里最关键的一个数。
- **`AllocRetries` / `CudaOOMs`** —— torch 分配器压力计数，溢出的进程内指纹
- **`[Resident]`** —— 每个关键时刻显存里到底住着什么
- **`[Model]`** —— 每一遍采样实际用的是哪个 checkpoint（ComfyUI 的日志不说）
- **`[Baseline]`** —— 自动判定冷/热启动。**热启动的计时不可与冷启动比较**，会明确警告

最后一条是踩出来的：热启动因为权重还在内存里，能虚快 12%，
拿它跟冷启动比会得出完全相反的结论。

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Slash214/ComfyUI-MinMax-H3-Production-Director.git
pip install -r ComfyUI-MinMax-H3-Production-Director/requirements.txt
```

依赖、模型、工作流、各任务模式的用法与上游完全一致，
**请参照[上游 README](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)**。
本仓库不重复维护那部分文档，以免与上游脱节。`example_workflows/` 下的示例同样来自上游。

### 本机验证环境

```
GPU        : RTX 3080 20GB (sm_86)
RAM        : 64GB
OS         : Windows
PyTorch    : 2.12.1 + cu130
ComfyUI    : 0.33.3
VRAM mode  : NORMAL_VRAM
DynamicVRAM: enabled
```

**其他配置未经测试。** 24GB 以上、或 40/50 系的卡大概率用不上这些改动——
你没有「墙一」那么紧的显存，也没有「墙二」的量化限制。

---

## 已知限制与未完成

- **`balanced_20gb` 在多段场景下的净收益尚未验证。** 每段都要重新 stage 一次 TE（约 15GB），
  单段是纯赚，多段可能不划算。**做长视频/多镜头前请自行对比 `standard`。**
- `aggressive_lowmem` 是空档，行为等同 `standard`。
- 尚未实施：跳过 pre-refine 解码（约省 13s）、`--disable-pinned-memory` 对照、
  FFN 分块降峰值显存。
- 二采仍是全流程最贵的一段（约 40%）。实测它**不是 attention 瓶颈**——
  一采从 sage 拿到 −34.5%，二采只有 −4%，尽管二采 token 数是一采的 4.94 倍。
  瓶颈在 FFN 与权重在 PCIe 上的搬运，方向还在找。

---

## 致谢

本分支的**全部核心功能都来自上游**，作者是 **[AI搅拌手 / AIMixer](https://github.com/AIMixer)**。
时间轴、段间引导、二采/放大、多任务模式、前端 UI、一采确认——这些都是上游的工作。
本分支只是在外围加了内存调度与诊断。

- **[AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)** — 上游原仓库
- [Comfy-Org / ComfyUI](https://github.com/Comfy-Org/ComfyUI) — 官方 MiniMax H3 支持
- [MiniMax-AI](https://github.com/MiniMax-AI) — MiniMax H3 模型
- [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) — 段间运动续拍思路
- [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) — H3 3D latent 放大
- [Comfyit 搅拌站](https://comfyit.cn/) — 模型、工作流与教程配套

**遇到问题请先确认是不是本分支引入的**：把 `memory_strategy` 切回 `standard` 复现一次。
如果 `standard` 下同样存在，那是上游的问题，请去上游反馈——
不要占用上游维护者处理本分支的时间。

## 许可证

Apache-2.0，与上游一致。
