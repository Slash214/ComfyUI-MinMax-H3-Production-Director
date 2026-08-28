# MiniMax H3 Production Director

> **这是个人二开分支，不是官方版本。**
>
> 上游原仓库：**[AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)**
>
> **如果你只是想正常使用这个插件，请直接去上游安装。** 上游更新更勤、社区支持更好、功能更完整。
>
> 这个分支是为了**一台特定机器**（RTX 3080 20GB + 64GB RAM + Windows）做的显存与内存调优，
> 加了一套诊断工具，改动带有很强的个人取向。它对你不一定更好，也可能更差。

---

## 这个分支在做什么

原插件在 24GB 以上的卡上跑得很顺。但在 **20GB 显存的 RTX 3080** 上会遇到一个 Windows 特有的问题：

显存装不下的部分会被 WDDM 挪进 **Shared GPU Memory**——那本质上是系统内存，走 PCIe 访问。
结果是生成还能跑完，但**整台机器卡顿**，而且 NVML 看不到这件事（它只报专用显存）。

这个分支干两件事：

1. **测出来** —— 加一套能看见 WDDM 溢出、内存曲线、常驻模型、环境降级的诊断
2. **减下去** —— 在不动任何生成数值的前提下，减少同时常驻的模型总量

**核心原则：不以画质换速度。** 所有改动要么是纯内存调度，要么是经人工比对确认无可见差异的。

---

## 实测结果

固定场景：单段 R2V，124 帧 @24fps，一采 352×608，二采 768×1376，20 步，固定 seed，冷启动。

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
改成 `auto` 后一采从 6.84 降到 4.48 s/it。这不是本分支的代码功劳，是**诊断查出来的配置问题**，
但它说明了为什么"先测量再优化"值得。

**2. Text Encoder 在一采期间白占 15GB 显存。**
Qwen3-VL 文本编码器约 14,980 MiB，conditioning 算完后就没用了，但一直留到运行结束。
一采要 stage 约 20GB 的扩散模型——20GB 的卡装不下 35GB，于是 15GB 进了共享显存。

本分支在 conditioning 完成后**定向释放文本编码器**（只放它，VAE 和扩散模型不动）。
conditioning 张量此时已经算完，所以**不改变任何生成数值**。

**3. 一采模型该用 int8_convrot，不是 W4A8。**
实测 `ref2va_pruned_int8_convrot` 比 `ref2va_pruned_w4a8_mixed` **快 21%**
（4.48 vs 5.69 s/it），而且画质更好。W4A8 走 `asym_w4a8_int8` kernel，
int8_convrot 走原生 `convrot_w4a4`，两条路效率不同。
**W4A8 当一采模型是双输**；当二采模型仍然合理（低 denoise 精修 + 省显存）。

---

## 相对上游的改动

全部是**加法**——新增文件 + 少量调用点，尽量不重写上游文件，方便持续合并上游更新。

| 文件 | 类型 | 说明 |
|---|---|---|
| `director/env_diagnostics.py` | **新增** | 环境体检（只读） |
| `director/memory_policy.py` | **新增** | `balanced_20gb` 内存策略 |
| `director/memory_debug.py` | **新增** | RAM/VRAM/共享显存/耗时诊断 |
| `benchmarks/` | **新增** | 实测记录与跑测协议 |
| `nodes/director_common.py` | 改动 | 加 `memory_strategy`、`memory_debug` 两个控件 |
| `director/executor_core.py` | 改动 | 加若干诊断探针 + TE 释放调用点 |
| `director/refine_sampling.py` | 改动 | 二采前加一个探针 |
| `director/h3_context_patches.py` | 改动 | 移植上游 `78c9b2bc`（段间引导兼容 SolAttn） |

**没有改动的部分**：段间引导逻辑、采样数学、conditioning、导出、前端 UI、所有任务模式。
上游的功能一个没少。

### 新增控件

节点「性能」组里多了两个：

**`memory_strategy`**

| 值 | 行为 |
|---|---|
| `standard` | **与上游完全一致**，不做任何额外释放 |
| `balanced_20gb` | conditioning 后释放 TE、deferred pre-refine 解码、RAM 感知清理 |
| `aggressive_lowmem` | 预留位，当前等同 `standard` |

默认 `standard`——**上游行为是默认行为**，新策略必须显式开启。

**`memory_debug`**（默认关）

开启后在控制台和节点 `report` 输出诊断。关闭时所有探针直接 return，零开销。

---

## 诊断能看到什么

### 启动时的环境体检

插件加载时自动输出一次，只读，不加载模型、不改 ComfyUI 状态：

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

--- Attention ---
backend        : attention_pytorch
SageAttention  : available
Sol-Attn ready : False — sm_86 is below SM89 ...

--- H3 PackedLayout ownership (continuity) ---
owner          : stock

--- Core regression scan ---
! L168: v = v.clone()  <- full value-tensor clone in the H3 attention path

--- ComfyUI launch ---
vram mode      : NORMAL_VRAM
pinned memory  : enabled
system RAM     : 57153 / 65393 MiB available
```

几个它专门在查的坑：

- **`comfy_kitchen` CUDA 后端是否被静默禁用。** ComfyUI 在 `comfy/quant_ops.py` 里按
  `torch.version.cuda >= 13` 卡这个后端。不满足就只打一行 warning 然后走模拟路径——
  H3 的 int8-convrot 专用 kernel 全部失效，[实测慢 2.17 倍](https://note.com/tank_ai/n/nab26edd4ab96)。
  ComfyUI 便携版自带的 PyTorch 不会随核心更新，很容易长期处于这个状态而不自知。
- **Sol-Attn 稀疏注意力在这张卡上能不能用。** Triton kernel 需要 SM89+（Ada/Blackwell），
  SM86（RTX 30 系）没有 TMA，装了也只会回退到 dense。省得白折腾。
- **`PackedLayout` 被谁占了。** 段间引导会 patch 它，遇到不认识的第三方 wrapper 会直接报错。
- **core 里的已知高显存写法。**

### 运行时探针（`memory_debug = true`）

```
[Baseline] COLD start — timings are comparable to other cold runs.
[Memory] Run Start | RSS=... | AvailRAM=... | NVMLUsed=... | GPUShared=... | Δt=...
[Model]  First Sampling (pre) | MiniMaxH3Model | tensors=... | stored=...GiB | ...
[Resident] First Sampling (pre) | MiniMaxH3TEModel_ 14980MiB @cuda:0; ...
[Tokens] Before Refine Sampling | latent=(...) | video_tokens=... | ffn_intermediate~...MiB
```

- **`GPUShared`** —— Windows「GPU Process Memory」性能计数器。**NVML 看不到 WDDM 溢出，这个能看到。**
- **`AllocRetries` / `CudaOOMs`** —— torch 分配器的压力计数，溢出的进程内指纹
- **`[Resident]`** —— 每个关键时刻显存里到底住着什么
- **`[Model]`** —— 每一遍采样实际用的是哪个 checkpoint（ComfyUI 的日志不说）
- **`[Baseline]`** —— 自动判定冷/热启动。热启动的计时不可与冷启动比较，会明确警告

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Slash214/ComfyUI-MinMax-H3-Production-Director.git
pip install -r ComfyUI-MinMax-H3-Production-Director/requirements.txt
```

依赖、模型、工作流与上游完全一致，**请参照[上游 README](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)**。
本仓库不重复维护那部分文档，以免与上游脱节。

`example_workflows/` 下的示例工作流同样来自上游。

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

**其他配置未经测试。** 24GB 以上的卡大概率用不上这些改动。

---

## 已知限制与未完成

- `balanced_20gb` 在**多段**场景下的净收益尚未验证——每段都要重新 stage 一次 TE（约 15GB），
  单段是纯赚，多段需要单独测。**做长视频/多镜头前请自行对比。**
- `aggressive_lowmem` 是空档，行为等同 `standard`。
- 跳过 pre-refine 解码（可省约 13s）、`--disable-pinned-memory` 对照、
  FFN 分块降峰值显存——均已定位，尚未实施。
- Sol-Attn 稀疏注意力在 SM86 上不可用，等社区出 Ampere 后端。

---

## 致谢

本分支的全部核心功能来自上游，作者是 **[AI搅拌手 / AIMixer](https://github.com/AIMixer)**。
时间轴、段间引导、二采/放大、多任务模式、前端 UI——这些都是上游的工作，
本分支只是在外围加了内存调度与诊断。

- **[AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)** — 上游原仓库
- [Comfy-Org / ComfyUI](https://github.com/Comfy-Org/ComfyUI) — 官方 MiniMax H3 支持
- [MiniMax-AI](https://github.com/MiniMax-AI) — MiniMax H3 模型
- [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) — 段间运动续拍思路
- [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) — H3 3D latent 放大
- [Comfyit 搅拌站](https://comfyit.cn/) — 模型、工作流与教程配套

**遇到问题请先确认是不是本分支引入的**（把 `memory_strategy` 切回 `standard` 复现一次）。
如果 `standard` 下同样存在，那是上游的问题，请去上游反馈，不要占用上游维护者处理本分支的时间。

## 许可证

Apache-2.0，与上游一致。
