# 诊断跑法 · 第一轮

目的：**不改画质、不改段间引导**，先把"二采时电脑卡"这件事从猜测变成数字。

## 怎么跑

1. 拉这个分支到目标机器，重启 ComfyUI。
2. **启动日志**里会自动出现一段 `=== MiniMax H3 Director — Environment Report ===`，先把这段整个复制下来。
3. 在导演台节点上把 **`memory_debug` 打开**（`性能` 组里）。
4. 跑一个**单段** R2V，参数用你平时出片的那套，**不要为了测试降任何参数**：
   - 一采分辨率、二采分辨率、steps、sigmas、LoRA、refine 全部照旧
   - 124 帧 / 24fps
   - 固定 seed
5. 跑完把整个控制台日志 + 节点 `report` 输出一起发回来。

如果时间允许，同一套参数再跑一次 `memory_strategy=standard`，做对照。

## 日志里要看什么

新增了三类行：

```
[Memory] ... | NVMLUsed=... | GPUShared=... | AllocRetries=... | Δt=...
[Tokens] ... | latent=(...) | video_tokens=... | ffn_intermediate~...MiB
[Resident] ... | MiniMaxH3<...> 11234MiB @cuda:0; ...
```

关键位置有四个：

| 标签 | 说明 |
|---|---|
| `First Sampling (pre)` | 一采基线：token 数 + 当时常驻了什么 |
| `Refine (pre)` | 二采前、**放大之前** |
| `Before Refine Sampling` | 二采前、**放大之后** ← 全流程真正的峰值 |
| `Before Final VAE Decode` | 解码前 |

### 三个待验证的假设

**1. 二采溢出到 WDDM Shared Memory**

看 `Before Refine Sampling` 那行：

- `NVMLUsed` 是否贴近 `NVMLTotal`
- `GPUShared` 是否从接近 0 跳到几个 GiB
- `AllocRetries` 是否开始增长

三个里中两个 → 假设成立，"卡"就是 PCIe 换页，不是算力不够。

> `GPUShared` 走的是 Windows「GPU Process Memory」性能计数器。如果这行没出现，说明计数器读不到（不影响其他项），靠 `NVMLUsed` 和 `AllocRetries` 也能判断。

**2. Text Encoder 在二采时还占着显存**

看 `[Resident] Before Refine Sampling` 那行有没有 Qwen3-VL / TEModel 之类的条目。

`balanced_20gb` 在 RAM 充足时会跳过 `unload_all_models`，模型全部保持常驻。这对系统 RAM 是好事，但如果 15GB 的 TE 在二采时还压在 20GB 卡上，那它就是二采峰值的主因——正好对应"整体没那么卡、但二采还是卡"。

若成立，解法是**定向卸载**（只放 TE，保住 VAE 和 diffusion），而不是回到 `standard` 的无差别 `unload_all_models`。

**3. Chunk FeedForward 值不值得**

看 `[Tokens] Before Refine Sampling` 的 `video_tokens`：

| video_tokens | 切 2 块省下 | 判断 |
|---|---|---|
| ~8K | ~238 MiB | 不值得 |
| ~16K | ~476 MiB | 边缘 |
| ~32K | ~950 MiB | 值得 |
| ~65K | ~1.9 GiB | 明显值得 |

该节点输出经 `assert_close(rtol=0, atol=0)` 验证与原实现**逐比特相同**，吞吐中性，只降峰值显存。若假设 1 成立且 token 数够大，这是第一个该试的东西。

## 本轮改了什么

全部是**加法**，没有触碰任何生成数学路径：

| 文件 | 改动 |
|---|---|
| `director/env_diagnostics.py` | **新增**。只读环境体检 |
| `director/memory_debug.py` | 新增 GPUShared / AllocRetries / token 计数 / 常驻模型 dump 和 `probe()` |
| `director/executor_core.py` | 加 2 处 `probe()` 调用 + 报告里附环境体检 |
| `director/refine_sampling.py` | 放大后、二采前加 1 处 `probe()` |
| `director/h3_context_patches.py` | 移植上游 `78c9b2bc`：段间引导与 SolAttn 共存 |
| `__init__.py` | 启动时输出一次环境体检 |

`memory_debug` 关闭时所有探针直接 return，零开销。所有诊断代码包在 `try/except` 里，失败降级为 "unknown"，不会中断生成。
