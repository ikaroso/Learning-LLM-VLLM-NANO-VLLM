基于 nano-vllm 学习
- [(47 封私信 / 10 条消息) 推理框架极简入门：用Nano-vLLM搭建知识体系 - 知乎](https://zhuanlan.zhihu.com/p/2008285806222132143)
- [GeeeekExplorer/nano-vllm: Nano vLLM](https://github.com/GeeeekExplorer/nano-vllm)
- https://github.com/CalvinXKY/InfraTech/blob/main/llm_infer/nano_vllm.ipynb
- [bcefghj/learn-nano-vllm: nano-vllm 面试导向学习指南](https://github.com/bcefghj/learn-nano-vllm)
- [手搓 vLLM 推理引擎 | LLM 101](https://boots-coder.github.io/LLM-101-CN/deep-dives/nano-vllm.html)
- [(补充) pagedattention 在 nanovllm 实现的逻辑 - 知乎](https://zhuanlan.zhihu.com/p/1976230365208270830)

上述文章，很适合有ai infra基础的人学习，但对于初学者来说，逻辑跳跃太多，阅读困难。

---

# 动机：有了 KV Cache，为什么还要 PagedAttention？

上一篇已经搞清：**KV Cache 省的是「decode 时不要对历史 token 重算 K/V」**。

但「存下来」只是第一步。工程上真正卡脖子的，往往不是算不算得出，而是：

> **多请求、变长序列下，这块 KV 显存怎么切、怎么借、怎么还？**

KV Cache 的实现大致有三档：Naive → PagedAttention → Prefix Caching。Paged 解决的是 **显存管理与 Attention 寻址方式**，不是再发明一种新的 Attention 公式。

## Naive 哪里痛？

每个请求（seq）各自持有一段 **连续** KV，形状大致是：

```text
[num_layers, **max_seq_len**, num_kv_heads, head_dim]
[transformer层数，请求的最大可能的token长度，多头attention的头数，多头attention的隐藏层数量]
```

两条常见路，都不理想：

1. **按最大长度预分配**  
   实际只生成了 20 token，也占满 `max_seq_len=2048` → **内部碎片**巨大。
2. **边生成边 `torch.cat` / 扩容**  
   频繁拷贝和申请GPU内存；多请求动态增删后，GPU上的空闲区被打散，很难再拼成大块 → **外部碎片**，调度困难。

再叠加服务场景：

- 并发请求长短不一，无法按「最长的那条」统一批处理而不浪费；
- 相同 system prompt 的多个请求，各自拷一份 KV，**即使有相同的prompt前缀，也无法共享**。

所以动机可以收成一句话：

> KV Cache 解决「算过不再算」；PagedAttention 解决「多请求变长下，显存按需借还、尽量不浪费」。

---

# 三种方案对照

## 方案 A：Naive KV Cache（连续内存）

- 原理：每条 request 申请一块连续 Tensor，或动态 cat。
- 痛点：必须预估 max_len，或伴随昂贵拷贝；碎片严重；难共享。

```text
假设 max_seq_len = 8，实际只用了前几格（# = 已用，. = 预留浪费）

请求 A（实际长 3）:  [ # # # . . . . . ]   ← 整段连续，尾部全闲着
请求 B（实际长 5）:  [ # # # # # . . . ]
请求 C（实际长 2）:  [ # # . . . . . . ]

显存里大概长这样（每请求一条“专用跑道”）：

  |---- A 的整段 max_len ----||---- B 的整段 max_len ----||---- C ... ----|
  |###.....                  ||#####...                  ||##......       |

问题：短请求也占满 max_len；A/B 即使 system prompt 相同，KV 也各拷一份。
```

## 方案 B：PagedAttention（分页内存，vLLM / nano-vllm 核心）

类比 OS 虚拟内存：

| 概念 | 对应 |
|------|------|
| 物理页 | 固定大小的 KV **block**（如 16 token，代表针对这个模型，每个block能存下16个token的KV CACHE） |
| 物理页框池 | 全局预先申请的大 Tensor `kv_cache` |
| 页表 | 每请求一份 **block_table**：逻辑块下标 → 物理 `block_id` |

实现骨架：

1. 预分配全局大 Tensor（物理池）；
2. `BlockManager` 负责分配 / 释放物理块 ID；
3. Attention **不能再假设某条序列的 K/V 在显存里连续**，而要按 `block_table` gather 再算（或专用 CUDA kernel）。


数据结构直觉：

```text
BLOCK_SIZE = 16   # 下图为示意，用 block_size=4 画更短
Physical Cache: [num_blocks, block_size, num_kv_heads, head_dim]  # 每层 K/V 各一份， 即一共有 layers * 2 份这样的Physical Cache
Block Table:    [batch, max_blocks_per_seq]  # int，存物理块 ID （block_id） —— 每个请求，各自对应max_blocks_per_seq个block
```

```text
例：block_size=4
    请求 A 实际 10 token → 需要 3 块（最后一块只用 2 格）
    请求 B 实际  6 token → 需要 2 块

逻辑视图（对请求自己是“连续”的）：
  A:  [A0][A1][A2]     token: 0-3 | 4-7 | 8-9 .
  B:  [B0][B1]         token: 0-3 | 4-5 .

物理页池（全局一块大显存，按下标切块；空=空闲）：
  id:   0    1    2    3    4    5    6    7
      [B0] [A1] [  ] [A2] [  ] [B1] [  ] [A0] ...

页表（逻辑块 → 物理块，物理上不必相邻）：
  block_table[A] = [7, 1, 3]
  block_table[B] = [0, 5]

读 A 的第 5 个 token：
  logical_block = 5 // 4 = 1  →  physical = 1
  offset        = 5 %  4 = 1  →  读池[1] 的第 1 行
```

## 方案 C：Prefix Caching（在 Paged 的基础上，增加共享前缀）

在 BlockManager 上加 Hash / Radix：相同前缀（如 system prompt）的多个请求 **指向同一物理块**（引用计数 + 只读共享），显存零拷贝复用。  
→ 依赖方案 B 的「块级映射」；连续 Naive 很难做干净。（注意，）

至于怎么共享，其实逻辑也很直觉：前缀的数个token做hash，如果has码对上了而且token内容也都对上了，那就共享这段前缀token对应的KV cache。
注意，这个所谓的共享，是跨sequence（请求）共享

```text
A、B 共享同一段 system prompt（占满物理块 7），后半段各自生成：

  block_table[A] = [7, 1, 3]     # 前缀用物理块 7
  block_table[B] = [7, 5]        # 同样指向 7，不拷贝

物理池：
  id:   0    1    2    3    4    5    6    7
      [  ] [A1] [  ] [A2] [  ] [B1] [  ] [共享前缀]
                                           ↑
                                      ref_count = 2

Naive 对比：A、B 各有一份完整连续 KV，前缀在显存里出现两次。
```

## Paged 具体缓解什么？

| 问题 | Naive | Paged |
|------|--------|--------|
| 内部碎片 | 按 max_len 预留，浪费大 | 最多浪费不到 **一个 block** |
| 外部碎片 | 动态扩容/释放后难复用 | 任意空闲物理块都可借给任意请求 |
| 前缀共享 | 几乎只能各拷一份 | 多份 block_table 可指向同一物理块 |

> 注意：Paged **并不消灭**内部碎片，只是把它从「整段 max_len」压到「≤ 一个 block」。

---

# Prefill 与 Decode的概念

## Prefill

手上有长度为 `T` 的 prompt：一次前向算完，得到第 `T+1` 个 token，并把这 `T` 个 token 的 K/V 写入 cache。

Paged 视角（把上一句拆开）：

1. **算要几块**：每个物理块只能装 `block_size` 个 token 的 KV（如 16）。prompt 有 `T` 个 token，就要 `ceil(T / block_size)` 块。  
   例：`T=20`，`block_size=16` → 需要 2 块（第一块装 0–15，第二块装 16–19，第二块还空着 12 个槽）。
2. **领块 + 记表**：从全局空闲池里借出这些物理块（假设借到的 block_id 是 `7` 和 `1`），写进该请求的 `block_table = [7, 1]`。  
   表的含义：逻辑上第 0 段 token → 物理块 7；逻辑上第 1 段 → 物理块 1（两块在显存里不必相邻）。
3. **按地址写入**：第 `t` 个 token 的 K/V 落到  
   `block_id = block_table[t // block_size]`，`offset = t % block_size`  
   即「第T个token的KV cache在哪个物理块、块内第几行」。例：token 17 → `block_id=1`，`offset=1`，即第1个物理块的第1行。

需要多少块？从**字节**想也通，但会约成按 **token 数**：

```text
# 一个 token 在一次推理里产生的 KV 总字节（所有层、K+V）
token_kv_bytes = 2 * num_layers * num_kv_heads * head_dim * dtype.itemsize

# 一个物理 block 的容量（前面 allocate 里的 block_bytes）
# = token_kv_bytes * block_size
# 因为一个 block 就是专门装 block_size 个 token 的 KV

num_blocks = ceil( T * token_kv_bytes / block_bytes )
           = ceil( T * token_kv_bytes / (token_kv_bytes * block_size) )
           = ceil( T / block_size )
```

所以写 `ceil(T / block_size)` 不是省略了字节，而是 **block 的定义本身就是「能装 block_size 个 token」**；按字节除一遍，分子分母里的「每 token KV 字节」会消掉，剩下的就是 token 个数 / 每块 token 容量。

## Decode

之后每步只来 1 个新 token：算它的 K/V，**追加**进 cache；Attention 用新 Q 去查历史全部 K/V，计算出attention score，计算出attention结果。

Paged 视角：

- 当前末尾 block 未满 → 写进同一物理块的下一个 `offset`；
- 已满 → 再从 free list 领 **任意一个** 空闲 `block_id`，挂到 `block_table` 末尾。

请求结束（输出EOS或请求已经达到最长token数量）：块归还池子，立刻可给别人用。

---

# 关键澄清：开一大块显存 ≠ 还没做 Paged

常见疑惑：

> nano-vllm 的 `allocate_kv_cache` 就是把可用显存全拿来做 KV，和 Naive「尽量多占显存」有啥区别？

**区别不在「占多少」，而在「这块显存怎么切、怎么映射」。**

| 层次 | 做什么 | 对应代码/模块 |
|------|--------|----------------|
| 物理层 | 一次性开大 Tensor，按 `block_size` 切成很多槽 | `allocate_kv_cache` |
| 逻辑层 | 每请求 `block_table`：逻辑块 → 物理 `block_id` | `BlockManager` |
| 计算层 | Attention 按表 gather K/V | paged attention kernel / 读 cache 路径 |

因此：

> **大张量 = 物理页池；PagedAttention = block_table 间接寻址 + 按块按需借还。**  
> 只看 `allocate_kv_cache`，只能看到底座；分页语义要到 BlockManager 与读 cache 才完整。

两边都可以「把剩余显存尽量给 KV」——这点不稀奇。Naive 是「每请求一条连续带（常按 max_len）」；Paged 是「全局一池，按块租借，每请求的KV cache在物理上不连续，因而显存利用率高」。

---

# nano-vllm怎么实现和使用paged attention：1. 初始化完整调用链

初始化只做两件事：**圈好地（物理 KV 池）**、**招好管理员（BlockManager）**；还没有任何请求的 `block_table`，池里也没有有效 KV。

**本节边界（先钉死，避免和推理段搅在一起）：**

1. **一份模型实例 ↔ 一块大池**：一个 `ModelRunner`（一张卡上的这份权重）对应一个 `self.kv_cache`。不是按请求建池。
2. **多请求不在初始化里切分大池**：初始化只把池子按 `block_id=0..N-1` 摆好，并交给 `BlockManager` 的 free list；谁租哪些号，要到 §2 `schedule` / `allocate` 才发生。
3. **分层是池内布局，不是「每层一份独立额度」**：同一 `block_id` 在每层各有一格、同号成列；紧的是共享的 N 个 id。

---

## 0. 从用户代码到调用链（先建立「谁调谁」）

用户侧通常只有一行构造（`example.py`）：

```python
from nanovllm import LLM
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
# 之后才是 llm.generate(...) —— 那是 §2 推理，不在本节
```

`LLM` 本身不写逻辑，只是别名：

```python
# nanovllm/llm.py
class LLM(LLMEngine):
    pass
```

所以 **`LLM(...)` ≡ `LLMEngine.__init__(...)`**。真正干活的是引擎初始化，源码顺序如下（`nanovllm/engine/llm_engine.py`）：

```python
def __init__(self, model, **kwargs):
    config = Config(model, **config_kwargs)
    Sequence.block_size = config.kvcache_block_size

    # ① TP>1：先起子进程，每个子进程里也会跑一遍 ModelRunner.__init__
    for i in range(1, config.tensor_parallel_size):
        process = ctx.Process(target=ModelRunner, args=(config, i, event))
        process.start()

    # ② 主进程 rank=0：构造 ModelRunner —— 内部同步完成 1.1→1.4
    self.model_runner = ModelRunner(config, 0, self.events)

    # ③ Tokenizer / eos（与分页无关，但 Scheduler 要用 eos）
    self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
    config.eos = self.tokenizer.eos_token_id

    # ④ 必须在 ② 之后：此时 config.num_kvcache_blocks 已被 allocate_kv_cache 写成真实 N
    self.scheduler = Scheduler(config)   # 内部再 new BlockManager(N, block_size)
```

`warmup_model` / `allocate_kv_cache` **不是** `LLMEngine` 直接调用的；它们嵌在 `ModelRunner.__init__` 里（`nanovllm/engine/model_runner.py`）：

```python
def __init__(self, config, rank, event):
    # ... dist / device / dtype ...
    self.model = Qwen3ForCausalLM(hf_config)   # 1.1
    load_model(self.model, config.model)       # 1.1
    self.sampler = Sampler()                   # 1.1

    self.warmup_model()                        # ← 1.2 唯一调用点
    self.allocate_kv_cache()                   # ← 1.3 唯一调用点
    if not self.enforce_eager:
        self.capture_cudagraph()               # ← 1.4（可选）
    # ... 恢复 default device/dtype；TP>1 时 rank>0 进 loop ...
```

串成一张总图：

```text
example.py:  LLM(path, ...)
                 │
                 ▼
llm.py:      LLM = LLMEngine          # 无额外逻辑
                 │
                 ▼
llm_engine.py: LLMEngine.__init__
                 │
                 ├─ Config(...)
                 ├─ Sequence.block_size = ...
                 ├─ (TP>1) Process(target=ModelRunner, rank=1..)   # 子进程各自跑下面整段
                 │
                 ├─ ModelRunner(config, rank=0, ...)               # model_runner.py
                 │      │
                 │      ├─ 1.1 建模型 / load_model / Sampler
                 │      ├─ 1.2 self.warmup_model()                 # 摸 peak；尚无 KV 池
                 │      ├─ 1.3 self.allocate_kv_cache()            # 开池；写出 config.num_kvcache_blocks=N
                 │      └─ 1.4 self.capture_cudagraph()            # 可选
                 │
                 ├─ AutoTokenizer + config.eos
                 └─ 1.5 Scheduler(config)
                        └─ BlockManager(N, block_size)
                             free_block_ids = [0..N-1]
```

**依赖关系（顺序不能反）：**

| 步骤 | 产出 | 后面谁用 |
|------|------|----------|
| 1.1 建模型+加载权重 | 权重占显存；Attention 上 `k_cache=空张量` | 1.2 前向要跑通模型；1.3 挂接要找到带 `k_cache` 的模块 |
| 1.2 warmup | `memory_stats` 里的 peak / current | 1.3 用 peak 估算能开多少 block |
| 1.3 allocate | `self.kv_cache` + `config.num_kvcache_blocks=N` + 每层挂接 | 1.5 `BlockManager(N, ...)`；推理时写/读 cache |
| 1.4 cudagraph | decode 用的图（可选） | §2 decode `run_model` |
| 1.5 Scheduler | free list 管理员 | §2 `schedule` / `allocate` 才真正租块 |

初始化结束时：

| 已有 | 还没有 |
|------|--------|
| 模型权重在 GPU | 任何请求 / 任何 `Sequence` |
| **本实例**的全局 KV 物理池（按层×block_id 编排） | 任何 `block_table`（租约） |
| BlockManager 的 free list（`0..N-1` 全空闲） | 池里有效的 K/V 内容 |
| （可选）CUDA Graph | 按请求划分显存（根本不会在 init 做） |

阅读提示：1.3 里若被「层 / block_id / 多请求」绕晕，先跳到同节的「前置：大池、block、block_id、分层」，再回看 `layer_id` 挂接与容量公式。

---

## 1.1 建模型 / 加载权重 / Sampler

**调用位置：** `ModelRunner.__init__` 里、`warmup_model` **之前**（见上一节源码）。  
这一步只把「能跑前向的模型」准备好；**还不碰 KV 池**。

```python
# nanovllm/engine/model_runner.py — ModelRunner.__init__ 前半段
dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
torch.cuda.set_device(rank)
torch.set_default_dtype(hf_config.dtype)
torch.set_default_device("cuda")
self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)
self.sampler = Sampler()
# 下一行就是 self.warmup_model() —— 见 1.2
```

要点：

1. **先改全局 dtype/device**：后面 `torch.empty(...)` 开 KV 池时不传 dtype，就靠这里的 `set_default_dtype(hf_config.dtype)`。
2. **`Qwen3ForCausalLM`**：每层里的 `Qwen3Attention` 会构造 `layers.attention.Attention`。Attention 构造时：

```python
# nanovllm/layers/attention.py
self.k_cache = self.v_cache = torch.tensor([])   # 空张量占位
```

也就是说：模型结构建好时，cache **还不是** 物理池切片；要等到 1.3 里用 `self.kv_cache[0/1, layer_id]` 覆盖挂接。

3. **`load_model`**：把权重填进模型（占显存），影响后面 `mem_get_info` / peak。
4. **`Sampler`**：只在 `run` 末尾、rank 0 上从 logits 采样 token；与分页无关。

和上层的关系再钉一句：

```text
LLMEngine.__init__
  └─ self.model_runner = ModelRunner(...)   # 进入本段 1.1
        └─ （同一次 __init__ 继续）warmup → allocate → cudagraph
LLMEngine.__init__（ModelRunner 返回后）
  └─ self.scheduler = Scheduler(...)        # 1.5，用到 allocate 写回的 N
```

---

## 1.2 `warmup_model`：先跑一遍假推理

**调用位置（唯一）：** `ModelRunner.__init__` 中，紧接在 `Sampler()` 之后：

```python
self.sampler = Sampler()
self.warmup_model()          # ← 这里
self.allocate_kv_cache()     # 必须等 warmup 跑完，才能读到可信的 peak
```

**为何放在初始化里、且在 allocate 之前？** 因为 1.3 算 `num_kvcache_blocks` 要用 warmup 留下的显存 peak；不先摸峰，池子容易开太大，正式推理时激活一冲就 OOM。

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
    seq_len = min(max_num_batched_tokens, max_model_len)
    num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
    for seq in seqs:
        seq.num_scheduled_tokens = seq_len
    self.run(seqs, True)          # is_prefill=True
    torch.cuda.empty_cache()
```

**和 1.3 的衔接（读 peak）：**

后面算能开多少 block 时用了：

```python
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
config.num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
```

warmup 先把「模型前向可能冲到的峰值」跑出来并记进 peak，再按「预算 − 峰值相关项」估算 KV 池大小，避免池子开太大把推理时激活显存挤爆。

直觉可以记成：「先量无 KV 时模型+激活大概要多少，剩下再给 KV 池」。  
但源码公式比「总显存 − peak」更细一点：

```text
预算上限 ≈ total * gpu_memory_utilization
可用 ≈ 预算上限 - used - (peak - current)
     = total * util - used - peak + current
N = 可用 // block_bytes
```

`peak - current`：warmup 冲高后又回落的那截（激活等临时占用），要从预算里扣掉，免得正式跑时再冲一次 OOM。

**为何不算真正分页？**

1. 假 `Sequence` **没有** 经过 `BlockManager.allocate`，`seq.block_table` 仍是 `[]`。
2. `prepare_prefill` 里明确跳过写 slot：

```python
if not seq.block_table:    # warmup
    continue
```

3. Attention 里：

```python
if k_cache.numel() and v_cache.numel():
    store_kvcache(...)
```

此时 `k_cache` 仍是空张量（`numel()==0`），**不会写 cache**；prefill 直接用本次算出的连续 `k,v` 做 `flash_attn_varlen_func`（且 `block_tables=None`）。

**调用链小结（warmup 这一跳）：**

```text
ModelRunner.__init__
  └─ warmup_model()
       └─ self.run(seqs, is_prefill=True)
            ├─ prepare_prefill  （block_table=[] → 不写 slot_mapping）
            ├─ run_model → model forward（激活占显存，冲高 peak）
            └─ sampler（rank0）
       返回后 peak 已记在 CUDA memory_stats 里，供紧接着的 allocate_kv_cache 读取
```

---

## 1.3 `allocate_kv_cache`：开物理页池

**调用位置（唯一）：** `ModelRunner.__init__` 中，紧接在 `warmup_model()` 之后：

```python
self.warmup_model()
self.allocate_kv_cache()     # ← 这里；读完 peak，开池，写出 N
if not self.enforce_eager:
    self.capture_cudagraph()
```

**和初始化上下游的关系：**

```text
上游：1.2 warmup 已把 peak/current 写进 memory_stats；Attention.k_cache 仍是空张量
本步：开 self.kv_cache，挂到每层；config.num_kvcache_blocks = N
下游：ModelRunner.__init__ 返回后，LLMEngine 才 Scheduler(config)
      → BlockManager(N, block_size) 用同一个 N 建 free list
```

下面这段只负责 **物理层**：算一个 block 多大、能开多少个、把大张量挂到每层的 `k_cache` / `v_cache`。  
副作用：写回 `config.num_kvcache_blocks`，供后面 `Scheduler` / `BlockManager` 使用。  
（以下为 nano-vllm 源码 + 我的注解，原样保留。）

```
   def allocate_kv_cache(self):

        config = self.config

        hf_config = config.hf_config

        free, total = torch.cuda.mem_get_info()

        used = total - free

        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        num_kv_heads = hf_config.num_key_value_heads // self.world_size

        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
		
		
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
		# 计算一个block占多少字节数
		# 一个block，需要存储self.block_size个（行）K和V的缓存。
		# ×2，因为要存 K 和 V（不是 Q）
		# hf_config.num_hidden_layers = Transformer 层数（不是词表/隐藏维 D）
		# num_kv_heads，是 KV 头个数
		# head_dim，是每个 KV head 的维度
		# hf_config.dtype.itemsize, 代表模型参数里，每个元素占多少字节，比如float32，那就是4
		# 【整个计算的逻辑在于】 说白了，整个block里，要存储self.block_size个token对应的K和V cache
		# 因此，block_bytes首先算出了这样子花费多少个元素（这样存的维度是多少），然后乘上每个元素的字节数
		# 最终得到总的字节数

        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        # 目前空闲的GPU内存支持申请多少个block

        assert config.num_kvcache_blocks > 0
        # 能申请至少一个block

        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
		# 直接开一个大张量，作为整个 KV cache 的底层存储容器
		# 但是，这个张量要按照维度组织，不能直接 block_bytes x config.num_kvcache_blocks （虽然大小是一致的）
		# 而且注意，这里没有指定tensor的大小（字节数），因为前面已经指定过torch.set_default_dtype(hf_config.dtype)
		# 解释维度的组织方式，因为后面取缓存要按维度取：
		- 第 0 维：K 或 V 对应的缓存值
		- 第 1 维：模型第几层
		- 第 2 维：第几个 block
		- 第 3 维：这个 block 里有几个 token （哪几行缓存）
		- 第 4 维：每个 token （行） 的哪些 KV head
		- 第 5 维：每个 head 的向量长度 （即第5维存的就是每个head对应的一行缓存）
		
		
        layer_id = 0
		
		# 找出模型里每层对应的在self.kv_cache里的位置
        for module in self.model.modules():

            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):

                module.k_cache = self.kv_cache[0, layer_id]
                # 第layer_id层的K cache

                module.v_cache = self.kv_cache[1, layer_id]

                layer_id += 1
```

挂接之后，每层 Attention 的 `k_cache` / `v_cache` 形状为  
`[num_kvcache_blocks, block_size, num_kv_heads, head_dim]`，与后面 `slot = block_id * block_size + offset` 的线性下标一致（kernel 里按 `slot * D` 写）。

### 这段 `layer_id` 循环在干什么？

大池已经按 `[K/V, layer, block, ...]` 排好了；循环不是「再切一次维」，而是给**每一层 Attention 发一把钥匙**：让它的 `self.k_cache` / `self.v_cache` 指向大池里「本层那一行」。

```text
self.kv_cache 物理布局（示意，只画 K；V 同理再有一份）

          block_id →   0    1    2    3   ...
                     ┌────┬────┬────┬────┐
  layer 0  ───────── │    │    │    │    │  ← Attention层0.k_cache 指向这一整行
                     ├────┼────┼────┼────┤
  layer 1  ───────── │    │    │    │    │  ← Attention层1.k_cache 指向这一整行
                     ├────┼────┼────┼────┤
  layer 2  ───────── │    │    │    │    │  ← ...
                     └────┴────┴────┴────┘

赋值：
  module.k_cache = self.kv_cache[0, layer_id]   # 视图，不拷贝显存
  module.v_cache = self.kv_cache[1, layer_id]
```

为什么必须挂接？因为 forward 时 Attention **只认自己的成员**，不会去翻 `ModelRunner.kv_cache`：

```text
构造时：  Attention.k_cache = tensor([])     # 空占位
挂接后：  Attention.k_cache ──view──► 大池[K, 本层, :, :, :, :]

forward：
  store_kvcache(..., self.k_cache, ...)      # 写进「本层那一行」里的某些 block
  flash_attn_with_kvcache(..., self.k_cache, block_table=...)
```

若不做赋值，`k_cache.numel()==0`，`store_kvcache` 直接跳过，分页池等于没接上。

### 前置：大池、block、block_id、分层，到底什么关系？

先把名词钉死（后面图都靠这个）：

| 名词 | 是什么 |
|------|--------|
| **大池** `self.kv_cache` | 初始化时一次 `torch.empty` 出来的整块 GPU 显存 |
| **block（物理块）** | 大池里切出的固定大小格子，一格能装 `block_size` 个 token 的 K（或 V）向量 |
| **block_id** | 格子的编号，从 `0` 到 `N-1`（`N = num_kvcache_blocks`） |
| **block_table** | **某个请求**的「逻辑第 i 段 token → 用哪个 block_id」；存在 `Sequence` 上，不在大池里 |
| **分层** | 大池多了一个 `layer` 维：同一个 `block_id` 在每一层各有一格（存该层自己的 K/V 数值） |

只看 **一层、只看 K** 时，大池就是一条按 `block_id` 排开的格子带：

```text
一层上的 K 视图（挂接后的 module.k_cache）:

  block_id:   0      1      2      3     ...    N-1
            ┌──────┬──────┬──────┬──────┬     ┬──────┐
            │ 格0  │ 格1  │ 格2  │ 格3  │ ... │ 格N-1│
            │(可装  │      │      │      │     │      │
            │block_ │      │      │      │     │      │
            │size个 │      │      │      │     │      │
            │token) │      │      │      │     │      │
            └──────┴──────┴──────┴──────┴     ┴──────┘

BlockManager.free_block_ids = [还没租出去的 id]
某请求借到 id=2、id=0 → 只用格2和格0，其它格可以借给别人
```

加上 **层** 维之后（nano-vllm 实际布局），同一个 id 在每一层各出现一次：

```text
大池（只画 K；V 另有一份同样结构）

         block_id →  0    1    2    3   ...
                   ┌────┬────┬────┬────┐
  层 0             │    │    │    │    │  ← Attention0.k_cache
                   ├────┼────┼────┼────┤
  层 1             │    │    │    │    │  ← Attention1.k_cache
                   ├────┼────┼────┼────┤
  层 2             │    │    │    │    │
                   └────┴────┴────┴────┘
                      ↑
              竖着看：同一个 block_id=1
              =「一层一格」组成的一列，数值各层不同
```

关系一句话：

> **大池** = 全部格子；**block_id** = 格子编号；**分层** = 编号在每层各有一格；  
> **block_table** = 某条请求租了哪些编号（以及顺序），是租约，不是显存本身。

### `block_table` 怎么用？（从零看一张表）

假设 `block_size=4`，请求 A 现有 7 个 token，租了 2 个块：

```text
token 下标:  0 1 2 3 | 4 5 6
逻辑分段:    --段0--   -段1-

block_table[A] = [2, 0]
                  │  │
                  │  └─ 逻辑段1 → 物理 block_id=0
                  └──── 逻辑段0 → 物理 block_id=2
```

在「一层」上看，数据落在哪些格：

```text
该层格子:  id0    id1    id2    id3
           ┌────┐      ┌────┐
           │456.│      │0123│     ← A 的 token
           └────┘      └────┘
             ↑           ↑
        table[1]=0   table[0]=2

读第 5 个 token：
  逻辑段 = 5 // 4 = 1  → block_id = table[1] = 0
  格内偏移 = 5 % 4 = 1 → 读 id0 格子的第 1 行
```

全层一起看时：**表只有一份**（租约共享），每层往自己那一行的同号格里写**本层**算出的 K/V：

```text
block_table[A] = [2, 0]   ← 全模型共用这一张租约

层0 的 id2 / id0 ← 存层0 的 K（数值）
层1 的 id2 / id0 ← 存层1 的 K（数值不同）
层2 的 id2 / id0 ← 存层2 的 K
...
```

所以「各层都出现同一批 block_id」的意思只是：**租约相同，各层各写各的数**；不是又有三份不同的 block_table。

### 按层切开：是设计选择，不是唯一做法

你的直觉对：**存储上**完全可以做成「一个扁平大池 + 自己做 (layer, block) 映射」，只要读写时别把层用错。  
nano-vllm / vLLM 用「张量上显式加 layer 维 + 每层挂接切片」，主要是工程上省事，不是物理定律。

| | 按层维切开（当前做法） | 扁平大池 + 手动映射 |
|--|------------------------|---------------------|
| 正确性 | 本层指针只指向本层，不容易串层 | 也能正确，但每次要算/传入 layer |
| 前向 | `self.k_cache` 直接用 | Attention 或 kernel 需带 layer 下标 |
| 页表 | 一份 `block_table`，各层同号 | 也可以一份表；或做成更复杂的全局 id |

仍有一条**硬约束**（和「池子怎么切」无关）：

> 算第 L 层 Attention 时，只能用第 L 层的 K/V。  
> 第 0 层的向量和第 1 层的向量不能当同一套 cache 用——那是模型语义，不是分页布局问题。

「按层切开」保证的是：**存放位置天然对齐这条语义**；不是说「不做 layer 维就存不了」。

### 和「动态增减」的关系

```text
固定（初始化）：
  · 大池里有 N 个 block_id（每层各 N 格，同号成列）
  · 第 L 层模块挂到大池第 L 行

动态（推理时）：
  · 变的是租约 block_table（每个请求独立向 free list 借/还 id）
  · 不是改大池形状，也不是换层

增：may_append → 再借一个 id，append 进 block_table
减：deallocate → id 还回 free；大张量不缩水
```

### 从容量公式到张量编排（一次讲清）

常见疑惑：

> `num_kvcache_blocks` 明明是「整卡预算最多能开几个 KV block」，  
> 张量却是 `[2, L, N, ...]`，挂接后每层又都能用 N 格——岂不是变成「每层各 N 个」，显存乘了 L 倍？

**不会。** 公式和编排是对同一笔账的两种写法。

#### 1. 先定「一个 block_id 有多贵」

```text
token 在「一层」上的 K 或 V：  block_size × heads × dim × itemsize
一个 block_id 要覆盖：
  ×2     → K 和 V
  ×L     → 每一层各存一份（同号槽，数值不同）

block_bytes = 2 × L × block_size × heads × dim × itemsize
```

这里的「1 个 block」= BlockManager 里的 **一个 `block_id`**，不是「某一层上的一格」单独计价。

#### 2. 再用预算去除，得到 N

```text
N = num_kvcache_blocks = 显存预算 // block_bytes
```

含义：free list 里最多有 **N 个可出租的 block_id**（整池共享，不是每层一份 free list）。

#### 3. 开张量：把这笔总账排成多维

```text
总元素数 = 2 × L × N × block_size × heads × dim
总字节   = 总元素数 × itemsize
         = N × block_bytes          ← 和上面除法对得上
```

编排成：

```text
kv_cache.shape = [2, L, N, block_size, heads, dim]
                  │  │  │
                  │  │  └─ block_id 轴（与 free list 一一对应）
                  │  └──── 层轴（前向按层取切片）
                  └─────── K / V
```

#### 4. 为何「每层都是 N 格」却不加倍？

因为 **N 格是同一套 block_id 在每一层上的投影**，不是又独立开了 L 份额度：

```text
          block_id:  0   1   2  ...  N-1
层0:               [ ] [ ] [ ] ... [ ]
层1:               [ ] [ ] [ ] ... [ ]
...
层L-1:             [ ] [ ] [ ] ... [ ]

领 block_id=7 → 每一层的「第 7 格」都被这条请求占用（一起算进 1 个 block_bytes）
可同时出租的 id 数 = N，不是 N×L
```

直觉上像：**(5,6,7) 与 (6,5,7) 元素个数一样**——轴顺序变了，总量不变。  
这里也一样：你也可以想象成先排 `[2, N, L, ...]`，总字节仍是 `N × block_bytes`；写成 `[2, L, N, ...]` 只是方便 `kv_cache[0, layer_id]` 抽出本层视图。

```text
错误心智：  N = 全卡总块数 → 再分给 L 层 → 每层 N/L
            或：每层再各用 N 个 → 总块数 N×L（多乘了一次 L）

正确心智：  N = 共享 block_id 个数
            算 block_bytes 时已经 ×L
            张量里每层长度也是 N = 同号槽各层一份，账已付清
```

#### 5. 一条链串起来

```text
显存预算
  │
  ▼
block_bytes =「1 个 block_id 在全部 L 层上的 K+V」
  │
  ▼
N = 预算 // block_bytes          ← 能出租多少个 id
  │
  ▼
torch.empty(2, L, N, block_size, ...)   ← 总容量 = N 份 block_bytes
  │
  ├─ BlockManager(N)：free_block_ids = [0..N-1]
  └─ 每层 Attention.k_cache → kv_cache[0, layer]   形状 [N, block_size, ...]
       推理时：借 id → 各层写自己那一行的同号格；还 id → 整列（各层同号格）一起空闲
```

---

## 1.4 `capture_cudagraph`（可选）

**调用位置：** 仍在 `ModelRunner.__init__` 末尾，**必须在 1.3 之后**（图里的 Attention 已经挂上真实 `k_cache`）：

```python
self.allocate_kv_cache()
if not self.enforce_eager:
    self.capture_cudagraph()   # ← 这里
```

`enforce_eager=True`（如 `example.py`）时跳过；之后 decode 走 eager 前向。

`capture_cudagraph` 干的事（只服务 **decode**，`set_context(False, ...)`）：

1. 按 `max_bs = min(max_num_seqs, 512)` 预分配一批 **固定地址** 的 buffer：

```python
input_ids, positions, slot_mapping, context_lens, block_tables, outputs
# block_tables 列数 = ceil(max_model_len / block_size)
```

2. 对若干 batch size（`1,2,4,8,16,...`）各录一张 CUDA Graph：  
   warmup 一次 → `with torch.cuda.graph(...)` 再 capture 一次 `self.model(input_ids[:bs], positions[:bs])`。
3. 把这些 buffer 存进 `self.graph_vars`，推理时 `run_model` 在 decode 路径：

```python
# 非 prefill、非 eager、且 batch≤512
graph_vars["slot_mapping"][:bs] = context.slot_mapping
graph_vars["context_lens"][:bs] = context.context_lens
graph_vars["block_tables"][:bs, :...] = context.block_tables
graph.replay()
```

也就是说：图里已经假定 Attention 会读 `context` 里的 `slot_mapping` / `block_tables`；capture 阶段只是把「带分页参数的 decode 前向」录下来，**并不分配任何请求的物理块**。

---

## 1.5 `Scheduler` / `BlockManager`：逻辑页表管理器就位

**调用位置：** 不在 `ModelRunner` 里，而在 `LLMEngine.__init__` 后半段——等 `ModelRunner(...)` **整段返回**（1.1–1.4 都做完）之后：

```python
# nanovllm/engine/llm_engine.py
self.model_runner = ModelRunner(config, 0, self.events)  # 内部已跑完 warmup + allocate，N 已写回 config
self.tokenizer = AutoTokenizer.from_pretrained(...)
config.eos = self.tokenizer.eos_token_id
self.scheduler = Scheduler(config)                       # ← 1.5 从这里进入
```

前面 1.3 已经在 GPU 上开好了大池 `kv_cache`。  
但大池自己**不会记账**：不知道「哪些 `block_id` 空着、哪个请求租了哪些号」。  
所以还要在 **CPU / Python 侧** 再建一个「管理员」——这就是 `BlockManager`；它挂在 `Scheduler` 下面。

### 谁在什么时候创建出来？

```python
# LLMEngine.__init__ 里，顺序不能反：
self.model_runner = ModelRunner(...)   # 内部会 allocate_kv_cache，写出 config.num_kvcache_blocks = N，即上面的1.1-1.4
...
self.scheduler = Scheduler(config)     # 用 N 去建 BlockManager
```

```python
# Scheduler.__init__
self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
```

为什么必须先 1.3？因为 `Config` 里 `num_kvcache_blocks` 默认是 `-1`，只有 `allocate_kv_cache` 算完之后才变成真实的 N。管理员得知道「货架上一共有几格」。

### 两个角色分别干什么？（先建立直觉）

```text
Scheduler（调度员）
  · 决定下一步跑哪些请求（prefill / decode）
  · 需要块时，开口向 BlockManager 借；结束了让它还
  · 初始化时只是把 waiting/running 队列建好，此时队列都是空的

BlockManager（仓库管理员）
  · 不管 GPU 上的数字长什么样
  · 只维护：每个 block_id 闲不闲、谁引用了、前缀 hash 是什么
  · 初始化时：宣布 0..N-1 全部空闲，等待出租
```

和 1.3 的关系：

```text
1.3  ModelRunner.kv_cache     ← 真正的显存货架（GPU）
1.5  BlockManager             ← 货架的登记本（Python）
       free_block_ids         ← 「哪些格号还能租」
       blocks[i]              ← 格号 i 的登记卡片（引用计数等）

两者通过 block_id 对齐：
  登记本上的「7号已租给请求A」
  ⇔  大池里各层的第 7 格归 A 写
```

### `BlockManager.__init__` 四行在建什么？

```python
self.blocks = [Block(i) for i in range(num_blocks)]  # N 张卡片，编号 0..N-1
self.hash_to_block_id = {}                           # 前缀缓存用，init 时空的
self.free_block_ids = deque(range(num_blocks))       # 空闲队列：[0,1,2,...,N-1]
self.used_block_ids = set()                          # 已租出的集合：空
```

```text
刚建好时：

  free_block_ids:  0  1  2  3  ...  N-1     ← 全都可租
  used_block_ids:  （空）
  任何请求的 block_table: 还不存在（还没有请求）

  GPU 大池里格子也在，但里面是未初始化/无意义数据
  ——「有货架」≠「已经有人的 KV」
```

注意：这里 **再也不会** `torch.empty` 一块显存。显存已经在 1.3 开完了；这里只是笔记本。

### 初始化结束时是什么状态？

术语沿用 1.3 大池维度（不要另造含义）：

```text
self.kv_cache.shape = [2, L, N, block_size, heads, dim]
                       │  │  │
                       │  │  └─ 第2维：block_id（列号 0..N-1）——BlockManager 出租的就是这个号（对每个请求独立处理）
                       │  └──── 第1维：Transformer 第几层（模型里第几个 Attention）
                       └─────── 第0维：K(=0) / V(=1)
```

这里说的「层」= **第1维**，即 Transformer 的第几个 Attention 层；  
「一格」= 固定 `(K或V, 某层, 某个 block_id)` 下、能装 `block_size` 个 token 的那一块。

| 有 | 没有 |
|----|------|
| GPU 大池已按上表开好：`N` 个 block_id；对每个 id，在 **每一层的 K、每一层的 V** 上各占一格（同号，即第2维相同） | 任何一个请求 |
| 登记本：0..N-1 全在 free 里 | 任何一份 `block_table` |
| Scheduler 空的 waiting / running 队列 | 往格子里写过的有效 K/V |

```text
示意（先 K 后 V 各有一套「层 × block_id」；这里只画 K）：

  第1维\第2维   id0  id1  id2  ...  idN-1
  Transformer层0  ■    ■    ■   ...   ■
  Transformer层1  ■    ■    ■   ...   ■
  ...
  租出 block_id=2 ⇒ 每一层 K 的 id2 格、每一层 V 的 id2 格，都划给同一租约
```


所以你在初始化段「看不到多请求怎么分池」是对的——**还没开始分**。  
分法预告（要到 §2 才执行）：

```text
请求进来 → Scheduler 调用 block_manager.allocate
         → 从 free_block_ids 弹出几个号
         → 写进该请求的 seq.block_table = [3, 0, ...]
         → 之后 Attention 按这些号去大池里读写
```

一句话：**1.3 建货架，1.5 建登记本并宣布全部空闲；租给谁是推理时的事。**  
（`blocks` / `free_block_ids` / `block_table` / `seq.block(i)` 的对照总表见 **§2「关键对象一览」**。）

---

# nano-vllm：2. 真正推理时发生了什么

承接 §1：大池 `kv_cache` 已开好，`BlockManager.free_block_ids=[0..N-1]`，尚无任何请求。

初始化只「圈地 + 建登记本」；**分页语义从这里才真正开始转**：请求入队 → 租块写 `block_table` → 算 slot → 写/读大池 → 还块。

**本节边界：**

1. 入口是 `llm.generate(...)`，不是再造池子。
2. 每一轮外层循环都是一次 `step()`：`schedule → run → postprocess`。
3. Prefill / Decode 走同一条 `step` 链，只是 `schedule` / `prepare_*` / Attention 分支不同。

手算省例（仅方便算数；真实默认 `block_size=256`，且源码里常要求长度与块对齐相关逻辑按 `% block_size` 理解）：

```text
block_size = 4
prompt 有 7 个 token：t0..t6  →  Sequence.num_blocks = ceil(7/4) = 2
```

阅读顺序建议：先扫 **「0. 调用链」** → 再精读 **「关键对象一览」** → 再跟 2.1 起的主线。

---

## 0. 从用户代码到调用链（先建立「谁调谁」）

用户侧（`example.py`）在 `LLM(...)` 之后：

```python
outputs = llm.generate(prompts, sampling_params)
```

`generate` / `step` / `add_request` 都在 `nanovllm/engine/llm_engine.py`：

```python
def generate(self, prompts, sampling_params, use_tqdm=True):
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)          # ← 2.1：只入 waiting，不租块
    while not self.is_finished():             # waiting/running 都空才结束
        output, num_tokens = self.step()      # ← 每轮：schedule → run → postprocess
    ...

def step(self):
    seqs, is_prefill = self.scheduler.schedule()                 # ① 决定本批谁跑、是否 prefill；可能租块
    token_ids = self.model_runner.call("run", seqs, is_prefill)  # ② 前向写 KV + 采样出新 token
    self.scheduler.postprocess(seqs, token_ids, is_prefill)      # ③ 记账 / append / 结束则还块
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

`ModelRunner.call`：TP=1 时就是直接 `self.run(...)`；TP>1 时 rank0 还会经共享内存通知其它 rank 同步跑同一方法。

涉及文件：

| 文件 | 角色 |
|------|------|
| `nanovllm/engine/llm_engine.py` | 外层：`generate` / `add_request` / `step` |
| `nanovllm/engine/sequence.py` | 每个请求一个 `Sequence`（含 `block_table`） |
| `nanovllm/engine/scheduler.py` | `schedule` / `postprocess`；内部持有 `BlockManager` |
| `nanovllm/engine/block_manager.py` | `can_allocate` / `allocate` / `may_append` / `deallocate` / `hash_blocks` |
| `nanovllm/engine/model_runner.py` | `run` / `prepare_prefill` / `prepare_decode` / `run_model` |
| `nanovllm/layers/attention.py` | `Attention.forward` / `store_kvcache` |
| `nanovllm/utils/context.py` | `set_context` / `get_context`（本步临时参数） |

总图（与源码一致）：

```text
example.py:  llm.generate(prompts, sp)
                 │
                 ▼
llm_engine.py: LLMEngine.generate
                 │
                 ├─ 2.1  add_request → Sequence(...) → Scheduler.add → waiting
                 │
                 └─ while not finished:
                      step():
                        ① Scheduler.schedule()                         # 2.2
                             ├─ (prefill) can_allocate / allocate
                             └─ (decode)  can_append / may_append / 可能 preempt→deallocate
                        ② ModelRunner.call("run", seqs, is_prefill)    # 2.3
                             └─ ModelRunner.run
                                  ├─ prepare_prefill 或 prepare_decode → set_context(...)
                                  ├─ run_model → model(...) → 每层 Attention.forward
                                  │                ├─ store_kvcache(..., slot_mapping)
                                  │                └─ flash_attn_*(..., block_table=...)
                                  ├─ Sampler → token_ids
                                  └─ reset_context()
                        ③ Scheduler.postprocess(...)                   # 2.4
                             ├─ hash_blocks
                             ├─ 更新 num_cached_tokens；可能 append_token
                             └─ 结束则 deallocate
```

**依赖关系（一轮 step 内不能反）：**

| 步骤 | 产出 | 后面谁用 |
|------|------|----------|
| 2.1 入队 | `Sequence` 在 `waiting`；`block_table=[]` | 2.2 `schedule` 从 waiting 取 |
| 2.2 schedule | 租约 `block_table`；`num_scheduled_tokens`；`is_prefill` | 2.3 `prepare_*` 算 slot / 组 batch |
| 2.3 run | 本步 K/V 写入大池；采出 `token_ids` | 2.4 追加 token / 登记 hash / 判断结束 |
| 2.4 postprocess | 序列变长或 FINISHED；满块 hash | 下一轮 schedule（decode 或新 prefill） |

---

## 关键对象一览（读 2.2 / BlockManager 之前必看）

后面 `can_allocate` / `allocate` / `prepare_*` 会反复甩这些名字。  
**先在这里钉死「各自是什么、谁和谁对齐」**；细节算法仍在 2.2 起展开。

### 1. 一张总图

```text
┌─ GPU（§1.3 已开好）─────────────────────────────────────────┐
│  kv_cache[2, L, N, block_size, heads, dim]                    │
│                 ↑                                              │
│            物理格号 block_id = 0 .. N-1                         │
│            每格能装 block_size 行 K（或 V）向量                  │
└───────────────────────────────────────────────────────────────┘
                              ▲
                              │ 用同一个 block_id 当「门牌号」
                              │
┌─ CPU：BlockManager（登记本，§1.5）─────────────────────────────┐
│  free_block_ids  : 还能租出去的门牌号队列                       │
│  used_block_ids  : 当前有人引用的门牌号集合                     │
│  blocks[i]       : 门牌号 i 的「卡片」（ref/hash/token_ids）     │
│  hash_to_block_id: 前缀指纹 → 门牌号（prefix cache）            │
└───────────────────────────────────────────────────────────────┘
                              ▲
                              │ allocate 写出 / may_append 追加
                              │
┌─ CPU：每条请求一个 Sequence ───────────────────────────────────┐
│  token_ids       : 整条序列的 token id 列表                     │
│  block(i)        : 按逻辑切出第 i 段 token（内容，不是门牌号）   │
│  block_table     : 租约 list，block_table[i] = 物理 block_id    │
│  num_cached_tokens : 已有历史 KV 的长度（init=0；allocate/postprocess 改）│
│  num_scheduled_tokens : 本轮要新算几个（schedule 写；postprocess 清 0）  │
└───────────────────────────────────────────────────────────────┘
```

### 2. 共用旋钮：`block_size`

| | 含义 |
|--|------|
| 配置 | `Config.kvcache_block_size`（默认 256） |
| 逻辑 | 一段最多装多少个 **token id** |
| 物理 | 一格最多装多少行 **K/V** |
| 对齐 | `Sequence.block_size`、池子第 3 维、`BlockManager.block_size` **都是它** |

### 3. `BlockManager` 四个成员（登记本，不是显存）

`blocks[i]` **不是** `kv_cache` 里的张量，只是 Python 侧一张卡片。  
`[Block(i) for i in range(N)]` 初始化出来就是 **长度为 N 的 list**，第 i 个元素是一个普通对象：

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id   # 门牌号，等于下标 i
        self.ref_count = 0         # 几条序列正挂着这个号
        self.hash = -1             # 满块内容指纹；-1 = 未登记
        self.token_ids = []        # 该满块当时的 token 原文（供比对）

# BlockManager.__init__
self.blocks = [Block(i) for i in range(num_blocks)]   # N = num_kvcache_blocks
self.free_block_ids = deque(range(num_blocks))        # 空闲门牌号
self.used_block_ids = set()                  # 正在被引用的门牌号
self.hash_to_block_id = {}                   # 前缀 hash → block_id
```

刚 init 完时（N=4 示意）：

```text
self.blocks = [
  Block(block_id=0, ref_count=0, hash=-1, token_ids=[]),
  Block(block_id=1, ref_count=0, hash=-1, token_ids=[]),
  Block(block_id=2, ref_count=0, hash=-1, token_ids=[]),
  Block(block_id=3, ref_count=0, hash=-1, token_ids=[]),
]
# 真正存 K/V 的是 GPU 上 kv_cache[..., block_id, :, ...]
# self.blocks[block_id] 只回答：这格租给谁了、指纹是什么
```

| 成员 | 类型 | 干什么 |
|------|------|--------|
| `blocks[block_id]` | `Block` | 该门牌的 `ref_count`、`hash`、`token_ids`（内容指纹用） |
| `free_block_ids` | `deque[int]` | 还可租的 `block_id`；`_allocate_block` 从左边 `popleft` |
| `used_block_ids` | `set[int]` | 当前 `ref_count>0` 的号；和 free **互补**（同号不同时在两边） |
| `hash_to_block_id` | `dict` | prefix：满块内容指纹 → 可复用的物理号 |

```text
注意两种「空闲」：
  · 在 free 里：没人引用，可再租；hash 表里未必删干净（内容还可能被 prefix 认领）
  · 不在 hash 表：内容不当前缀复用；或从未登记过
```

`Block` 卡片字段：

| 字段 | 含义 |
|------|------|
| `block_id` | 门牌号，等于在 `blocks` 里的下标，也等于 GPU 大池第 2 维下标 |
| `ref_count` | 多少条序列的 `block_table` 正挂着这个号；0 则可回 free |
| `hash` / `token_ids` | 该满块内容的指纹与原文，供 prefix 比对 |

### 4. `Sequence`：逻辑切分 vs 租约

```python
# sequence.py（要点）
token_ids          # 整条：[t0, t1, ..., t6, ...]
num_blocks         # ceil(len / block_size)
block(i)           # return token_ids[i*block_size : (i+1)*block_size]  ← 只要内容
block_table        # list[int]，起初 []；allocate 后如 [7, 1]
```

| | `seq.block(i)` | `seq.block_table[i]` |
|--|----------------|----------------------|
| 是什么 | 逻辑第 i 段的 **token id 列表** | 这段 KV 存在哪个 **物理 block_id** |
| 在哪 | 只读 `token_ids` 切片 | 租约，由 BlockManager 写入 |
| 有租约前 | 就能算（入队后立刻有） | 还是 `[]` |
| 用途 | hash / prefix 比内容 | `prepare_*` 算 slot；FA gather |

省例 `block_size=4`，7 个 token：

```text
token_ids:  t0 t1 t2 t3 | t4 t5 t6
逻辑段号 i: ---- 0 ----   -- 1 --

seq.block(0) = [t0,t1,t2,t3]     # 内容
seq.block(1) = [t4,t5,t6]

allocate 之后例如：
block_table  = [7, 1]            # 租约：逻辑0→物理7，逻辑1→物理1

对应关系：
  逻辑段 i 的 token 内容     = seq.block(i)
  这些 token 的 K/V 存哪     = 大池[..., block_table[i], ...]
```

### 5. 和 GPU 大池怎么对上号：slot

大池完整形状：

```text
kv_cache[2, L, N, block_size, heads, dim]
         │  │  │      │
         │  │  │      └─ 第3维：格内第几个 token（offset）
         │  │  └──────── 第2维：物理 block_id
         │  └─────────── 第1维：Transformer 层
         └────────────── 第0维：K / V
```

挂接后，某一层的 `k_cache` / `v_cache` 只剩后四维：`[N, block_size, heads, dim]`。  
**写一个 token 的 KV 向量时**：`heads×dim` 整坨一起写；真正要选的是前两维 `(block_id, offset)`。

`slot` 不是随便叫的「行号」，而是：

> **在已经固定第0维（K或V）、第1维（层）之后，把第2维×第3维（`block_id` × `offset`）拉直得到的全局线性下标。**  
> 每个 `slot` 对应一个完整的 `(heads, dim)` 向量落点。

```text
slot = block_id * block_size + offset
```

和二维下标拉直同一套：`a[i][j] → i * 列数 + j`，这里 `(i,j)=(block_id, offset)`，列数=`block_size`。

```text
block_size=4 时，(第2维, 第3维) 拉直：

  block_id=0:  slot  0  1  2  3
  block_id=1:  slot  4  5  6  7
  block_id=2:  slot  8  9 10 11
  ...

store_kvcache：按 slot 选中「第2、3维」上的那个位置，写入该处的 heads×dim
```

```text
例：token t5，block_table=[0,1]，block_size=4
  逻辑段 i=1，offset=1，block_id=1
  slot = block_id * block_size + offset = 5
  → 在「本层 K（或 V）」里，第2维=1、第3维=1 的那个格子
  → 写入那里的 (heads, dim) 向量
```

本步临时参数（`Context`，`prepare_*` 填、`Attention` 读）：

| 名字 | 含义 |
|------|------|
| `slot_mapping` | 本步每个要算的 token → 线性 slot（写 cache 用） |
| `block_tables` | 各 seq 的 `block_table` pad 成 GPU 张量（读历史用） |

### 6. 三条容易混的界线

```text
① blocks[i]（卡片） ≠ kv_cache 的第 i 格（张量）
   前者是登记信息；后者是真 KV。门牌号相同。

② seq.block(i)（token 列表） ≠ block_table[i]（物理号）
   前者是「有什么」；后者是「存在哪」。

③ free_block_ids 空闲 ≠ hash 表无记录
   人走了可以回 free，指纹还可能留着 → prefix「命中但还要从 free 认领」。
```

读到这里再往下看 2.1 / 2.2：`can_allocate` 在查「内容能不能复用 + free 够不够」；`allocate` 在改 `free`/`used`/`ref_count` 并填写 `block_table`。

下面先跟 **一条冷启动请求 A 的第一次 prefill step**（2.1→2.4），再接 **第一次 decode**（2.5），最后多请求 / prefix（2.6）。

---

## 2.1 `generate` → `add_request`：入队，还不租块

**调用位置：** `LLMEngine.generate` 开头的 for 循环；每个 prompt 调一次。

```python
# LLMEngine.generate
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)     # 每个 prompt 变成一个 Sequence，丢进 waiting
while not self.is_finished():
    output, num_tokens = self.step() # 循环 step，直到所有序列 FINISHED
```

```python
# LLMEngine.add_request
if isinstance(prompt, str):
    prompt = self.tokenizer.encode(prompt)  # 字符串 → token id 列表
seq = Sequence(prompt, sampling_params)     # 新建请求：block_table=[]，尚未租块
self.scheduler.add(seq)    # 仅 waiting.append(seq)；不调用 BlockManager
```

`Scheduler.add` 只是 `waiting.append`，**不会**调用 `BlockManager.allocate`。

此时（省例）：

```text
seq.token_ids = [t0..t6]
seq.num_tokens = 7
seq.block_table = []          # 尚无租约
seq.num_cached_tokens = 0
seq.num_scheduled_tokens = 0
seq.status = WAITING
scheduler.waiting = [seq]
scheduler.running = []
free_block_ids = [0,1,2,...,N-1]   # 与 §1 结束时相同
```

**和初始化的衔接：** 用的仍是 §1 开好的同一块 `kv_cache`、同一个 `BlockManager`；这里只是多了一个还没租房的请求对象。

---

## 2.2 `step` ①：`Scheduler.schedule` — 决定跑谁、租不租块

**调用位置：** `LLMEngine.step` 第一行。

```python
# LLMEngine.step
seqs, is_prefill = self.scheduler.schedule()                 # ← 这里
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

`schedule` 优先尝试 **prefill**（`waiting` 非空）；若本轮一个 prefill 都没排上，再走 **decode**（`running`）。  
返回值：`(本批 seqs, is_prefill: bool)` —— 后面 `run` / `postprocess` 都靠这个 bool 选分支。

### Prefill 分支源码（与仓库一致）

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    scheduled_seqs = []          # 本轮最终要交给 run 的序列列表
    num_batched_tokens = 0       # 本轮已累计要算的 token 数（受 max_num_batched_tokens 限制）
    # ---------- prefill：优先消化 waiting 队列 ----------
    while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.waiting[0]    # wating全都是prefill请求，只看队头，先不 pop（可能本轮 chunk 吃不完，还留在 waiting）
        remaining = self.max_num_batched_tokens - num_batched_tokens  # 本轮还能再塞多少 token
        # remaining 从哪来：
        #   max_num_batched_tokens ← Config（默认 16384），「单步 prefill 最多算多少 token」的硬顶
        #   num_batched_tokens     ← 本轮 while 里已经排进去的 token 累计（初始 0）
        #   remaining              ← 额度还剩多少；排一条就减（见下面 += num_scheduled_tokens）
        # 若一条 prompt 比 remaining 还长 → num_scheduled_tokens = remaining（chunked）
        # → 本轮算不完，seq 留在 waiting；下轮 start=num_cached_tokens 接着算
        if remaining == 0:
            break                # batch token 额度用尽，不再排更多 prefill
        if not seq.block_table:
            # 新请求：还没有租约 → 先探测能不能租、能命中多少 prefix 满块
            num_cached_blocks = self.block_manager.can_allocate(seq)
            if num_cached_blocks == -1:
                break            # free 不够，本轮排不上这个（以及后面的）waiting
            # 真正还要算的 prompt token 数 = 总长 − 已命中前缀 token
            num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
        else:
            # 已有租约（例如 chunked prefill 第二段）：只算还没 cached 的尾巴
            num_tokens = seq.num_tokens - seq.num_cached_tokens
        # 额度不够装下整段，且本轮已经排了别人 → 不允许再开新的 chunked 请求
        # （只允许「本轮第一个」seq 做 chunked prefill）
        if remaining < num_tokens and scheduled_seqs:
            break
        if not seq.block_table:
            # 探测通过后真租：写 block_table，改 free/used
            self.block_manager.allocate(seq, num_cached_blocks)
        # 本步实际要算多少：可能被 remaining 截断（chunked）
        seq.num_scheduled_tokens = min(num_tokens, remaining)
        num_batched_tokens += seq.num_scheduled_tokens
        # 若本轮算完后「已 cached + 本步」刚好等于 prompt 全长 → 晋升到 running
        if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
        scheduled_seqs.append(seq)   # chunked 未吃完时：仍在 waiting，但也在本批里跑
    if scheduled_seqs:
        return scheduled_seqs, True  # True = 本轮是 prefill
    # decode 分支见 2.5
    ...
```

对新请求：`block_table==[]`，因此会依次：

1. `BlockManager.can_allocate(seq)` —— 能不能租、能复用几个满块
2. （若不是 -1）`BlockManager.allocate(seq, num_cached_blocks)` —— 真写 `block_table`
3. 设 `num_scheduled_tokens`；若本轮能吃完 prompt，则 `waiting → running`

```text
schedule(prefill) 调用链：

LLMEngine.step
  └─ Scheduler.schedule
       ├─ BlockManager.can_allocate(seq)   # 探测：返回 -1 或可复用满块数
       └─ BlockManager.allocate(seq, n)    # 写 seq.block_table；改 free/used
       返回 (seqs, True)
```

### 2.2.1 `BlockManager.can_allocate`（探测，不改租约）

**调用位置：** 仅当 `not seq.block_table` 时，由 `schedule` prefill 分支调用。

> 若对 `blocks` / `free` / `seq.block(i)` / `block_table` 仍糊：先回看本节开头的 **「关键对象一览」**。

#### 先弄清：`seq.block(i)` 是什么？和 KV 池里的 block 啥关系？

nano-vllm 里「block」这个词会出现三次，**共用同一个 `block_size`，但是三层东西**：

| 名字 | 在哪 | 到底是什么 |
|------|------|------------|
| **逻辑块** `seq.block(i)` | CPU，`Sequence` | prompt/生成序列上，第 i 段的 **token id 列表** |
| **物理块** `block_id` | GPU，`kv_cache[..., block_id, :, ...]` | 大池里一格，能装 **最多 `block_size` 个 token 的 K/V** |
| **页表项** `block_table[i]` | CPU，每条请求一份 | 把「逻辑段 i」映射到「物理号 `block_id`」 |

关系一句话：

> `seq.block(i)` = 「这段有哪些 token」；  
> 物理 block = 「这段 token 的 K/V 存在哪一格」；  
> `block_table[i]` = 两者之间的租约。  
> **有关系，但是「内容切片」≠「显存格子」。**

```text
同一 block_size=4 对齐：

逻辑（token 内容）          物理（KV 显存，§1 allocate 开的）
seq.block(0)=[t0..t3]  ──block_table[0]=7──►  大池里 block_id=7 那一格
seq.block(1)=[t4..t6]  ──block_table[1]=1──►  大池里 block_id=1 那一格

can_allocate / hash 只看左边（token 列表）
store_kvcache / Attention 只写/读右边（物理格）
中间靠 block_table 翻译
block_table[0] = x, x表示seq.block(0)这些token对应的kv cache存在物理池子block的x位置
```

#### size 关系（都钉在同一个 `block_size` 上）

配置里只有一个旋钮：`Config.kvcache_block_size`（默认 256，且要求 `%256==0`）。  
`Sequence.block_size`、物理池第 3 维、`BlockManager.block_size` **都是它**。

| 对象 | 尺寸怎么量 | 和 `block_size` 的关系 |
|------|------------|------------------------|
| 逻辑块 `seq.block(i)` | **token 个数** | 满块恰好 `block_size` 个；末块 `1 .. block_size` |
| 物理块（一层的 K 或 V） | 张量形状 | `[block_size, num_kv_heads, head_dim]` → **行数 = block_size** |
| 一个 `block_id` 整份代价 | 字节 | `block_bytes = 2 × L × block_size × heads × dim × itemsize`（§1.3） |
| 一条序列要几块 | 块数 | `num_blocks = ceil(num_tokens / block_size)` = `len(block_table)` |
| 全池有几格 | 块数 | `N = num_kvcache_blocks`（预算 // block_bytes），与单条序列无关 |

省例对齐看一眼：

```text
block_size = 4
token 数 = 7

逻辑：
  block(0) 长度 4   block(1) 长度 3
  num_blocks = 2  →  allocate 后 len(block_table) 也必须是 2

物理（每个 block_id 一格）：
  形状固定能装 4 行 KV
  格 7：写满 4 行（t0..t3）
  格 1：只写前 3 行（t4..t6），第 4 行空着  ← 内部碎片 ≤ 1 个 block

所以：
  逻辑满块 token 数  =  物理块容量（行数）  =  block_size
  逻辑末块 token 数  ≤  block_size   （物理格仍按满容量占着）
  1 个逻辑段  ↔  1 个物理 block_id   （经 block_table[i]）
```

一句话：**个数上 1:1 对齐；字节上物理块贵得多（还要 ×层 ×K/V ×头维），逻辑块只是 int 列表。**

`seq.block(i)` 源码（`sequence.py`）——只切 token，不碰 GPU：

```python
# Sequence
@property
def num_blocks(self):
    # 这条序列按 block_size 切成几段（向上取整）
    return (self.num_tokens + self.block_size - 1) // self.block_size

def block(self, i):
    # 取出「逻辑第 i 段」的 token id 子列表
    assert 0 <= i < self.num_blocks
    return self.token_ids[i * self.block_size : (i + 1) * self.block_size]
```

省例 `block_size=4`，`token_ids=[t0..t6]`（共 7 个）：

```text
num_blocks = ceil(7/4) = 2

逻辑段 i=0：seq.block(0) = token_ids[0:4] = [t0, t1, t2, t3]   ← 满块，长度 4
逻辑段 i=1：seq.block(1) = token_ids[4:8] = [t4, t5, t6]       ← 末块，长度 3（不到 4）

注意：
  · 这里的 i = 0,1 是「逻辑段号」
  · 此时 block_table 还是 []，还没有物理号
  · 以后若 allocate 得到 block_table=[0,1]，才是：逻辑段0→物理块0，逻辑段1→物理块1
  · seq.block(i) 始终只回答：「这段里有哪些 token id」，不回答「存在哪个物理格」
```

`can_allocate` 要用它，是因为 prefix 匹配比的是 **token 内容**（对满块算 hash），所以先按逻辑段把 token 切出来——还没轮到碰 §1 开好的那块 `kv_cache`。

#### `can_allocate` 源码

```python
def can_allocate(self, seq: Sequence) -> int:
    h = -1                                   # 链式 hash 的前缀种子；第一块无前缀用 -1
    num_cached_blocks = 0                    # 能命中 prefix 的「满块」个数
    num_new_blocks = seq.num_blocks          # 先假设整段都要新租（后面命中共享再减）
    # 只扫满块：末块可能不满，不参与 prefix 匹配（故 range 到 num_blocks-1）
    # 对于每一个块，看这个块是否已经被缓存
    for i in range(seq.num_blocks - 1):
        token_ids = seq.block(i)             # 逻辑第 i 段的 token 列表（满块时长度=block_size）
        h = self.compute_hash(token_ids, h)  # 带上前一块 hash，做成链式指纹
        block_id = self.hash_to_block_id.get(h, -1)
        # 表里没有，或表里有但内容对不上 → 前缀在此断开
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break
        num_cached_blocks += 1               # 内容可复用（hash+token 对上了）
        if block_id in self.used_block_ids:
            # 「内容命中」≠「不占 free」：见下方两种命中
            # 仅当块仍在 used（有人正引用）时，共享不额外占 free
            num_new_blocks -= 1
    if len(self.free_block_ids) < num_new_blocks:
        return -1                            # 资源不够：告诉 schedule 本轮别排这个
    return num_cached_blocks                 # ≥0：可复用满块数（冷启动常为 0）
```

**为何 hash 命中后还要看 `used_block_ids`？不重复。**

两层判断问的是两件不同的事：

| 判断 | 回答的问题 |
|------|------------|
| `hash` 命中且 `token_ids` 一致 | **内容**还能复用吗？（prefix 对得上） |
| `block_id in used_block_ids` | **现在有没有人占着**这格？（要不要从 free 里再领） |

关键背景：请求 `deallocate` 时，`ref_count→0` 只会把 `block_id` **放回 free**，**不一定立刻清掉** `hash_to_block_id` 里的指纹。  
于是会出现「**内容还在缓存表里，但物理块已经闲置在 free 上**」——这正是 prefix cache 的常见状态：人走了，货还留着，直到某次 `_allocate_block` 真正复用该格才清 hash。

因此命中后有两条路（与后面 `allocate` 对称）：

```text
① 命中 + 已在 used（别人还在用）
   → 只需 ref_count++，不从 free 再拿
   → num_new_blocks -= 1

② 命中 + 不在 used（块在 free 上「晒着」，内容仍有效）
   → allocate 时仍要从 free_block_ids.remove(该 block_id)
   → 仍然占 1 个 free 名额，所以这里不减 num_new_blocks
```

对应 `allocate`：

```python
if block_id in self.used_block_ids:
    block.ref_count += 1          # 路径①：纯共享
else:
    block.ref_count = 1
    self.free_block_ids.remove(block_id)  # 路径②：从 free「认领」回 used
    self.used_block_ids.add(block_id)
```

所以：`num_cached_blocks += 1` 记的是「能复用内容」；`num_new_blocks -= 1` 记的是「这次不用额外消耗 free」。只有①两者同时成立。

在做什么：

1. `num_new_blocks = seq.num_blocks`：先假设整段都要新租（省例=2）。
2. `for i in range(num_blocks - 1)`：只检查**满块**能否命中 prefix；末块不参与。省例只跑 `i=0`，此时 `seq.block(0)==[t0..t3]`。
3. 对满块算链式 hash，查 `hash_to_block_id`；对不上就 `break`。
4. 命中且该 `block_id` 已在 `used`：`num_new_blocks -= 1`（可共享，少占一个 free）。
5. `free` 不够 → 返回 `-1`（本请求本轮排不上）；否则返回可复用满块数。

冷启动手算：

```text
hash_to_block_id 为空 → i=0 查不到 → break
num_cached_blocks=0, num_new_blocks=2
free 够 → 返回 0
```

> 返回值是 **可复用满块个数**，不是 bool。`-1` 才表示「资源不够，schedule 应 break」。

### 2.2.2 `BlockManager.allocate`（真租，写 `block_table`）

**调用位置：** `can_allocate` 成功后，仍在 `schedule` 里、且仍是 `not seq.block_table`。

```python
def allocate(self, seq: Sequence, num_cached_blocks: int):
    assert not seq.block_table               # 只允许新请求调一次；已有租约走别的路径
    h = -1
    # ----- 前段：命中的 prefix 满块 → 挂到同一物理 block_id（共享）-----
    for i in range(num_cached_blocks):
        token_ids = seq.block(i)             # 同 2.2.1：逻辑第 i 段的 token 列表，不是物理号
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id[h]  # can_allocate 已保证存在
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1             # 已有人用：只加引用，不从 free 再弹
        else:
            # prefix 常见状态：hash 表里有、但 ref_count 曾归零、格在 free 上闲着
            # → 从 free 挪回 used，认领同一物理格（KV 内容若未被覆盖仍可复用）
            block.ref_count = 1
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
        seq.block_table.append(block_id)     # 租约第 i 项 = 共享物理号
    # ----- 后段：没命中的逻辑块 → 从 free list 新租 -----
    for i in range(num_cached_blocks, seq.num_blocks):
        seq.block_table.append(self._allocate_block())
    # 「已 cached」= 命中前缀覆盖的 token 数（冷启动为 0；不是「已写入大池」）
    seq.num_cached_tokens = num_cached_blocks * self.block_size
```

**为何 `for i in range(num_cached_blocks)` 一定从逻辑段 0 起？**

不是「物理 `block_id` 必须从 0 起」，而是：**能共享的只可能是序列开头的一段连续前缀**。

1. **名字就叫 Prefix Caching**：共享的是 system prompt 等**从第一段开始相同**的内容，不是序列中间某段碰巧一样。
2. **`num_cached_blocks` 的定义就是「从头连续命中了几块」**  
   `can_allocate` 从 `i=0` 扫起，一旦某块 miss 就 `break`，返回的是前缀长度。不会出现「第 0 段 miss、第 2 段却算命中」。
3. **hash 是链式的**，后一块指纹绑着前一块：

```python
h = -1
h = compute_hash(block0_tokens, h)   # 只含段0
h = compute_hash(block1_tokens, h)   # 含「段0+段1」整段前缀
```

所以查表命中「段1 的 hash」时，语义上已经要求段0 也相同。中缀/后缀单独匹配，这套表做不到，也不打算做。

```text
可以：  [共享段0][共享段1][新租段2]...     ← allocate 前段 for i in 0..num_cached-1
不行：  [新租][碰巧同内容][新租]           ← 不会把中间那段当 prefix 命中

物理号可以是 [7, 1, 3]，完全不必是 0,1,2
只是逻辑下标 i 必须从 0 连续数到 num_cached_blocks-1
```

#### `_allocate_block`：真正「从 free 领一张新门牌」

`allocate` / `may_append` 要**新租**一格时，都落到这个私有函数。  
它和 `allocate(seq, ...)` 不是一回事：

| | `allocate(seq, num_cached_blocks)` | `_allocate_block()` |
|--|-----------------------------------|---------------------|
| 粒度 | 整条序列的租约（可能多格 + prefix 共享） | **恰好 1 个**物理 `block_id` |
| 谁调用 | 仅 `schedule` prefill（新请求） | `allocate` 后段；`may_append` |
| 做的事 | 填整张 `block_table` | 改登记本：free→used，reset 卡片 |

完整源码：

```python
def _allocate_block(self) -> int:
    # 1) 从空闲队列头拿一个门牌号（FIFO）
    block_id = self.free_block_ids.popleft()
    block = self.blocks[block_id]
    assert block.ref_count == 0              # 空闲格不应还有人引用

    # 2) 若这格以前当过 prefix 缓存：旧指纹作废（内容即将被新请求覆盖）
    if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
        del self.hash_to_block_id[block.hash]

    # 3) 卡片清零，准备给新租约用
    block.reset()                            # ref_count=1，hash=-1，token_ids=[]
    self.used_block_ids.add(block_id)        # 记入「正在用」
    return block_id                          # 调用方再 append 进 seq.block_table
```

逐步在登记本上发生什么（示意 `free=[0,1,2,...]`）：

```text
调用前：
  free_block_ids = [0, 1, 2, ...]
  used_block_ids = {...}
  blocks[0] 可能还留着旧 hash（人已走、格在 free 上「晒着」）

_allocate_block() 一次之后：
  free_block_ids = [1, 2, ...]      # 0 被 pop 走
  used_block_ids 加上 0
  blocks[0].ref_count = 1
  blocks[0].hash = -1，token_ids=[] # 旧前缀指纹清掉
  返回 0

注意：
  · 这里只改 CPU 登记本，不 torch.empty，也不写 KV
  · GPU 大池里「格 0」的旧数值可能还在，但逻辑上已当作未初始化；
    正式内容要等后面 store_kvcache 写入
```

谁调用它：

```text
allocate 后段：
  for i in range(num_cached_blocks, seq.num_blocks):
      seq.block_table.append(self._allocate_block())
  # 冷启动 num_cached=0、num_blocks=2 → 连调两次，得到 [0,1]

may_append（decode）：
  if len(seq) % block_size == 1:
      seq.block_table.append(self._allocate_block())
  # 需要新块时再领一张挂到表尾
```

和「prefix 命中后从 free 认领」的区别（`allocate` 前段 `else` 分支）：

```text
命中且不在 used：
  free_block_ids.remove(特定 block_id)   # 按号从中间拿走，保留旧 hash/token_ids
  ref_count = 1                          # 复用内容，不 reset

_allocate_block：
  popleft 任意队头                    # 不关心内容，当新格用
  清 hash + reset                      # 旧前缀作废
```

主线 `num_cached_blocks=0`：`allocate` 第一个 for 不跑；第二个 for 跑两次 `_allocate_block`。

```text
之前：free=[0,1,2,...]  block_table=[]
之后：free=[2,3,...]    block_table=[0,1]
      num_cached_tokens = 0   # 注意：这里「cached」指 prefix 命中的 token 数，冷启动仍是 0
```

回到 `schedule`：`num_tokens = 7 - 0 = 7`，设 `num_scheduled_tokens=7`（假设 batch 额度够），因 `0+7==7`，seq 进入 `running`。

**① 结束返回：** `seqs=[A], is_prefill=True`。  
租约有了，**大池数值仍空**——写 KV 是下一步 `run` 的事。

```text
① 结束后状态：
  block_table = [0, 1]
  num_scheduled_tokens = 7
  status = RUNNING
  waiting=[], running=[A]
  GPU 大池：格子 0、1 已「划给 A」，但内容尚未写入
```

---

## 2.3 `step` ②：`ModelRunner.run` — 算、写 cache、采样

**调用位置：** `LLMEngine.step` 第二行：

```python
token_ids = self.model_runner.call("run", seqs, is_prefill)
```

进入 `model_runner.py`：

```python
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    # 按本轮是 prefill 还是 decode，组装 input / slot_mapping，并 set_context
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None  # 仅 rank0 采样
    logits = self.run_model(input_ids, positions, is_prefill)             # 前向（内含写 KV）
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()                      # 清掉本步 Context，避免泄漏到下一 step
    return token_ids                     # 交给 step → postprocess
```

```text
run 内部链：

ModelRunner.run(is_prefill=True)
  ├─ prepare_prefill → set_context(...)     # 算 slot_mapping 等
  ├─ run_model → model.forward
  │     └─ 每层 Attention.forward
  │           ├─ store_kvcache(..., slot_mapping)   # 写入本层大池切片
  │           └─ flash_attn_varlen_func(...)
  ├─ Sampler → [t7]
  └─ reset_context()
```

主线 `is_prefill=True` → 先 `prepare_prefill`。

### 2.3.1 `prepare_prefill`：给「本步前向」准备输入和写地址

**调用位置：** `run` 里 `is_prefill=True` 时。  
**不改** `block_table`（租约在 schedule 里已经定好了），只**读**租约，算出本步该喂什么、K/V 写到哪。

---

#### 这一函数到底在干什么？（先忘掉源码）

`schedule` 已经决定：本批有哪些 `seqs`、每条要算多少 token。  
接下来模型要前向算结果，并且写kv cache，需要两样东西：

```text
① 喂给模型的输入：哪些 token id、每个在序列里的位置
② 算完 K/V 之后写到大池的哪里（每个 token 一个 slot）
```

`prepare_prefill` **只做准备，不算模型**。准备完后 `set_context(...)`，后面 Attention 再真正算和写。

可以记成三句话：

```text
1. 把本批要算的 token 拼成一条长 input_ids（多请求就首尾相接）
2. 为每个 token 算出它该写入的 slot，放进 slot_mapping（和 input 一一对应）
3. 若有多条变长序列，用 cu_seqlens（前缀和）告诉 Flash Attention算子的API：「哪一段属于谁」
```

---

#### 只用一条请求，把结果先算出来（省例）

已知（来自前面 schedule / allocate）：

```text
block_size = 4
token_ids  = [t0,t1,t2,t3,t4,t5,t6]   # 7 个
block_table = [0, 1]                  # 逻辑段0→物理格0，逻辑段1→物理格1
num_cached_tokens = 0                 # 冷启动，从头算
num_scheduled_tokens = 7              # 本步把 7 个都算完
```

**第一步：本步要算哪一段？**

```text
start = num_cached_tokens = 0
end   = start + num_scheduled_tokens = 7
→ 算 token 下标 [0, 7) 即 t0..t6
```

**第二步：拼输入**

```text
input_ids = [t0, t1, t2, t3, t4, t5, t6]
positions = [ 0,  1,  2,  3,  4,  5,  6]   # 每个 token 在序列里的绝对位置
```

只有 1 条序列时，前缀和很简单：

```text
cu_seqlens_q = [0, 7]    # 从下标 0 到 7，整段都是这一条
cu_seqlens_k = [0, 7]    # 冷启动：无「已在 cache 的前缀」，K 与 Q 一样长
```

（若有 Prefix 命中，会出现 K 侧更长、且必须带 `block_tables`——见下文「旁路讲清」。）

**第三步：每个 token 写到哪个 slot？**

回顾：`slot = block_id * block_size + 格内行号`

```text
token  全局下标  逻辑段i  block_id  格内行  slot
t0        0        0         0        0      0
t1        1        0         0        1      1
t2        2        0         0        2      2
t3        3        0         0        3      3
t4        4        1         1        0      4
t5        5        1         1        1      5
t6        6        1         1        2      6
```

所以：

```text
slot_mapping = [0, 1, 2, 3, 4, 5, 6]

把每个token的K或者V cache，映射到了GPU大池的对应全局位置
```

和 `input_ids` **下标对齐**：本步算出的第 j 个 (K,V) → 写入 `slot_mapping[j]`。

#### 为什么「这个 token 的 KV」会恰好落到「这个 slot」？

不是碰巧，而是 **三处约定用了同一套地址公式**，前后对得齐。

**约定 1：大池怎么排（§1.3）**

一层 `k_cache` 形状 `[N, block_size, heads, dim]`。  
按行主序把「所有格的所有行」拉直后，第 `slot` 行就是：

```text
slot = block_id * block_size + offset
⇔ 落在物理格 block_id、格内第 offset 行
```

`store_kvcache` 写的就是这个线性行：`cache[slot * D + ...]`。

**约定 2：序列 token 归哪一格（allocate / block_table）**

对序列里全局下标为 `t` 的 token，分页规定：

```text
逻辑段 i = t // block_size
格内行 o = t %  block_size
物理格   = block_table[i]          # 租约已在 schedule 定好
```

因此 **这个 token 的 K/V「应该」住的地方** 就是：

```text
该住的 slot = block_table[t // block_size] * block_size + (t % block_size)
```

**约定 3：prepare_prefill 按约定 2 填表（不是另算一套）**

本步 `input_ids` 里第 j 个，对应全局下标 `t = start + j`（冷启动 start=0 则 t=j）。  
填 `slot_mapping` 的循环，等价于对每个这样的 `t` 套约定 2 的公式，所以：

```text
slot_mapping[j]  ==  block_table[t // block_size] * block_size + (t % block_size)
                 ==  「token t 按分页规则该住的那一行」
```

**写回时对齐**

```text
前向算出：第 j 个 token 的 (k,v)     ← 就是 token t 的 KV
store：   写入 slot_mapping[j]        ← 正好是约定 2 给 t 留的行
```

串起来：

```text
tokenize 位置 t
  → 逻辑段 / 格内行（÷、% block_size）
  → block_table 换成物理 block_id
  → ×block_size+offset 得到 slot
  → prepare 写入 slot_mapping[j]
  → store_kvcache 把第 j 个 KV 写进该 slot

所以「恰好对应」= 布局公式、租约、填表、写入四步共用同一地址定义。
```

若 `block_table` 错了，或 `slot_mapping` 和 `input_ids` 顺序对不齐，就会写到别人的格——分页正确性全靠这条链不断。

**第四步：交给后面**

```text
set_context(... slot_mapping ..., block_tables=None)
return input_ids, positions
→ run_model 用 input 前向
→ Attention 里 store_kvcache 按 slot_mapping 写入大池
```

冷启动到这里，**主线已经走通**。下面源码只是把上述步骤写成循环（还要兼容多请求 / prefix / warmup）。

---

#### `slot` 和 `block`（对照表）

```text
slot = block_id * block_size + offset
```

| 词 | 一句话 |
|----|--------|
| `block_id` | 物理哪一格（门牌） |
| `offset` | 格内第几行 |
| `slot` | 两者合成的线性行号 |
| `slot_mapping` | 本步 `input_ids[j]` 对应的那个 slot |

```text
物理格拉直后（block_size=4）：
  格0: slot 0,1,2,3
  格1: slot 4,5,6,7
  格2: slot 8,...
```

`slot_mapping = []` 只是空列表累加器，后面往里填数字；不是特殊开关。

---

#### 源码按「意图」读（A→B→C→D）之前：先认清这些计数

`prepare_prefill` 里的 `start` / `seqlen_q` / `end` / `seqlen_k` **不是** Sequence 上的字段，  
而是本函数根据下面两个「进度条」算出来的临时变量。  
先把进度条钉死，这段代码才读得懂。

##### `Sequence` 上两个进度字段

| 字段 | 初始值 | 含义（人话） |
|------|--------|--------------|
| `num_cached_tokens` | `__init__` 里 `= 0` | **已经有 KV、注意力可以当「历史」用的 token 个数**（从序列开头数） |
| `num_scheduled_tokens` | `__init__` 里 `= 0` | **本轮 `step` 里 schedule 决定要新算多少个 token**（算完就清零） |

它们**什么时候被改**：

```text
num_cached_tokens
  创建：Sequence(...) → 0
  写入① allocate（prefix 命中时）：
        = num_cached_blocks * block_size
        # 共享前缀「视同已在 cache」，冷启动仍是 0
  写入② postprocess 每轮：
        += num_scheduled_tokens
        # 本步刚算完并（将）写入池的那些，记进「已 cached」
  清零：deallocate 时 → 0

num_scheduled_tokens
  创建：Sequence(...) → 0
  写入：schedule 里
        prefill: = min(本步还该算的 prompt 剩余, batch 额度)
        decode:  = 1
  清零：postprocess 末尾 → 0（本轮用完即清）
```

用时间线看（冷启动 7 token prompt，一次算完）：

```text
入队后：
  num_tokens=7, num_cached_tokens=0, num_scheduled_tokens=0

schedule(prefill)：
  num_scheduled_tokens = 7          # 「这一步请算 7 个」
  num_cached_tokens 仍是 0

prepare_prefill / run：
  用这两个数决定算 [0,7)、填 slot……

postprocess：
  num_cached_tokens += 7 → 7        # 「这 7 个已经有 KV 了」
  num_scheduled_tokens = 0          # 本轮任务单作废
  append t7 → num_tokens=8

下一轮 schedule(decode)：
  num_scheduled_tokens = 1          # 「这一步请算 1 个（last_token）」
```

有 Prefix 时多一步：`allocate` 先把 `num_cached_tokens` 设成前缀长度（如 4），  
`schedule` 再设 `num_scheduled_tokens = 7-4 = 3`（只调度后缀）。

##### 本段临时变量（由上面两个推出来）

```text
start    = num_cached_tokens
         = 序列从头数，已经「有历史 KV」的长度
         = 本步新算的起点下标

seqlen_q = num_scheduled_tokens
         = 本步要新算多少个（Q 的个数；也是本步要 store 的个数）

end      = start + seqlen_q
         = 本步算完后，有效上下文的右端（开区间）
         = 本步新算区间是 token 下标 [start, end)

seqlen_k = end
         = 注意力时 K 侧要覆盖的长度（从 0 到 end）
         = 前缀历史 + 本步新算
```

图示（Prefix：前 4 个已 cached，本步调度 3 个）：

```text
token 下标:  0  1  2  3     |    4  5  6
             <-- cached -->|<- 本步新算 ->|
             start=4         seqlen_q=3
                             end=7
             <-------- seqlen_k=7 -------->

input_ids 只含 [4,5,6]
Q 长度 3；K 需要看见 0..6，所以长度 7
```

冷启动则 `start=0`，三段数字变成：`seqlen_q=7, end=7, seqlen_k=7`。

---

#### 源码按「意图」读（A→B→C→D）

```python
def prepare_prefill(self, seqs: list[Sequence]):
    input_ids, positions = [], []
    cu_seqlens_q, cu_seqlens_k = [0], [0]
    max_seqlen_q = max_seqlen_k = 0
    slot_mapping = []
    block_tables = None

    for seq in seqs:                          # 本批可能有多条；先只想「一条」也行
        # --- A. 用进度条算出：本步新算区间、Q/K 各多长 ---
        start = seq.num_cached_tokens         # 已有历史 KV 的长度 = 新算起点
        seqlen_q = seq.num_scheduled_tokens   # schedule 本轮要新算的个数
        end = start + seqlen_q                # 新算区间 [start, end)
        seqlen_k = end                        # K 覆盖 [0, end)；有前缀时 > seqlen_q
        # 冷启动：start=0 → seqlen_k == seqlen_q
        # Prefix：start>0 → seqlen_k > seqlen_q（详见下文「旁路讲清」）

        # --- B. 拼进本批大向量（多条就首尾相接）---
        # 把本条 seq 本步要新算的 token 追加到「批级」大列表里
        # 多条请求时：A 的一段 + B 的一段 + … 拍成一条长向量，后面靠 cu_seqlens 划界
        input_ids.extend(seq[start:end])           # token id，对应本步要算的 [start,end)
        positions.extend(range(start, end))        # 每个 token 在序列里的绝对位置（RoPE 用）

        # 前缀和：记下「到目前为止，批里累计了多少 Q/K token」
        # 例：先 A 的 seqlen_q=3，再 B 的 seqlen_q=2 → cu_seqlens_q = [0, 3, 5]
        # FA 用 [cu[i], cu[i+1]) 取出第 i 条在长向量里的那段
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)  # 有 prefix 时 seqlen_k 更大，累计也会更大
        max_seqlen_q = max(seqlen_q, max_seqlen_q)         # 本批最长的单条 Q（FA 参数）
        max_seqlen_k = max(seqlen_k, max_seqlen_k)         # 本批最长的单条 K

        if not seq.block_table:               # warmup：假 Sequence 没租约，无法算物理 slot
            continue                          # 仍拼了 input（为了摸峰），但不填 slot_mapping

        # --- C. 按逻辑块，把 [start,end) 中每个位置的 token 映射成 slot ---
        # 目标：为本步每个新算 token 算出物理池地址，按顺序 append 进 slot_mapping
        # 使得 slot_mapping[j] ↔ input_ids 里本条贡献的第 j 个 token
        #
        # 为何按逻辑块拆开算：block_table[i] 物理号可不连续，
        # 不能假定「token 下标连续 ⇒ slot 也全局连续」，必须一段一段换算。
        #
        # 每个 i 都会得到一对 [slot_start, slot_end)，再 extend。
        # 下面两个 if 互相独立，不要读成「第一段只算 start」：
        #   · 第一个 if：只可能改 slot_start（是否从格中起写）
        #   · 第二个 if/else：每个 i 必走其一 → 一定能得到 slot_end
        start_block = start // self.block_size   # 本步起点落在第几个逻辑段
        end_block = (end + self.block_size - 1) // self.block_size  # 开区间右端之上的段号
        # 例：start=0,end=7,block_size=4 → start_block=0, end_block=2 → i=0,1
        for i in range(start_block, end_block):
            # ---- ① 先定本段左端 slot_start ----
            slot_start = seq.block_table[i] * self.block_size   # 该物理格的起点
            if i == start_block:
                # 仅第一段：起点可能在格内中间（chunk/prefix 接着写）
                # 例：start=5, block_size=4 → 5%4=1 → 从本格 offset=1 起
                slot_start += start % self.block_size

            # ---- ② 再定本段右端 slot_end（每个 i 必进下面其一）----
            if i != end_block - 1:
                # 还不是最后一段 → 本段写满整格
                slot_end = seq.block_table[i] * self.block_size + self.block_size
            else:
                # 最后一段（也可以同时是 start_block）→ 写到 end 为止
                # end - i*block_size = 相对「逻辑段 i 起点」要写几个
                # 例：end=7,i=1 → 7-4=3 → 本格写 offset 0,1,2
                slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size

            # ---- ③ 追加 [slot_start, slot_end) ----
            slot_mapping.extend(range(slot_start, slot_end))

        # 手算核对：
        # 冷启动 start=0,end=7,block_table=[0,1],block_size=4
        #   i=0（start_block，且非最后）:
        #     start=0; end=4  → slots 0,1,2,3
        #   i=1（最后一段）:
        #     start=4; end=7  → slots 4,5,6
        # 只有一段 start=5,end=7（都在逻辑段1）:
        #   i=1 同时是 start_block 和最后一段：
        #     slot_start=4+(5%4)=5; slot_end=4+(7-4)=7 → slots 5,6
        #     左端「格中起写」+ 右端「写到 end」两个分支都生效

    # --- D. block_tables 交不交给 Attention？---
    #
    # block_tables 是什么：
    #   把本批每条 seq 的 block_table（逻辑段→物理 block_id）
    #   pad 成 GPU 张量 [batch, max_blocks]，供 FlashAttention 按表从 KV 池里 gather。
    #
    # 「带」是什么意思：
    #   通过 set_context(..., block_tables=...) 传下去；
    #   Attention 里若 context.block_tables is not None，就改用池子里的 K/V + 这张表做注意力。
    #   若不带（保持 None）：Attention 直接用本步前向刚算出的连续 k,v，不查表、不读池历史。
    #
    # 什么时候必须带？
    #   本步 input/Q 只含「新算后缀」，但注意力还要看见「已在池里的前缀」
    #   （Prefix 命中，或 chunked prefill 第二段：start=num_cached_tokens>0）
    #   → 仅靠本步算出的短 k,v 不够，必须按表去池子取前缀。
    #   判据：本批 K 侧总长 > Q 侧总长（cu_seqlens_k[-1] > cu_seqlens_q[-1]）
    #
    # 注意：这是「整批一刀切」，不是按条分别决定！
    #   例：20 条全新 prefill（各条 Q=K）+ 1 条截断后续算（该条 K>Q）
    #   → 总和 K>Q → prepare_block_tables(seqs) 对本批「所有」seq 建表
    #   → Attention 全员走 k_cache + block_table，不只那 1 条
    #   那 20 条冷启动其实也可以用本步连续 k,v，但实现图省事不做混合路径；
    #   它们本步已 store 进自己的格，按表读回的就是刚写入的内容，结果仍正确。
    #
    # 冷启动主线（批里没有任何 K>Q）：Q、K 总和相等 → 不带（None）。
    # 详见下文「旁路讲清」。
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
        block_tables = self.prepare_block_tables(seqs)  # 各 seq.block_table → GPU 张量
    # else: block_tables 保持开头的 None

    # 把本步参数塞进全局 Context，供每层 Attention.get_context() 读取
    # 最后一项 block_tables：None=不按表读池；非空=按表 gather 历史
    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                slot_mapping, None, block_tables)       # True = is_prefill
    return input_ids, positions                         # 交给 run_model 做前向
```

对照省例，C 段循环实际就是：

```text
i=0, block_id=0 → extend 0,1,2,3
i=1, block_id=1 → extend 4,5,6
→ slot_mapping = [0,1,2,3,4,5,6]
```

---

#### 旁路讲清：为什么「有 Prefix 时 K 比 Q 长」？

主线是冷启动（`start=0`），Q、K 一样长，可以暂时不管。  
但代码里到处出现 `seqlen_k` / `cu_seqlens_k` / `block_tables`，背后是同一件事——**Prefix 命中后，本步不必重算前缀的 Q，但注意力仍要「看见」前缀的 K/V**。

**场景**：上一条请求已把某段 system prompt（满块）写进物理池并登记了 hash。  
新请求 B 前 4 个 token 与之相同，`can_allocate` 返回 `num_cached_blocks=1`，`allocate` 后：

```text
block_size = 4
B 的 prompt 共 7 个 token：t0..t6（前 4 个与缓存相同）
block_table = [7, 2]          # 段0 共享物理格7；段1 新租格2
num_cached_tokens = 4         # = 1 * block_size，前缀已在 cache
num_scheduled_tokens = 3      # 本步只新算 t4,t5,t6
```

**本步各长度：**

```text
start    = 4
seqlen_q = 3          # 只对 t4,t5,t6 算 Q（以及它们的 K/V 并写入）
end      = 4+3 = 7
seqlen_k = 7          # 注意力时 K 侧要覆盖 t0..t6 整段上下文

input_ids = [t4, t5, t6]     # 注意：没有 t0..t3
positions = [4, 5, 6]

cu_seqlens_q = [0, 3]        # Q 拼接向量长度 3
cu_seqlens_k = [0, 7]        # 告诉 FA：K 侧等效长度 7
```

**为什么 K 必须是 7 不是 3？**

```text
做 causal attention 时：t4 的 Q 要和 t0..t4 的 K 做点积，
t5 的 Q 要和 t0..t5 的 K，… 不能只看见 t4..t6。

t0..t3 的 K/V 已经在物理格 7 里，本步不重算，
但 FA 仍要通过 block_table 把它们 gather 进来。
```

**和代码分支怎么对上：**

```text
cu_seqlens_k[-1] > cu_seqlens_q[-1]   # 7 > 3
  → block_tables = prepare_block_tables(...)   # 带上 [[7,2], ...]

Attention.forward:
  store_kvcache：只把本步 3 个新 token 的 K/V 写到格2（slot 对应 t4..t6）
  block_tables is not None：
    k,v = k_cache, v_cache                 # 改从池子读
    flash_attn_varlen_func(..., block_table=...)
      Q 长度按 cu_seqlens_q（3）
      K 长度按 cu_seqlens_k（7）= 池里前缀 + 刚写入的后缀
```

**对照冷启动（主线）：**

```text
start=0, seqlen_q=7, seqlen_k=7
cu_seqlens_q[-1] == cu_seqlens_k[-1]
  → block_tables = None
  → FA 直接用本步算出的连续 k,v（本步就算了全部 7 个）
```

一句话：

> **Q 长度 = 本步要新算多少；K 长度 = 注意力需要的完整上下文多长。**  
> Prefix 让「新算」变短，但「上下文」仍含前缀 → `seqlen_k > seqlen_q`，必须带 `block_tables` 从池子补前缀 K/V。

---

#### 多请求时多出来的只有「拼接 + 前缀和」

两条序列 A 算 3 个、B 算 2 个时：

```text
input_ids     = [A0,A1,A2, B0,B1]
slot_mapping  = [sA0,sA1,sA2, sB0,sB1]   # 各用各的 block_table 算
cu_seqlens_q  = [0, 3, 5]                 # FA：[0:3) 是 A，[3:5) 是 B
```

前缀和 = 累计长度表，用来在「拍成一条长向量」后划界。**单条时就是 `[0, 长度]`，可以先忽略。**

---

#### 读完应留下的最小印象

```text
prepare_prefill
  = 拼 input_ids / positions
  + 填 slot_mapping（每个要算的 token → 写到哪个 slot）
  + （多条时）用 cu_seqlens 划界
  + 若本批存在「上下文比本步新算更长」（Prefix）→ 带 block_tables
  + set_context 交给后面的 Attention

不租块、不算模型、不写 GPU 数值；写数值是 store_kvcache 的事。
Q 长 = 本步新算多少；K 长 = 注意力要看见的完整上下文多长。
```

### 2.3.2 `run_model` → 每层 `Attention.forward`

上一节 `prepare_prefill` 只做了两件事：算出 `input_ids/positions`，并把 `slot_mapping` 等塞进 `Context`。  
**真正算模型、写 KV 池，从这里开始。**

#### 调用路线（从上到下钉死）

```text
LLMEngine.step
  └─ ModelRunner.call("run", seqs, is_prefill=True)     # ②
       └─ ModelRunner.run
            ├─ prepare_prefill(...)     # 上一节：set_context 已完成
            ├─ run_model(input_ids, positions, True)     # ← 本节入口
            │    └─ self.model(input_ids, positions)     # Qwen3ForCausalLM.forward
            │         └─ 对每个 Transformer 层 L = 0..L-1：
            │              Qwen3DecoderLayer
            │                └─ Qwen3Attention.forward(positions, hidden_states)
            │                     · qkv_proj → split → 得到本层 q, k, v   # ← K/V 在这里算出来
            │                     · RoPE(q,k)
            │                     └─ o = self.attn(q, k, v)              # 传入已算好的张量
            │                          └─ Attention.forward(q, k, v)     # ← 只负责写池 + FA
            │                               · store_kvcache(k, v, ...)     # 用的是参数 k,v，不是现场算
            │                               · flash_attn_*(...)
            │                     · o_proj(o)
            │         └─ compute_logits(...)             # 最后一层 hidden → logits
            ├─ Sampler(logits) → token_ids               # 如 [t7]
            └─ reset_context()
```

要点：

1. **`Context` 在进模型前已设好**；每一层 `Attention.forward` 都 `get_context()` 读**同一份**（同一 `slot_mapping` / `cu_seqlens` / `block_tables`）。
2. **写池是「每层各写一次」**：层 L 的 `k_cache` 是 §1.3 挂的 `kv_cache[0, L]`；slot 号各层相同，数值各层不同。
3. Prefill 走 `self.model(...)` 普通前向；CUDA Graph 只给 decode 小 batch（见 §1.4）。

#### `run_model` 源码

```python
def run_model(self, input_ids, positions, is_prefill):
    # prefill / eager / 大 batch → 普通前向；小 batch decode 才可能 replay CUDA Graph
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        # model(...) 内部逐层调用到 Attention.forward（见上图）
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        ...  # decode 才可能走 CUDA Graph（§1.4 录的图）
```

#### 落到每一层：先算 q,k,v，再进 `Attention.forward`

K/V **不是**在 `Attention.forward` 里算的。调用顺序是：

```python
# models/qwen3.py — Qwen3Attention.forward
qkv = self.qkv_proj(hidden_states)
q, k, v = qkv.split(...)          # ← 本步这些 token 的 K/V 在这里已经算好
q, k = self.rotary_emb(positions, q, k)
o = self.attn(q, k, v)            # 把算好的 q,k,v 传给下面的 Attention
...
```

```python
# layers/attention.py — Attention.forward(q, k, v)
# 这里的 k,v 是上面传进来的「本步新算」结果（参数），不是空的
def forward(self, q, k, v):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache  # 本层物理池切片
    if k_cache.numel() and v_cache.numel():
        # 把「参数里已经算好的」k,v 按 slot_mapping 写入物理池
        # warmup 时 k_cache 仍是空张量 → numel()==0 → 跳过
        # prefix 段不在本步 k,v 里（没进 input），所以这里只写本步新算的那些 token
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
    if context.is_prefill:
        if context.block_tables is not None:
            # 有 prefix/chunk 后续：注意力要用池子里的完整 K/V，不能只用本步短 k,v
            k, v = k_cache, v_cache
        o = flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=self.scale, causal=True,
            block_table=context.block_tables)   # 冷启动 None：继续用本步连续 k,v
    else:
        o = flash_attn_with_kvcache(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=self.scale, causal=True)
    return o
```

一句话：`store_kvcache` 写的是 **调用参数 `k,v`**（上层 `qkv_proj` 刚算出来的）；`Attention` 自己不负责投影。  
**Q 不进 cache**——只存 K 和 V。

#### 两个 FlashAttention API：各自干什么？

`Attention.forward` 里写池之后，真正算注意力只走两个入口（都来自 `flash_attn` 包）：

| | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
|--|--|--|
| nano-vllm 何时用 | **prefill**（`is_prefill=True`） | **decode**（`is_prefill=False`） |
| 典型 Q | 每条序列多个 query（整段/后缀） | 每条序列 **1 个** query |
| 变长批怎么划界 | `cu_seqlens_q` / `cu_seqlens_k` | 每条一个样本 + `cache_seqlens` |
| K/V 从哪来 | 本步连续 `k,v`，**或** 池子+`block_table` | **始终** 池子+`block_table` |

下面按源码参数逐项对照（callback：这些参数都来自 `prepare_*` → `set_context`）。

##### A. `flash_attn_varlen_func` — prefill 用的「变长打包注意力」

源码调用：

```python
o = flash_attn_varlen_func(
    q, k, v,
    max_seqlen_q=context.max_seqlen_q,
    cu_seqlens_q=context.cu_seqlens_q,
    max_seqlen_k=context.max_seqlen_k,
    cu_seqlens_k=context.cu_seqlens_k,
    softmax_scale=self.scale,
    causal=True,
    block_table=context.block_tables,   # 冷启动 None；Prefix/chunk 后续非空
)
```

**它在解决什么问题？**  
一批里多条序列、长度不同，不想 pad 成矩形大张量浪费算力。做法是：把各条的 Q（以及冷启动时的 K/V）**首尾相接拍成一条长向量**，再用前缀和告诉算子「哪一段属于谁」。

**`cu_seqlens_*`（cumulative sequence lengths）是什么？**

```text
例：两条 prefill，A 本步算 3 个 token，B 本步算 2 个
  拼好的 q 长度 = 5
  cu_seqlens_q = [0, 3, 5]
    → 序列0 用 q[0:3)，序列1 用 q[3:5)

K 侧同理用 cu_seqlens_k。
冷启动无 prefix：每条 seqlen_k == seqlen_q，故 cu_seqlens_k 与 cu_seqlens_q 相同。
有 prefix：K 侧更长（要看见前缀），cu_seqlens_k 的累计更大。
```

`max_seqlen_q` / `max_seqlen_k`：本批里单条最长的 Q/K 长度（内核分配用）。

**`block_table` 两种用法（和前面 `k,v = k_cache` 联动）：**

```text
① 冷启动（block_tables is None）
   k,v = 本步刚算的连续张量（与 q 一样按 cu_seqlens 打包）
   算子：直接对这段连续 K/V 做因果注意力
   （store_kvcache 已把同样内容写入池，供以后 decode/prefix；本步 attn 不必再读池）

② Prefix / chunk 后续（block_tables 非空）
   先执行 k,v = k_cache, v_cache
   算子：Q 仍按 cu_seqlens_q（只有后缀）；
         K/V 按 cu_seqlens_k（前缀+后缀总长），并用 block_table 从分页池 gather
```

省例冷启动一条：

```text
q,k,v 各 7 个 token；cu_seqlens_q=cu_seqlens_k=[0,7]
block_table=None
→ 等价于对 t0..t6 做标准因果 self-attn（变长 API 的「batch=1」特例）
```

省例 Prefix（前 4 已在池，本步算 t4..t6）：

```text
q 长度 3；cu_seqlens_q=[0,3]
cu_seqlens_k=[0,7]          # 注意力要看见整段 7
k,v 换成整池；block_table=[[0,1], ...]
→ Q 只有后缀 3 个，但每个 Q 都能 attend 到池里 t0..t6 的 K/V
```

##### B. `flash_attn_with_kvcache` — decode 用的「带分页 KV cache 的注意力」

源码调用：

```python
o = flash_attn_with_kvcache(
    q.unsqueeze(1),          # [num_seqs, 1, num_heads, head_dim]
    k_cache, v_cache,       # 本层分页池 [num_blocks, block_size, kv_heads, dim]
    cache_seqlens=context.context_lens,   # 每条当前有效长度
    block_table=context.block_tables,     # decode 一定非空
    softmax_scale=self.scale,
    causal=True,
)
```

**它在解决什么问题？**  
Decode 时每条序列本步只有 **1 个新 query**，却要和 **整段历史 K/V** 做注意力；历史散落在不同物理 block 里。这个 API 专为「Q 短、K/V 在 paged cache」设计。

**为何 `q.unsqueeze(1)`？**  
上层传来的 `q` 形状是 `[num_seqs, heads, dim]`（每条一个 token 拍在一起）。  
该 API 要的是 `[batch, seqlen_q, heads, dim]`，decode 的 `seqlen_q=1`，故 `unsqueeze(1)`。

**`cache_seqlens`（即 `context.context_lens`）是什么？**  
`prepare_decode` 里填的 `len(seq)`——**含本步 `last_token` 在内的总长**。  
算子据此知道：每条序列在 cache 里前多少个 token 位置是有效的（结合 `block_table` 去取）。

**`block_table`：**  
形状约 `[batch, max_blocks_per_seq]`，第 i 行 = 第 i 条序列的 `block_table`（不足处 pad）。  
算子按「逻辑 token 下标 → 逻辑块 → 表里物理 block_id → 格内 offset」去读 K/V。

**和 `store_kvcache` 的分工（重要）：**

```text
FlashAttention 官方 API 也可以传入本步的 k,v，让内核顺带 inplace 更新 cache。
nano-vllm 没有走这条：
  1) 先自己 store_kvcache（按 slot_mapping 写好本步 last_token）
  2) 再调 flash_attn_with_kvcache，且不传本步 k,v
  → 该调用在本仓库里 =「只读已写好的分页池做注意力」
```

省例首次 decode（`last_token=t7`，`len=8`，`block_table=[0,1]`）：

```text
1) store：t7 的 K/V → slot 7
2) flash_attn_with_kvcache：
     q = t7 的 query（seqlen_q=1）
     cache_seqlens = [8]          # 有效历史+当前共 8
     block_table = [[0,1]]
     → 从池中按表取 t0..t7 的 K/V，与 q 做因果注意力
```

##### C. 一张对照收束

```text
阶段      API                        Q 从哪来        K/V 从哪来
--------  -------------------------  -------------  --------------------------
冷 prefill  varlen                   本步 input      本步连续 k,v（表=None）
Prefix预填  varlen                   本步后缀 Q      池 + block_table（K 更长）
decode      with_kvcache             last_token 的 Q 池 + block_table + cache_seqlens

共同点：真正「写入池」在 nano-vllm 里都由 store_kvcache 先做完；
        两个 FA API 负责的是「用对的 Q 去 attend 对的 K/V」。
```

#### 先搞清：本步「用来算 K/V 的 token」是哪些？

整条链只认一份输入：`prepare_*` 拼出来的 `input_ids`（以及对应的 `positions`）。  
**出现在 `input_ids` 里的每一个 id，都会在每一层被算出一组 (q,k,v)；其中 k,v 再按 `slot_mapping` 写入池。**

##### 1. 谁决定「算哪些 token」？

```text
schedule
  → 写下 seq.num_scheduled_tokens
  → （prefill）还可能设好 num_cached_tokens（prefix）或沿用 chunk 进度

prepare_prefill（对本条 seq）
  start = num_cached_tokens
  end   = start + num_scheduled_tokens
  input_ids.extend(seq[start:end])     # ← 只有这段进模型
  positions.extend(range(start, end))
  slot_mapping 也只为这段 token 填地址
```

因此：

| 场景 | 进 `input_ids`、会算 K/V 的 token |
|------|----------------------------------|
| 冷启动一次吃完 | 整段 prompt，如 `t0..t6` |
| Prefix 命中 | **只有后缀**，如 `t4..t6`（前缀已在池，不算、不写） |
| Chunked 第二段 | 上一截之后的剩余 prompt |
| Decode | 只有 `last_token` 一个（`prepare_decode`） |

**不会**对「已在 cache 里的前缀」再跑一遍 `qkv_proj`——它们根本没进本步 `input_ids`。

##### 2. 这些 token 上的 K/V 具体怎么算出来？

```text
run_model
  └─ model(input_ids, positions)          # 长度 = 本步 N_tok
       └─ embed_tokens(input_ids)         # 每个 id → hidden 向量，得到 [N_tok, hidden]
       └─ 对每一层 L：
            hidden → Qwen3Attention
              qkv = qkv_proj(hidden)      # 一次线性，再 split
              q, k, v = split(qkv)
              # 此时：
              #   k[j] = 本步第 j 个 input token（即全局下标 start+j）在层 L 的 Key
              #   v[j] = 同位置的 Value
              #   j 与 input_ids / slot_mapping 下标对齐
              q,k = RoPE(positions, q, k)
              o = Attention.forward(q, k, v)  # 写池 + 注意力
```

源码对应（`Qwen3Attention.forward`）：

```python
qkv = self.qkv_proj(hidden_states)     # hidden 来自本步 input_ids 的嵌入+下层输出
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
q = q.view(-1, num_heads, head_dim)
k = k.view(-1, num_kv_heads, head_dim)
v = v.view(-1, num_kv_heads, head_dim)
q, k = self.rotary_emb(positions, q, k)  # positions 与 input 一一对应
o = self.attn(q, k, v)
```

对齐关系（必须钉死）：

```text
input_ids[j]          本步第 j 个要算的 token id
positions[j]          它在序列里的绝对位置（RoPE）
k[j], v[j]            它在本层的 Key / Value
slot_mapping[j]       它的 KV 应写入的物理 slot

j 三者共用同一下标；N_tok = input_ids.numel() = k.shape[0] = slot_mapping.numel()
```

省例冷启动：

```text
input_ids = [t0..t6]     # schedule 决定算 7 个，prepare 整段塞入
→ 每层 qkv_proj 得到 k[0..6], v[0..6]
→ store 写到 slot 0..6
```

省例 Prefix（前 4 个已 cached）：

```text
input_ids = [t4,t5,t6]   # 只有后缀
→ 只算 k/v 各 3 个 → 只写后缀对应 slot
→ 前缀 K/V 仍留在池里原格，本步不重算
```

##### 3. 算完之后怎么进池？

调用点在 `Attention.forward`（`layers/attention.py`），**先写再算注意力**：

```python
def forward(self, q, k, v):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache
    if k_cache.numel() and v_cache.numel():          # 已挂接大池才写
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
    # ……随后 flash_attn_* ……
```

含义：本层 `qkv_proj` 刚得到的 `k,v`，立刻按 `context.slot_mapping` 写进本层 `self.k_cache` / `self.v_cache`。  
`slot_mapping` 在 `prepare_*` 里已算好；这里**不再算地址**，只按表搬运。

#### K/V 到底怎么写进物理池？（`store_kvcache`）

##### A. 五份数据各自是什么

| 参数 | 形状 / 来源 | 含义 |
|------|-------------|------|
| `key` / `value` | `[N_tok, num_kv_heads, head_dim]`，本层刚算出 | 本步要写入的 K/V；第 `idx` 行 = 本步第 `idx` 个 token |
| `k_cache` / `v_cache` | `[num_blocks, block_size, num_kv_heads, head_dim]`，§1.3 挂接 | 本层物理池；一行向量落在 `(block_id, offset)` |
| `slot_mapping` | 长度 `N_tok` 的 1D int 张量，`prepare_*` 填 | `slot_mapping[idx]` = 第 `idx` 个 token 应写到的线性 slot |

```text
N_tok = 本步新算 token 数 = input_ids.numel() = key.shape[0] = slot_mapping.numel()
slot  = block_id * block_size + offset     （「关键对象一览」§5）
```

`store` **不**读 `block_table`：地址已全部折叠进 `slot_mapping`。

##### B. Python 包装：这些 assert 在检查什么？

先别看断言本身——内核假定了一种**内存排布**，assert 只是在开工前确认「真实排布是否符合假定」。不符合就直接炸，避免静默写错地址。

###### 先搞清：`stride` 不是「这一维有多长」

`shape[i]` = 第 i 维有几个下标。  
`stride[i]` = **下标在第 i 维 +1 时，在一维内存里要跳过多少个元素**。

小例子（连续存放的 `key`，形状 `[2, 2, 4]`，即 2 个 token、2 个 head、每个 head 长 4）：

```text
逻辑下标 key[token, head, dim]：

  token0:  head0: a0 a1 a2 a3 | head1: b0 b1 b2 b3
  token1:  head0: c0 c1 c2 c3 | head1: d0 d1 d2 d3

一维内存里实际顺序（行主序）：
  [a0 a1 a2 a3  b0 b1 b2 b3  c0 c1 c2 c3  d0 d1 d2 d3]
   └─ token0 共 8 个 ─┘  └─ token1 共 8 个 ─┘

此时：
  shape  = (2, 2, 4)
  stride = (8, 4, 1)
           │  │  └─ dim+1  → 跳 1 个元素（相邻标量挨着）
           │  └──── head+1 → 跳 4 个元素（跨过一个 head_dim）
           └─────── token+1 → 跳 8 个元素（跨过 heads*dim = D）
```

内核不会用「多维下标」取数，只用：

```text
读第 idx 个 token 的整段 K：从  key_ptr + idx * key.stride(0)  起，连续读 D 个
写到第 slot 行：            从  k_cache_ptr + slot * D          起，连续写 D 个
```

所以必须保证：每个 token 的 D 个数在内存里是**连续一坨**，且 cache 里每个 slot 也是**连续一坨、且相邻 slot 正好隔 D**。下面四条 assert 就是在验这两件事。

###### 完整包装 + 逐条对照

```python
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 一个 token 的 K（或 V）拉平后有 D 个标量
    # 上例：2*4=8；真实模型常见 num_kv_heads * head_dim

    assert key.stride(-1) == 1 and value.stride(-1) == 1
    # ① 最内维（head_dim）连续：stride(-1)==1
    #    含义：key[..., d] 与 key[..., d+1] 在内存里紧挨着
    #    若不连续（例如被 transpose 弄乱），tl.arange(0,D) 一次 load 会读到错位置

    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # ② head 维步长 = head_dim
    #    含义：head0 的尾巴紧挨 head1 的开头，中间没有空洞
    #    上例 stride(1)=4=head_dim → heads 与 dim 拼成连续的 D
    #    ①+② 合起来：key[idx] 在内存里是连续 D 个数，才能当「一整行向量」搬

    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    # ③ 物理池：沿「格内 offset」走一步，正好跨过一整行向量（D 个元素）
    #    k_cache 形状 [num_blocks, block_size, heads, dim]
    #    正常行主序时：
    #      stride(1) = heads * dim = D
    #      stride(0) = block_size * D
    #    这样 slot = block_id*block_size+offset 时，
    #    第 slot 行起点 = slot * D —— 内核写地址公式才成立
    #    若 stride(1)≠D，说明布局不是「按 token 行排开」，slot*D 会写飞

    assert slot_mapping.numel() == N
    # ④ 本步有 N 个 token 的 K/V，就必须有 N 个写入地址；下标一一对应

    store_kvcache_kernel[(N,)](
        key, key.stride(0),      # 传「token→token」步长，内核用 idx * 它定位
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping,
        D,
    )
    # grid=(N,)：每个 token 一个 program，program_id=idx
```

用一句话收束：

```text
①②：本步 key/value 里，每个 token 的 D 个数是连续的 → 能一次读走
③：  物理池里，每个 slot 的 D 个数也是连续的，且相邻 slot 隔 D → 能用 slot*D 写
④：  表和 token 数量对得上 → 不会漏写/越界下标
```

正常路径（`qkv_proj` 刚 view 出来的 `k`、§1.3 挂接的 `k_cache`）都会满足；assert 防的是以后有人改了布局（transpose、非连续 slice）却仍调用这个 kernel。

`key.stride(0)` 一般等于 `D`（连续时）。内核写成 `idx * key_stride` 而不是写死 `idx * D`，是为了即使外层还有 padding/batch 维导致 token 间距 ≠ D，只要每个 token 内部仍连续，也能读对——但本仓库实际传入的 `key` 通常就是 `[N, heads, dim]` 连续张量，`stride(0)==D`。

##### C. 物理池为何能按 `slot * D` 写？

读这一小节前先 **callback** 三件事（见「关键对象一览」§5 / prepare 填表）：

**1. `slot` 是什么？**

一层 `k_cache` 形状是 `[num_blocks, block_size, heads, dim]`。  
写一个 token 的 K，真正要选的是前两维：住在哪个物理格、格内第几行——即 `(block_id, offset)`。

GPU 内核不想每次都带两个下标，于是把这两维 **拉直成一个整数**：

```text
slot = block_id * block_size + offset
```

这就是 `slot`：在「已经固定本层、且只谈 K（或只谈 V）」的前提下，  
**所有「能放一个 token 的 KV 向量」的格子，按行主序编的全局编号。**

`block_size=4` 时对照：

```text
(block_id, offset)  →  slot
(0,0)(0,1)(0,2)(0,3) →  0  1  2  3
(1,0)(1,1)(1,2)(1,3) →  4  5  6  7
(2,0)...             →  8  9 ...
```

每个 `slot` 对应一整段 `(heads, dim)` 向量，不是一个标量。

**2. 这个编号从哪来的逻辑？**

和二维数组拉直同一套：`a[i][j] → i * 列数 + j`，这里列数 = `block_size`。  
大池在 §1.3 就是按这个顺序连续开的，所以：

```text
逻辑位置 (block_id, offset)
  ↔ 线性编号 slot
  ↔ 内存里「第 slot 行向量」
```

三处共用这一公式，否则对不齐：

| 谁 | 做什么 |
|----|--------|
| `allocate` / `block_table` | 决定某 token 住 `(block_id, offset)` |
| `prepare_*` | 算 `slot = block_id*bs+offset`，写入 `slot_mapping[j]` |
| `store_kvcache` | 只认 `slot`，按 `slot` 写内存 |

**3. 为什么写成 `slot * D`，而不是只写 `slot`？**

`slot` 是「第几行向量」；一行向量有 `D = heads * head_dim` 个标量。  
内存按标量编址，所以第 `slot` 行的起点元素下标是：

```text
起点 = slot * D
     = (block_id * block_size + offset) * D
```

等价说法：把 `k_cache` 看成 `[num_blocks * block_size, D]`，取第 `slot` 行。

```text
slot | (block_id, offset) | 相对 k_cache_ptr 的元素区间
-----|--------------------|---------------------------
  0  | (0, 0)             | [0, D)
  1  | (0, 1)             | [D, 2D)
  2  | (0, 2)             | [2D, 3D)
  3  | (0, 3)             | [3D, 4D)
  4  | (1, 0)             | [4D, 5D)   ← 下一物理格从这开始
  ...
```

省例串一下：`t5`，`block_table=[0,1]`，`block_size=4`

```text
逻辑段 i = 5//4 = 1，offset = 5%4 = 1
block_id = block_table[1] = 1
slot = 1*4 + 1 = 5
写入区间 = [5D, 6D)  → 即 k_cache[1, 1, :, :]
```

`assert k_cache.stride(1) == D` 就是在确认：沿 offset +1，内存正好前进 D——  
上面「相邻 slot 隔 D」成立，`slot * D` 才不会写飞。

##### D. Triton 内核：完整源码 + 逐步

###### 先搞清：`idx` 哪来的？`N` 为什么没进参数列表？

`N` **不是漏传**，它写在**启动网格**里，不写在内核形参里：

```python
# Python 侧（包装函数）
N, num_heads, head_dim = key.shape          # N = 本步 token 数
store_kvcache_kernel[(N,)](                 # ← 这里的 [(N,)] 就是 grid
    key, key.stride(0),
    value, value.stride(0),
    k_cache, v_cache,
    slot_mapping,
    D,
)
```

Triton 约定：

```text
kernel[(grid_x,)](args...)
  → GPU 上并行启动 grid_x 个「program」（可想成 grid_x 个工人）
  → 第 i 个工人里：tl.program_id(0) == i
  → 因此 i ∈ {0, 1, ..., grid_x-1}

本例 grid_x = N
  → 正好 N 个工人，每人负责一个 token
  → idx = tl.program_id(0) 就是「我是第几个 token」
```

所以：

| 东西 | 在哪 | 作用 |
|------|------|------|
| `N` | 启动语法 `[(N,)]` | 决定开几个并行 program |
| `idx = tl.program_id(0)` | 内核内部、运行时注入 | 当前 program 的编号（0..N-1） |
| 形参里的指针 / `D` | 内核参数 | 大家共用的数据与常量 |

内核**不必**再接收 `N`：每个 program 只处理自己的 `idx`，不需要知道一共有多少人（除非你要做边界判断；这里 grid 正好开成 N，人人合法，无需再比 `idx < N`）。

等价心智模型：

```text
# 你以为的串行写法：
for idx in range(N):
    处理 token idx

# Triton 实际：
并行启动 N 份同一段代码，每份被自动塞进不同的 idx = program_id(0)
```

省例 `N=7`：启动 7 个 program，分别带着 `idx=0..6` 跑同一段 `store_kvcache_kernel` 体。

###### 源码全文（`layers/attention.py`）

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,          # 本步 K 张量的数据指针
    key_stride,       # = key.stride(0)：token→token 步长
    value_ptr,
    value_stride,
    k_cache_ptr,      # 本层物理池 K
    v_cache_ptr,
    slot_mapping_ptr, # int 数组，长度 N（N 由启动 grid 决定，此处不再传入）
    D: tl.constexpr,  # heads * head_dim，编译期常量
):
    idx = tl.program_id(0)
    # 不是传进来的参数：由「kernel[(N,)]」启动时，运行时发给本 program 的编号
    # 本 program 负责本步第 idx 个 token；idx ∈ [0, N)

    slot = tl.load(slot_mapping_ptr + idx)
    # 查表：这个 token 写到哪个线性 slot
    if slot == -1:
        return
    # slot==-1：占位/无效，跳过（正常 prefill/decode 路径一般不会走到）

    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 本步 key/value 里第 idx 行、连续 D 个元素的下标
    # 等价于 key[idx].reshape(D) 的元素地址
    # tl.arange(0, D) 见下方小注

    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 读出本步算出的第 idx 个 token 的 K、V 向量（各长度 D）
    # 注意：这里的 load ≠ 从 KV cache 读历史；见下方「load 读的是谁」

    cache_offsets = slot * D + tl.arange(0, D)
    # 物理池里第 slot「行」的连续 D 个元素

    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
    # 写入本层 k_cache / v_cache 的同一 slot
```

###### `tl.arange` 是什么？

和 Python 的 `range` 类似，但产出的是 **一串下标向量**（给 Triton 做向量化读写），不是 Python 迭代器：

```text
tl.arange(0, D)  →  [0, 1, 2, ..., D-1]   （长度 D 的索引）
```

用在偏移里：

```text
key_offsets = idx * key_stride + tl.arange(0, D)
            = [起点, 起点+1, ..., 起点+D-1]

# 起点 = 第 idx 个 token 在 key 张量里的首元素下标
# 于是一次 tl.load 就读走「这一整行」的 D 个标量
```

没有 `arange` 就得手写 D 次标量 load；有了它 =「从起点起连续取 D 个」。  
`cache_offsets = slot * D + tl.arange(0, D)` 同理：从第 `slot` 行起点连续写 D 个。

###### `tl.load` 读的是谁？「写入」代码在哪？

有两处「写」，别混：

**写①：临时 `key`/`value` 张量（load 读的就是它）**  
代码已在前面 **「##### 2. 这些 token 上的 K/V 具体怎么算出来？」** 列出：

```python
# models/qwen3.py → Qwen3Attention.forward
qkv = self.qkv_proj(hidden_states)   # ← 矩阵乘结果写进显存里的 qkv
q, k, v = qkv.split(...)
k = k.view(-1, num_kv_heads, head_dim)
v = v.view(-1, num_kv_heads, head_dim)
q, k = self.rotary_emb(positions, q, k)
o = self.attn(q, k, v)               # ← 把已算好的 k,v 传给 Attention
```

没有单独的 `tl.store` 写临时张量——`qkv_proj` / `view` 本身就是 PyTorch 在 GPU 上分配并填好 `k`/`v`。  
本内核的 `tl.load(key_ptr+...)` 读的就是这份显存。

**写②：长期 KV 池（真正的「进 cache」）**  
就是本内核末尾这三行（上文已列出）：

```python
tl.store(k_cache_ptr + cache_offsets, key)
tl.store(v_cache_ptr + cache_offsets, value)
```

时间线（同一层、同一 step，全在 GPU）：

```text
1) qkv_proj → split/view/RoPE     # 写①：临时 k,v 已在显存
2) Attention.forward → store_kvcache(k, v, k_cache, ...)
3) tl.load(key_ptr+...)           # 从写①读出
4) tl.store(k_cache_ptr+...)      # 写②：拷进长期池
```

| | 写① 临时 k/v | 写② 长期 cache |
|--|--|--|
| 代码位置 | 上文 §「K/V 怎么算」`qkv_proj` | 本内核 `tl.store` |
| 谁读 | 本内核 `tl.load` | 后续 flash_attn / 下轮 decode |
| 设备 | GPU | GPU |

逐步对照（伪代码等价）：

```text
for idx in range(N_tok):                          # 实际是 kernel[(N,)] 并行 N 份
    slot = slot_mapping[idx]
    if slot == -1: continue

    key_vec   = key[idx].reshape(D)               # 本步第 idx 个 token 的 K
    value_vec = value[idx].reshape(D)

    # 把 k_cache 看成 [num_blocks * block_size, D]
    k_cache_flat[slot, :] = key_vec
    v_cache_flat[slot, :] = value_vec
```

注意：

- **只写本步新算的 token**；前缀已在池里的不会出现在 `slot_mapping` 里，也就不会被本内核覆盖（除非地址算错互相踩）。
- **K 和 V 用同一 `slot`**；同一 token 的 K、V 落在两池的相同 `(block_id, offset)`。
- **每一层各调一次** `store_kvcache`；`slot` 号跨层相同，写入的数值是该层自己的投影结果。

##### E. 省例 1：冷启动 — 按 `store_kvcache_kernel` 全文逐步走

设定（一层、本步）：

```text
block_size = 4
block_table = [0, 1]
本步 token：t0..t6（冷启动一次吃完）
N = 7
slot_mapping = [0, 1, 2, 3, 4, 5, 6]
  # prepare：slot = block_table[t//4]*4 + (t%4)

token↔idx↔slot 对照见下方展开后的表。

教学用 D = 8（真实模型更大；公式一样）
key / value 连续存放 → key_stride = value_stride = 8
临时 key 显存按 token 排开：
  key[0] 占 [0..7]，key[1] 占 [8..15]，…，key[6] 占 [48..55]
```

Python 启动：

```python
store_kvcache_kernel[(7,)](
    key_ptr, 8,          # key, key_stride
    value_ptr, 8,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,    # 内容 [0,1,2,3,4,5,6]
    D=8,
)
# → 并行 7 个 program，idx = 0..6
```

先把内核 **每一行** 对 **idx=5**（t5，已跨到物理格1）代入，再给全表。

**program idx=5（负责本步第 5 个 token = t5）**

```python
@triton.jit
def store_kvcache_kernel(...):
    idx = tl.program_id(0)
    # idx = 5

    slot = tl.load(slot_mapping_ptr + idx)
    # 读 slot_mapping[5] → slot = 5
    # callback：slot=5 ⇔ block_id=5//4=1, offset=5%4=1
    #           ⇔ k_cache[1, 1, :, :]

    if slot == -1: return
    # 5 ≠ -1，继续

    key_offsets = idx * key_stride + tl.arange(0, D)
    # = 5 * 8 + [0,1,2,3,4,5,6,7] = [40,41,42,43,44,45,46,47]
    value_offsets = idx * value_stride + tl.arange(0, D)
    # = [40..47]

    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 从临时 key/value 读出 t5 的整段 K、V（各 8 个数）
    # 来自本层 qkv_proj，不是 cache

    cache_offsets = slot * D + tl.arange(0, D)
    # = 5 * 8 + [0..7] = [40..47]
    # 第 slot=5 行在池里的标量区间 [40, 48)

    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
    # k_cache[1, 1] ← key[5]；v_cache[1, 1] ← value[5]
```

对照表（代入前先钉死 t↔idx↔slot）：

```text
token 全局下标 t | 本步 idx | slot | (block_id, offset)
t0               | 0        | 0    | (0,0)
t1               | 1        | 1    | (0,1)
t2               | 2        | 2    | (0,2)
t3               | 3        | 3    | (0,3)
t4               | 4        | 4    | (1,0)
t5               | 5        | 5    | (1,1)
t6               | 6        | 6    | (1,2)
```

**全部 7 个 program 同一套步骤（代入结果，D=8）：**

```text
idx | slot | key_offsets / value_offsets | cache_offsets | 写入
----|------|----------------------------|---------------|------
 0  | 0    | [0..7]                     | [0..7]        | k/v_cache[0,0] ← 本步 key/value[0] (t0)
 1  | 1    | [8..15]                    | [8..15]       | [0,1] ← [1] (t1)
 2  | 2    | [16..23]                   | [16..23]      | [0,2] ← [2] (t2)
 3  | 3    | [24..31]                   | [24..31]      | [0,3] ← [3] (t3)
 4  | 4    | [32..39]                   | [32..39]      | [1,0] ← [4] (t4)
 5  | 5    | [40..47]                   | [40..47]      | [1,1] ← [5] (t5)
 6  | 6    | [48..55]                   | [48..55]      | [1,2] ← [6] (t6)
```

本例里每个 program 的 `key_offsets` 与 `cache_offsets` 碰巧相同（`slot_mapping[idx]==idx`，且冷启动从 t0 起算）。  
**这不是一般规律**——Prefix 里二者会错开（见 F）。

写完后：物理格1 的 offset=3（slot=7）本步未写；注意力只读有效长度。  
每一层各跑一遍（slot 号相同，数值是该层自己的 K/V）。

##### F. 省例 2：Prefix — 同样按内核全文逐步走

设定：前缀 t0..t3 已在池（slot 0..3 已有数）；本步只算后缀。

```text
block_size = 4
block_table = [0, 1]          # 与冷启动相同租约（或 hash 命中复用格0）
num_cached_tokens = 4
num_scheduled_tokens = 3
input_ids = [t4, t5, t6]      # start=4, end=7
N = 3
slot_mapping = [4, 5, 6]      # 只为后缀填地址；没有 0,1,2,3
D = 8，key_stride = value_stride = 8

临时 key/value 只有 3 行（本步只算了 3 个 token）：
  key[0] = t4 的 K，显存 [0..7]
  key[1] = t5 的 K，显存 [8..15]
  key[2] = t6 的 K，显存 [16..23]
```

Python 启动：

```python
store_kvcache_kernel[(3,)](
    key_ptr, 8,
    value_ptr, 8,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,    # 内容 [4, 5, 6]
    D=8,
)
# → 并行 3 个 program，idx = 0..2
# 注意：idx 是「本步第几个」，不是全局 token 下标 t
```

**program idx=0（本步第 0 个 = 全局 t4）全文代入：**

```python
idx = tl.program_id(0)
# idx = 0

slot = tl.load(slot_mapping_ptr + idx)
# slot_mapping[0] = 4
# callback：slot=4 ⇔ (block_id,offset)=(1,0) ⇔ k_cache[1,0]
# 全局 token t4 按分页该住的地方

if slot == -1: return
# 继续

key_offsets = idx * key_stride + tl.arange(0, D)
# = 0*8 + [0..7] = [0..7]
# → 临时 key 的第 0 行（t4），不是全局下标 4

value_offsets = [0..7]

key = tl.load(key_ptr + key_offsets)
value = tl.load(value_ptr + value_offsets)
# 读出本步算的 t4 的 K/V

cache_offsets = slot * D + tl.arange(0, D)
# = 4*8 + [0..7] = [32..39]
# 写到池里第 4 行，不是第 0 行

tl.store(k_cache_ptr + cache_offsets, key)
tl.store(v_cache_ptr + cache_offsets, value)
# k_cache[1,0] ← key[0]（t4）
# 池里 slot 0..3（前缀）本 program 完全不碰
```

**此处与冷启动的关键差别（callback）：**

```text
冷启动 idx=4 时：key_offsets=[32..39]，cache_offsets=[32..39]（碰巧相同）
Prefix  idx=0 时：key_offsets=[0..7]，  cache_offsets=[32..39]（错开！）

原因：
  临时 key 的行号 = 本步 idx（只有后缀 3 行，从 0 编）
  池里的 slot   = 全局分页地址（t4 仍住 slot 4）
  slot_mapping 负责把「本步第 idx 行」接到「该住的 slot」
```

**program idx=1（本步第 1 个 = t5）：**

```text
idx = 1
slot = slot_mapping[1] = 5              # → k_cache[1,1]
key_offsets   = 1*8+[0..7] = [8..15]    # 临时 key 第 1 行
value_offsets = [8..15]
tl.load → t5 的 K/V
cache_offsets = 5*8+[0..7] = [40..47]
tl.store → k_cache[1,1] ← key[1]
```

**program idx=2（本步第 2 个 = t6）：**

```text
idx = 2
slot = slot_mapping[2] = 6              # → k_cache[1,2]
key_offsets   = 2*8+[0..7] = [16..23]
cache_offsets = 6*8+[0..7] = [48..55]
tl.store → k_cache[1,2] ← key[2]
```

**全表：**

```text
idx | 对应 token | slot | key_offsets（临时张量） | cache_offsets（物理池） | 写入
----|------------|------|------------------------|------------------------|------
 0  | t4         | 4    | [0..7]                 | [32..39]               | [1,0] ← key[0]
 1  | t5         | 5    | [8..15]                | [40..47]               | [1,1] ← key[1]
 2  | t6         | 6    | [16..23]               | [48..55]               | [1,2] ← key[2]
```

写完后池状态：

```text
slot 0..3：仍是前缀 t0..t3 的旧 K/V（本步 3 个 program 都没写这些地址）
slot 4..6：刚写入的 t4..t6
slot 7：   仍空（末块未满）
```

##### G. 和前面「约定」的对齐（再钉一次）

```text
prepare：slot_mapping[j] = block_table[t//bs]*bs + (t%bs)   # t = start+j
store  ：把 key[j] / value[j] 写到该 slot

→ 算出来的第 j 个 KV，落在分页规则给 token t 留好的那一行
→ store 只搬运，不算地址；算错地址只可能出在 prepare / block_table
```
#### 主线含义（冷启动，`block_tables is None`）

写完 cache 之后，注意力仍用本步连续的 `k,v`（不必再经 `block_table` 回读）；写入是为后续 decode / 他人 prefix。  
Prefix / chunk 路径：先按上表 store 后缀，再把 `k,v` 换成 cache + `block_table`——见 2.3.1「旁路讲清」。

然后 `Sampler` → `[t7]`，`run` 返回，`reset_context()`。

```text
② 结束后：
  大池：每一层的 block0/1 都已写入 t0..t6 的 K/V（同号，数值不同）
  token_ids = [t7]   # 只是采样结果；t7 的 KV 还没写进池
  Context 已 reset
```

---

## 2.4 `step` ③：`Scheduler.postprocess` — 记账、追加 token、可能还块

**调用位置：** `LLMEngine.step` 第三行，`run` 返回之后。

```python
def postprocess(self, seqs, token_ids, is_prefill):
    for seq, token_id in zip(seqs, token_ids):
        # 在更新 num_cached_tokens 之前：把本轮新「满」的块登记进 hash 表（供 prefix）
        self.block_manager.hash_blocks(seq)
        seq.num_cached_tokens += seq.num_scheduled_tokens  # 本步已算过的 token 计入 cached
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            # chunked prefill：本轮只算了 prompt 的一段，
            # num_cached_tokens 已加上本步，但仍 < 整段 prompt 长度
            # → 采样出的 token 先丢掉不 append，等 prompt 全部算完再进入「生成」
            continue
        seq.append_token(token_id)           # 逻辑序列变长；该 token 的 KV 要等下轮 decode 才写池
        # 遇 eos 或达到 max_tokens → 结束并还块
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)  # ref_count--，为 0 则回 free
            self.running.remove(seq)
```

```text
postprocess 调用链：

LLMEngine.step
  └─ Scheduler.postprocess
       ├─ BlockManager.hash_blocks(seq)      # 给本轮新满块登记指纹
       ├─ num_cached_tokens += scheduled
       ├─ （若该 append）seq.append_token
       └─ （若结束）BlockManager.deallocate   # ref_count--，可能还回 free
```

主线：`num_cached_tokens: 0→7`，然后 `append_token(t7)` → `num_tokens=8`，`last_token=t7`。  
**t7 的 KV 尚未写入大池**（要等下一轮 decode 的 `store_kvcache`）。

`hash_blocks`（满块登记指纹，供以后 `can_allocate` 命中）：

```python
def hash_blocks(self, seq: Sequence):
    # 本轮开始前已满块边界 → 本轮结束后新的满块边界（整除 = 只统计「刚好写满」的块）
    start = seq.num_cached_tokens // self.block_size
    end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
    if start == end: return                  # 本轮没有新满块（例如只写了末块 3 个 token）
    ...  # 对 [start, end) 算链式 hash，写入 hash_to_block_id
```

本轮在 `+=` **之前**调用：`start=0, end=(0+7)//4=1` → 只给满块逻辑段 0（物理 `block_table[0]=0`）登记 hash。末块未满，不登记。

```text
第一次 prefill step 整轮结束：

  token_ids = [t0..t6, t7]     # t7 已 append，但是「逻辑序列」上的新 token
  num_cached_tokens = 7        # 已算过 KV 并写入池的是 t0..t6
  block_table = [0, 1]
  大池：slots 0..6 有值；slot 7 还空着（留给 t7）
  status = RUNNING，仍在 running 队列
```

---

## 2.5 下一轮 `step`：decode 走同一条链的另一分支

再次进入 `generate` 的 while → `step()`。  
此时 `waiting` 空、`running` 有 A → `schedule` 的 prefill while 进不去，落到 **decode 分支**。

### 2.5.1 `schedule` decode 分支 — 逻辑拆开讲

**调用位置：** 仍是 `Scheduler.schedule`；与 prefill 同函数后半段。  
**何时走到这里：** 本轮 `waiting` 里没排上任何 prefill（`scheduled_seqs` 仍空），才开始排 `running` 里的 decode。

先别看循环——先搞清 **decode 什么时候需要多租一块**。

#### 1. 原理：本步要写的是 `last_token`，它住在哪一行？

Callback：decode 的 `prepare_decode` 只喂 `seq.last_token`，并算：

```text
slot = block_table[-1] * block_size + last_block_num_tokens_pos - 1
```

也就是说：**本步 store 的是序列里已经存在的最后一个 token**（上轮 `postprocess` 刚 `append` 的那个）。  
它的全局下标 = `len(seq) - 1`。分页规则：

```text
该住的逻辑段 i = (len-1) // block_size
格内行         = (len-1) %  block_size
```

因此 `block_table` 里至少要有 `i+1` 个物理格，否则 `block_table[-1]` 还不是「该住的那一块」。

省例 `block_size=4`：

```text
len | last_token 下标 | 该住 (逻辑段, 格内行) | 需要几块 | 相对「上一拍」
----|-----------------|------------------------|----------|----------------
 7  | 6               | (1, 2)                 | 2        | prefill 已租 [0,1]
 8  | 7               | (1, 3)                 | 2        | 仍用末块最后一行，不用新块
 9  | 8               | (2, 0)                 | 3        | 跨进新逻辑段 → 要多租 1 块
10  | 9               | (2, 1)                 | 3        | 用新块第 1 行，不用再租
...
```

观察：需要多租一块，当且仅当 last_token 落在某块的 **第 0 行**，即：

```text
(len - 1) % block_size == 0
⇔  len % block_size == 1
```

所以源码里反复出现 `len(seq) % self.block_size == 1`——不是魔法，就是「本步要写的 token 正好是新块第一行」。

时间线和「差 1 个就满」别混：

```text
len=8（末块已满）：本步写 t7（下标7，末块最后一行），还不需要新块
postprocess append t8 → len=9
下一轮 len=9（%4==1）：本步写 t8，才 may_append 租第 3 块
```

#### 2. `can_append`：还不租，只问「若需要新块，free 够不够？」

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

把布尔当成 0/1：

```text
不需要新块（len%bs ≠ 1）：右边 = 0 → free >= 0 恒真 → 直接过
需要新块  （len%bs == 1）：右边 = 1 → 必须 free 里至少还有 1 个空格
```

这里 **不改** `block_table`，只做门禁。

#### 3. `may_append`：门禁过了，真的租一块挂上

```python
def may_append(self, seq: Sequence):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```

只在「本步 last_token 要进新块」时，从 `free` 取出一格 append 到表尾。  
必须在 `prepare_decode` **之前**做完，否则 `block_table[-1]` 还是旧末块，slot 会算错。

#### 4. decode 调度循环：不够就抢别人（或自己）

```python
# schedule 后半：decode
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()          # 从队头取出一个候选
    while not self.block_manager.can_append(seq):
        # 需要新块但 free 空 → 腾块
        if self.running:
            self.preempt(self.running.pop())   # 踢队尾别人：deallocate 整表 → waiting
            # 腾出 free 后，外层 while 再测 can_append(seq)
        else:
            self.preempt(seq)                  # 没别人可踢：连自己也拆掉
            break                              # 带着 break 离开 → 下面 else 不执行
    else:
        # Python while-else：while 正常结束（条件变假）才进这里
        # 即 can_append 已成功；若上面 break 则跳过
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)     # 若需要，现在才真正租块
        scheduled_seqs.append(seq)

assert scheduled_seqs                          # decode 分支要求本轮至少排到一个
self.running.extendleft(reversed(scheduled_seqs))
# callback：上面 popleft 把候选从 running 摘走了；排上的还要塞回去，
# 否则 postprocess 里 running.remove(seq) / 下一轮 decode 都找不到它们。
# extendleft(reversed(...)) = 按原相对顺序放回队头。
return scheduled_seqs, False                   # False = 本轮是 decode
```

流程图：

```text
从 running 队头取出 seq
        │
        ▼
  can_append(seq)? ──是──► may_append（必要时租块）→ 排进本轮 decode
        │否
        ▼
  running 里还有别人？
    │是                    │否
    ▼                      ▼
  preempt(队尾别人)      preempt(自己) + break
  （free 可能+）           （本 seq 本轮不排）
    │
    └──回到 can_append 再问
```

`while-else` callback：Python 里 `while` 的 `else` 在 **没有被 `break` 打断** 时执行。  
所以「租块 + 进 scheduled」只发生在 `can_append` 最终成功；自己被 preempt 时走 `break`，不会误 `may_append`。

#### 5. 两个省例钉死

**省例 A：第一次 decode（prefill 后 len=8）**

```text
len=8，8%4=0 ≠ 1
can_append：不需要新块 → True
may_append：if 不成立，block_table 仍 [0,1]
本步 store t7 → slot 7（格1最后一行）
```

**省例 B：某步 len=9，且 free 空、running 里还有别人**

```text
len=9，9%4=1 → 需要新块
can_append：free=0 → False
→ preempt(队尾) → 别人 deallocate，free 可能变 1
→ can_append 再测 → True
→ may_append：block_table.append(新块) 例如 [0,1,2]
→ 本步 store t8 → slot = 2*4 + 1 - 1 = 8（新块第 0 行）
```

**被 preempt 后 KV 怎样？** ——对这个请求而言，**专属租约没了**；逻辑 token 还在。

```text
preempt(seq) 实际做的事：
  1) block_manager.deallocate(seq)
       · 对 block_table 里每个物理格 ref_count--
       · ref_count→0 的格回到 free（别人可以租走、覆盖写入）
       · seq.num_cached_tokens = 0
       · seq.block_table.clear()
  2) status=WAITING，is_prefill=True，塞回 waiting 队头

保留：token_ids（prompt + 已生成）
丢掉：KV 租约 → 不能接着 decode，要重新 allocate + prefill（可碰运气 prefix）
```

一句话收束：

```text
len%bs==1  ⇒ 本步 last_token 要进新块
can_append ⇒ 先问 free 够不够（不够就 preempt 腾块）
may_append ⇒ 够了再真正租，保证 prepare_decode 时表尾已是新块
```

### 2.5.2 `prepare_decode` + Attention

**调用位置：** `ModelRunner.run` 里 `is_prefill=False` 时。

```python
def prepare_decode(self, seqs: list[Sequence]):
    input_ids, positions, slot_mapping, context_lens = [], [], [], []
    for seq in seqs:
        input_ids.append(seq.last_token)          # 本步只喂上一个采样出的 token（省例 t7）
        positions.append(len(seq) - 1)            # 该 token 的绝对位置（省例 7）
        context_lens.append(len(seq))             # 含 last_token 的总长，供 FA cache_seqlens
        # 写到「当前最后一块」的最后一行：
        # slot = 末块物理号 * block_size + 末块已有 token 数 - 1
        slot_mapping.append(
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        )
    block_tables = self.prepare_block_tables(seqs)   # decode 一定带表（要 gather 历史）
    set_context(False, slot_mapping=slot_mapping, context_lens=context_lens,
                block_tables=block_tables)           # False = is_prefill
    return input_ids, positions
```

省例手算 slot：

```text
block_table = [0, 1]
last_block_num_tokens = 8 - 1*4 = 4
slot = 1 * 4 + 4 - 1 = 7

→ 把 t7 的 K/V 写到 block1 的第 3 行（slot 7）
block_tables = [[0, 1]]   # FA 按表读历史 t0..t6（及写完后的 t7 位置语义由 cache_seqlens 约束）
```

`Attention.forward` 走 `else`（decode）：

1. `store_kvcache` 写入 slot 7（t7 的 KV 此时才进池）；
2. `flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens=[8], block_table=[[0,1]], ...)`  
   —— 用 t7 的 Q attend 池里按表取到的 t0..t7 的 K/V。  

参数含义与和 `flash_attn_varlen_func` 的对比，见上文 **「两个 FlashAttention API」**。

### 2.5.3 decode 的 `postprocess` 与「何时再开新块」

`postprocess` 同样：`hash_blocks` → `num_cached_tokens += 1`（7→8）→ `append_token(t8)` → `len=9`。

下一轮 decode 时 `len=9`，`9 % 4 == 1` → `may_append` 会再 `_allocate_block()`，例如 `block_table=[0,1,2]`，再为 t8 准备新块 slot 8。

```text
decode 链小结：

step (is_prefill=False)
  ① schedule → may_append?（仅当 len%block_size==1）
  ② run → prepare_decode → store 当前 last_token → FA with kvcache
  ③ postprocess → append 新采样 token（其 KV 留给再下一轮 decode store）
```

> 规律：采样出的新 token，**逻辑上**在本轮 `postprocess` 就 `append` 进序列；**物理 KV** 要到**下一轮** decode 的 `store_kvcache` 才写入。Prefill 末尾的 t7 如此，之后每个 decode token 也如此。

---

## 2.6 多请求 / Prefix（仍在同一套 API 上）

### 多请求

```text
同一 ModelRunner.kv_cache + 同一 BlockManager.free_block_ids

请求 A: block_table = [0, 1]
请求 B: block_table = [2, 5, 3]   # 号可以任意交错
schedule 一次可能：
  · 只排一批 prefill，或
  · waiting 空时排一批 decode
prepare_block_tables：把各 seq 的表 pad 成 [batch, max_blocks] 的 GPU tensor
```

谁占哪些 `block_id`，完全由各自 `allocate` / `may_append` 从 free list 弹出决定；**初始化不会按请求切池**。

### Prefix Caching（把 2.3.1「旁路」串成完整故事）

目标：两条请求共享同一段 **system prompt（满块）** 的 KV，不拷贝显存。

**请求 A 先跑完**（冷启动，与主线类似）：prefill 写入物理格，`hash_blocks` 把满块指纹记进 `hash_to_block_id`。

**请求 B 进来**，prompt 前缀与 A 相同：

```text
1) schedule → can_allocate
   从逻辑段 0 起算链式 hash，命中 → 返回 num_cached_blocks = 前缀满块数
   （例如 1；末块不满不参与命中）

2) allocate(seq, num_cached_blocks)
   前段：block_table 挂上共享的物理号，ref_count++
   后段：后缀 _allocate_block 新租
   num_cached_tokens = num_cached_blocks * block_size
   ← 这就是 prepare 里 start>0 的来源

3) prepare_prefill
   start = num_cached_tokens > 0
   seqlen_q = 只算「未命中后缀」
   seqlen_k = start + seqlen_q  > seqlen_q     ← 「K 比 Q 长」的唯一含义
   input_ids 只有后缀 token
   cu_seqlens_k[-1] > cu_seqlens_q[-1] → 带上 block_tables

4) Attention
   store：只写后缀 KV 进新租的格
   block_tables 非空 → 从 k_cache/v_cache + 表读「前缀+后缀」做注意力
   （前缀格与 A 共享，零拷贝）

5) 结束 deallocate
   ref_count--；共享格要等 A、B 都还完才回 free
   （hash 表里指纹可能仍保留，供下一条再命中）
```

和冷启动的对比（同一套 API，只是数字不同）：

| | 冷启动 A | Prefix 命中后的 B |
|--|----------|-------------------|
| `num_cached_tokens` | 0 | >0（满块前缀） |
| 本步 `input_ids` | 整段 prompt | 只有后缀 |
| `seqlen_q` vs `seqlen_k` | 相等 | K 更长 |
| `block_tables` | `None` | 非空 |
| attn 用的 K/V | 本步算出的连续张量 | 池子 + 表 gather |

> 若只看主线冷启动，可跳过本节；一旦在源码里看到 `cu_seqlens_k` / `block_tables is not None`，回到 2.3.1「旁路讲清」对照这张表。

---

## 2.7 对照总表（主线一条请求）

| 步骤 | 真实函数 | 调用自 | 主线结果 |
|------|----------|--------|----------|
| 入队 | `add_request` → `Scheduler.add` | `generate` | waiting=[A]；尚无租约 |
| 问能不能租 | `BlockManager.can_allocate` | `schedule` prefill | 返回 0 |
| 真租 | `BlockManager.allocate` | `schedule` prefill | block_table=[0,1] |
| 算写地址 | `prepare_prefill` | `run` | slot_mapping=[0..6] |
| 写池+attn | `Attention.forward` | `model` 前向 | 池中有 t0..t6 KV |
| 采样 | `Sampler` | `run` | t7 |
| 收尾 | `postprocess` + `hash_blocks` | `step` | append t7；满块 0 登记 hash |
| 下一轮租？ | `may_append` | `schedule` decode | len=8 不新开块 |
| 写 t7 + 读历史 | `prepare_decode` + FA with kvcache | `run` | store slot 7 |
| 结束还块 | `deallocate` | `postprocess`（eos/max_tokens） | ref_count→0 则回 free |

和 §1 的一句话对照：

```text
§1：开货架（kv_cache）+ 建登记本（BlockManager，全空闲）
§2：登记本上借号写进 block_table → prepare_* 把号译成 slot → Attention 按 slot 写、按表读 → 结束还号
```
