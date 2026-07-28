
KV Cache
- 指的是 LLM **推理（decode）** 时，把各层 Attention 里已经算好的 **$K$、$V$** 存进显存
- 下一步生成新 token 时，**直接读取** 历史 token 的 $K/V$，只增量计算新 token 的那一行
- 注意：cache 的是 **$K$ 与 $V$**，不是 $Q$

# KV Cache 的动机是什么

既然叫 cache，理由很简单：**被缓存的东西会被反复使用**。

但 KV Cache 真正省下的，不是「读显存比做一次矩阵乘法更快」——单次 matmul 有时反而比读内存更划算。

真正的问题是：**如果不 cache，每生成 1 个新 token，就要对整段历史 token 重新跑一遍各层前向**，再重算它们全部的 $K/V$。序列越长，浪费越大（近似 $O(T^2)$ 量级）。

因此 cache 要同时满足：

1. 历史上算过的 **$K/V$ 会在下一步被再次用到**
2. **不重算、直接读 cache** 比「对全部旧 token 再跑一遍网络」便宜得多
3. **$Q$ 不满足 1**——旧 token 的 $Q$ 在 decode 阶段没有复用场景

下面验证这三条是否成立。

---

## 为什么 1 / 为什么 2 / 为什么 Q 不 cache？

### 前提——自回归推理

自回归是 KV Cache 存在的前提：每步序列变长，旧 token 再次参与 Attention，才会产生「重复需要的 $K/V$」。

假设 LLM 是黑盒 $f$，推理过程：

```
step1: [token_1]                    →  f(...) = token_2

step2: [token_1, token_2]           →  f(...) = token_3

step3: [token_1, token_2, token_3]  →  ...

step_n: [token_1, ..., token_n]     →  token_{n+1} = eos → 停止
```

自回归即：每吐出一个 token，就拼回 prompt，再喂给模型，直到 EOS 或达到长度上限。

> 工程上还会分 **Prefill**（prompt 一次并行算完、填满 cache）和 **Decode**（之后每次只算 1 个新 token）。下文先按「逐 token 生成」讲 decode 逻辑，最后补 Prefill。

---

### 推理内部：Self-Attention 与「只要最后一行」

符号（与文档 1 一致）：

- $D$：模型主通道维（d_model）
- $d$：Attention 内部维（单头 head dim）
- $V$：词表大小
- $B=1$，当前序列长 $T$

某层 Attention 输入 $X \in \mathbb{R}^{T \times D}$，权重 $W_Q,W_K,W_V \in \mathbb{R}^{D \times d}$：

1. $Q = X W_Q \in \mathbb{R}^{T \times d}$
2. $K = X W_K \in \mathbb{R}^{T \times d}$
3. $V = X W_V \in \mathbb{R}^{T \times d}$
4. $A = \dfrac{QK^\top}{\sqrt{d}} \in \mathbb{R}^{T \times T}$
5. $O = \mathrm{softmax}(A)\,V \in \mathbb{R}^{T \times d}$，再经 $W_O$ 回到 $D$ 维

Attention 矩阵 $A$ 的第 $i$ 行：第 $i$ 个 token 对所有 token 的关注程度。

例：「我 喜欢 吃 苹果」，$T=4$，$A$ 是 $4\times4$。

---

#### 原因分析：推理 ≠ 训练

训练时：每个位置都要预测 next token，在LLM输出的时候，对于输入的T个token，需要完整的 $(T,V)$ logits。

**推理生成时**：只关心 **最后一个 token 的下一个词**。

- LM Head 输出 $(T,V)$，但只用最后一行 $M[-1]$
- Attention 也只需 **最后一行的 query** 去查全部 key
	- 因为我只关心最后一个token如何关注其他token
	- 不关心其他token如何关注最后一个token

因此 decode 可改写为：

1. $Q' = X[T-1:] \, W_Q \in \mathbb{R}^{1 \times d}$（只算最后一个位置）
2. $K = X W_K \in \mathbb{R}^{T \times d}$（需要**全部**位置）
3. $V = X W_V \in \mathbb{R}^{T \times d}$
4. $A' = \dfrac{Q' K^\top}{\sqrt{d}} \in \mathbb{R}^{1 \times T}$

---

#### 矩阵视角：为什么 cache K/V、不 cache Q

第 $T$ 步 decode 时：

| 对象 | 本步需要什么 | 是否可复用 |
|------|-------------|-----------|
| $Q'$ | 当前最后一个位置 | **否**——每步都是新 token 的新 query，因为Q'是根据最新的token算出来的，这个最新的token是第一次进入LLM |
| $K$ | 全部 $T$ 个位置 | **前 $T-1$ 行**在上一步已算过 |
| $V$ | 全部 $T$ 个位置 | **前 $T-1$ 行**在上一步已算过 |

对 $K$ 的计算：

$$
K = X W_K,\quad X \in \mathbb{R}^{T \times D},\; W_K \in \mathbb{R}^{D \times d}
$$

- 第 $T-1$ 步：$X$ 形状 $(T-1, D)$，得到 $K_{1:T-1}$
- 第 $T$ 步：$X$ 形状 $(T, D)$，其中 **只有最后一行 $X[T-1:]$ 是新的**

因此本步只需：

$$
k_T = X[T-1:] \, W_K \in \mathbb{R}^{1 \times d}
$$

再从 cache 读出 $K_{1:T-1}$，拼接：

$$
K = \mathrm{concat}(K_{\mathrm{cached}},\; k_T)
$$

$V$ 完全同理。

**$Q$ 不 cache 的原因**：不是算得贵，而是 **旧位置的 $Q$ 在 decode 里根本用不到**——我们只需要「当前最后一个 token 怎么看别人」，不需要「历史上每个 token 当时怎么看别人」。

---

## 因果性：为什么旧 K/V 真的「算过了就不用重算」？

「前 $T-1$ 行上次算过」还不够——还要说明：**第 $T$ 步重算时，前 $T-1$ 行的结果与上次一致**。  
这正是 **Causal Mask（因果掩码）** 保证的。

### 从设计角度讲

Decoder-only 使用 **因果 Self-Attention**：位置 $t$ 只能 attend 到 $\le t$ 的位置。

对 Attention score 加 mask，未来位置置为 $-\infty$，说白了就是对于这个矩阵，上三角置为负无穷：

$$
A_{t,j} = \begin{cases}
\dfrac{q_t^\top k_j}{\sqrt{d}} & j \le t \\
-\infty & j > t
\end{cases}
$$

推论（对每一层、每个位置 $i$）：

> 位置 $i$ 的 hidden state $h_i$，只依赖输入 token $x_1,\ldots,x_i$，**与 $x_{i+1}, x_{i+2}, \ldots$ 无关**。

形式化：设第 $l$ 层在序列长为 $T$ 时的输出为 $H^{(l)} \in \mathbb{R}^{T \times D}$。  
当序列从长 $T-1$ 扩展到 $T$ 时：

$$
H^{(l)}_{1:T-1} \;\text{（扩展前算出的前 $T-1$ 行）}\;=\; H^{(l)}_{1:T-1} \;\text{（扩展后重算的前 $T-1$ 行）}
$$

前 $T-1$ 行的表示 ** bitwise 不变**。  
在该层做 $K = H W_K,\; V = H W_V$ 时，前 $T-1$ 行的 $K/V$ 自然也 **不变**。

因此 KV Cache 在数学上是正确的，不是近似——前提是：

- 同一模型权重
- 同一数值精度 / 同一算子实现
- 因果 mask 正确施加

> 实际实现里，每层各自 cache 一份 $K_l, V_l$（含 RoPE 施加在 $K$ 上之后的结果），不是只在输入层 cache 一次。

### 从直觉角度讲

把读序列想成 **从左到右、不能偷看未来**：

```
我 → 喜欢 → 吃 → 苹果 → （下一个词？）
```

规则：**读到第 $i$ 个词时，你只能根据前 $i$ 个词形成理解**。

- 当你只读到「我 喜欢 吃」时，「吃」的含义已经确定
- 后面再出现「苹果」，**不会 retroactively 改掉「吃」在第三步时的理解**

Attention + 因果 mask 就是这个规则的矩阵版：

| 直觉 | 对应到模型 |
|------|-----------|
| 不能偷看未来 | 因果 mask |
| 第 $i$ 个 token 的理解只由前 $i$ 个词决定 | $h_i$ 不依赖 $x_{i+1:}$ |
| 第三步写下的「笔记」 | 该层该位置的 $K_i, V_i$ |
| 继续往后读不会改写旧笔记 | 扩展序列时 $K_{1:i}, V_{1:i}$ 不变 |

所以 KV Cache 的直觉是：

> **过去 token 在「当时」对各层 Attention 留下的 K/V 名片，不会因为后面又多了新 token 而作废。**

新 token 到来时，只需：

1. 为新 token 算一行 $Q', K, V$
2. 用新 $Q'$ 去「查阅」cache 里所有旧 $K$，加权聚合旧 $V$
3. 把新 token 的 $K,V$ append 进 cache

---

## Prefill vs Decode（补充）

| 阶段 | 输入 | 做什么 | KV Cache |
|------|------|--------|----------|
| **Prefill** | 整段 prompt（长度 $T_0$） | 并行算所有位置，填 cache | 写入 $K_{1:T_0}, V_{1:T_0}$ |
| **Decode** | 每次 +1 token | 只算新 token 的各层前向 | 读 cache + append 新一行 |

Prefill 不是「一个字一个字喂」；Decode 才是逐步 append。  
但 **因果性在两种阶段都成立**，所以 cache 逻辑一致。

---

# 理论上如何做 KV Cache

假设当前是第 $n$ 步 decode（序列已有 $n$ 个 token），对 **每一层** $l = 1..L$：

**已有 cache：**

$$
K^{(l)}_{\mathrm{cached}} \in \mathbb{R}^{n \times d_k},\quad
V^{(l)}_{\mathrm{cached}} \in \mathbb{R}^{n \times d_k}
$$

（$d_k$ 为 K/V 的头维；GQA/MQA 时 key/value 头数可能少于 query 头数，但「按 token 存一行」的逻辑相同。）

**第 $n+1$ 步：**

1. 新 token 过 Embedding + 各层，**只前向新位置**（依赖该层 cache 做 Attention）
2. 在该层算新一行：
   $$
   k^{(l)}_{n+1} = h^{(l)}_{n+1} W_K,\quad v^{(l)}_{n+1} = h^{(l)}_{n+1} W_V
   $$
3. 用 **当前 token 的 $Q$** 与 **全部 $K$**（cached + 新行）算 attention，聚合 **全部 $V$**
4. Append：
   $$
   K^{(l)} \leftarrow \mathrm{concat}(K^{(l)}_{\mathrm{cached}},\; k^{(l)}_{n+1})
   $$
   $V$ 同理
5. 更新 cache 为新的 $K^{(l)}, V^{(l)}$

伪代码：

```python
# 每层、每步 decode
q_new = x_new @ W_Q          # (1, d)
k_new = x_new @ W_K          # (1, d)
v_new = x_new @ W_V          # (1, d)

K = concat(K_cache, k_new)   # (T, d)
V = concat(V_cache, v_new)

scores = q_new @ K.T / sqrt(d)   # (1, T)
out = softmax(scores) @ V        # (1, d)

K_cache, V_cache = K, V
```

---

# 实际上如何实现 KV Cache（以 nano vllm 为例）




# 一条请求需要多少 KV Cache

粗算：对 **每一层**，要为序列中 **每个已处理 token** 各存一份 K 和一份 V。

设：

- $L$：层数
- $T$：当前序列长度（生成过程中逐渐增长，最大到上下文上限 $N$）
- $d_k$：每个 K/V 头的维度
- $n_{kv}$：K/V 头数（GQA 时 $n_{kv} < n_q$）
- $b$：每个元素字节数（如 FP16 则 $b=2$）

则当前 cache 体积近似：

$$
\mathrm{KV\_size} \approx 2 \times L \times T \times n_{kv} \times d_k \times b
$$

其中因子 $2$ 来自 K 和 V 各一份。

注意：

- $T$ 随生成 **逐步增长**，不是固定 $(N-1)$
- 即使「上下文打满」不再生成，**生成过程中**仍需要为已生成的每个 token 保留 cache，直到该请求结束释放
- 多请求并发时，总显存 = 各请求 cache 之和 + 模型权重

---

## 小结

| 问题 | 答案 |
|------|------|
| Cache 什么？ | 每层、每 token 的 $K, V$ |
| 为什么不 cache $Q$？ | Decode 只要最后一个位置的 $Q$；旧 $Q$ 无复用场景 |
| 为什么旧 $K/V$ 可信？ | 因果 mask → 旧位置表示不随新 token 改变 |
| 省在哪里？ | 避免每步对全历史 token 重跑各层前向 |
| 直觉 | 从左读不能改过去；旧 token 的 K/V「名片」仍然有效 |

一句话：

> KV Cache 利用 **因果 Attention 下历史表示不变** 这一性质，在自回归 decode 时只增量计算新 token 的 $K/V$，并复用 cache 完成「当前 $Q$ 查全部历史 $K/V$」，从而把「每步重算整段历史」变成「每步只算 1 个 token + 读 cache」。
