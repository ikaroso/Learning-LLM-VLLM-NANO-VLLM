
# 0. 基础概念


## 0.1 vocabulary (词表)

LLM的本质和其他DNN一样，依然是输入`张量`，输出`张量`，它无法凭空理解人类语言。
因此，需人工规定LLM能认识的所有`词`，并把它统计到一个表里。

我们假设世界上一共有`V`个`词`
那么`词表`就是一个$1\times V$的矩阵
- 例如，整个`词表`可能是`[hi, byebye, good morning, likes, oh yeah, ..., godess]`

举例，`词表`里的一个`词（token）`可以是`hi`，也可以是`，`，也可以是`hi, boy`，取决于具体方法。
这里不详细展开。

## 0.2 token与tokenize

### token

所谓`token`，就是词表中的一个词语

在代码里，对于这个词，用词表里的`位置id`表示
例如，`hi`用数字`15`表示，因为它是词表里的第`15`个元素

### tokenize

所谓tokenize，就是把一句话分成一系列token。

例如，如果我有一句话`hello, who are you`，也许就被拆分成`[1, 2, 3, 4, 5]`

至于具体怎么做tokennize，即把一句话拆分为一堆`token`
有一些现有方法，不做具体描述。

## 0.3 embedding （词嵌入）

词向量，指的是表示一个`token`的向量。

先捋一捋之前的逻辑链条：`词表里有V个词 -> 给定一句话 -> 代码中每一个词就用token表示（实际用位置id表示) -> ？`。
在下一步`？`中，需要用一种方式来让这个`词`参与模型的计算。

直接把位置id输入进去给模型，显然是不现实的。
目前的方案，是用`一个词向量`来代表一个`token`的内容
那么，也就是用`一个词矩阵`来代表整个`词表`的内容，这个`矩阵`的`size`是$V\times D$
- `V`代表词表中`token`的数量
- `D`代表这个`token`用多大的向量来表示，这个由开发者自己设计，可以是`1024`、`2048`等等
- 比如说，对于`hi`这个`token`，它的词向量可能就是$[0.5, 0.2, ..., 0.9]$

在这种情况下，整个`vocabulary`就形成了一个巨大的`词矩阵`

那么，对于一句话，如果它被分成了`T`个`token`
这段话就被表示为$T\times D$的矩阵

`词向量（矩阵）`的初始值可以是随机值，也可以是一些预训练的值
随着LLM的训练，反向传播时，会更新词向量的具体值
- 对于如何训练词向量，其实原理上很简单，既然它参与了模型的前向传播，那算loss之后反向传播，自然可以更新它的梯度


# 1. 基于decoder堆叠的主流LLM

下面是一个只包含一个 Transformer Block 的 decoder-only LLM 的完整数据流，使用旋转位置编码（RoPE）。
将根据这个图进行逐行，逐个算子的解释

预先知识：
- `B`：batch size
- `T`：序列长度（token 数）
- `D`：**模型主通道维度**（d_model）。Embedding、残差流、Block 输入/输出、FFN 的进出、LM Head 的输入，都是这个维
- `d`：**Attention 内部维度**（单头时即 head dim；多头时常写作 $d_{head}$）。Q/K/V 与 `softmax(S)V` 的结果在这个维上
- `d_{ff}`：FFN 中间升维宽度
- `V`：词表大小（vocabulary size）

> 记法约定：下文一律用大写 $D$ 表示「残差流 / 词向量长度」，小写 $d$ 只表示 Attention 内部维。二者可以相等（简化单头时设 $d=D$），但概念上不要混。

```
输入 token ids (B, T)
        |
        v
+-----------------------------+
| Token Embedding             |
| (B,T) -> (B,T,D)            |
| 无位置嵌入                   |
+-----------------------------+
        |
        v   x (B, T, D)
========= 唯一的 Decoder Block =========
        |
        |---------------------+  (残差分支, 维度保持 D)
        v                     |
   [ RMSNorm ]  (Pre-Norm)    |
        |                     |
        v                     |
   Self-Attention             |
     - Q,K,V: (B,T,D)->(B,T,d)|
     - RoPE 旋转 Q,K (不旋 V)  |
     - scores = QK^T/sqrt(d)  |
     - + Causal Mask (下三角)  |
     - out = softmax(s)·V     |  out: (B,T,d)
     - out -> Wo              |  Wo: d→D，回到主通道
        |                     |
        v                     |
       (+) <----- 残差 --------+
        |
        |   x = x + Attn(Norm(x))   # 两边都是 (B,T,D)
        |
        |---------------------+  (残差分支)
        v                     |
   [ RMSNorm ]  (Pre-Norm)    |
        |                     |
        v                     |
   FeedForward (SwiGLU)       |
     - h = xW1 ⊙ SiLU(xW3)    |
     - y = h·W2               |
     - D -> d_ff -> D         |
        |                     |
        v                     |
       (+) <----- 残差 --------+
        |
        |   x = x + FFN(Norm(x))
        v
====================================
        |
        v
   [ Final RMSNorm ]
        |
        v
+-----------------------------+
| LM Head (Linear)            |
| D -> V (vocab)              |
| (常与 Token Emb 权重共享)    |
+-----------------------------+
        |
        v
   Logits (B, T, V)
        |
        v
   softmax -> 下一个 token 概率
```


1. 整体结构（只有 1 个 Block）
- 真实 LLM 会把中间的 Block 堆叠 N 次（如 32、80 层），这里 N=1。

2. 旋转位置编码 RoPE

- **不使用**传统的可学习/正弦位置嵌入加到 embedding 上。
- RoPE 在**注意力内部**、对 `Q` 和 `K`（不对 `V`）按 token 位置做旋转变换。

3. 因果掩码（Causal Mask）

- decoder-only 的核心：位置 $t$ 只能看到 $\le t$ 的 token，用下三角 mask 把未来位置的 attention score 置为 $-\infty$。

1. 预先了解一些常见的现代化选择（对应上图）

- **Pre-Norm**（Norm 在子层之前），训练更稳定。

- **RMSNorm** 代替 LayerNorm（无均值中心化，更省算力）。

- **SwiGLU** FFN 代替标准 ReLU MLP。

- **权重共享**：LM Head 与 Token Embedding 共享权重（tied embeddings）。

5. 维度速查

其中 `B`=batch，`T`=序列长，`D`=模型主通道，`d`=Attention 内部维。下表按**单头 Self-Attention**理解（多头时 Q/K/V 与 out 变为 `(B, H, T, d_head)`，分数为 `(B, H, T, T)`，最后再拼回 `(B, T, D)`）。

| 阶段 | 形状 | 原理 / 原因 |
|------|------|-------------|
| 输入 ids | `(B, T)` | 分词后的整数 id；只是索引，还不能直接做矩阵运算 |
| Embedding 后 | `(B, T, D)` | 查词表矩阵，id → 向量；进入残差主通道 |
| Q/K/V | `(B, T, d)` | $W_Q/W_K/W_V: D\to d$；在 Attention 内部空间里做匹配与聚合 |
| Attention 分数 | `(B, T, T)` | $QK^\top/\sqrt{d}$；第 $t$ 行 = 位置 $t$ 对全序列的相关性 |
| Attn 加权后 out | `(B, T, d)` | $\mathrm{softmax}(S)V$；仍在 Attention 内部维，尚未回到主通道 |
| Attn 经 $W_O$ 后 | `(B, T, D)` | $W_O: d\to D$；投影回主通道，才能与残差相加 |
| Block 输出 | `(B, T, D)` | Attn + FFN + 残差后；每个 token 已融合上下文并完成变换 |
| Logits | `(B, T, V)` | LM Head：$D\to V$；每位置对「下一个 token」各打一遍分 |

# 1.1 Token Embedding ：输入处理

- 假设输入`B句等长度的prompt`
- `分词`后得到$B\times T$个`token`
- 每个`token`查`词矩阵`，得到一个$B\times T\times D$大小的`张量`，这就是decoder的输入
- 假设`B=1`会更便于理解，因为`B=1`代表只输入一句话作为prompt

---

# 1.2 RMSNorm：均方根归一化

- 输入：`张量(B, T, D)`
- 输出：`张量(B, T, D)`

处理过程：

### 1.2.1RMSNorm 对三维张量 (B, T, D) 的完整解释

#### 1. 输入张量形式

在 Transformer 和大语言模型中，隐藏状态（hidden states）通常表示为三维张量：

$$
X \in \mathbb{R}^{B \times T \times D}
$$

其中：

| 符号 | 含义 |
|---|---|
| B | Batch Size，一次输入的样本数量 |
| T | Sequence Length，序列长度，即 token 数量 |
| D | Hidden Dimension，每个 token 的隐藏向量维度 |

例如：

```text
X.shape = (8, 2048, 4096)
```

表示：

- Batch 中有 8 个样本
- 每个样本有 2048 个 token
- 每个 token 是 4096 维 hidden vector


### 1.2.2. RMSNorm 的核心思想

RMSNorm（Root Mean Square Layer Normalization）是一种归一化方法。

它的目标：

> 保持每个 token 的 hidden vector 的数值尺度稳定。

与 LayerNorm 不同：

- LayerNorm 会减去均值（mean）
- RMSNorm 不计算均值
- RMSNorm 只根据向量长度进行缩放


对于一个 token：

$$
x=[x_1,x_2,...,x_D]
$$

RMSNorm 只关注：

$$
\sqrt{\frac{1}{D}\sum_i x_i^2}
$$

也就是向量的均方根大小。


### 1.2.3. RMSNorm 在 (B,T,D) 上如何计算

输入：

$$
X\in R^{B\times T\times D}
$$


RMSNorm 的归一化维度：

$$\boxed{\text{只沿最后一个维度 }D\text{ 计算}}$$


也就是说：

- Batch 之间互不影响
- Token 之间互不影响
- 每个 token 独立对自己的 D 维 hidden vector 做 RMSNorm


对于：

$$X_{b,t,:}$$


表示：

- 第 b 个 batch
- 第 t 个 token
- 对应的 D 维 hidden vector


例如：

```
X.shape = (2,3,4)
```

表示：

```
2 个样本
3 个 token
每个 token 4 维 hidden
```


其中：

```
X[0,0]
=
[x1,x2,x3,x4]
```

会单独计算 RMS。



### 1.2.4. RMS 计算公式

对于某个 token：

$$
x=[x_1,x_2,...,x_D]
$$


#### Step 1：平方

$$
x_i^2
$$


例如：

$$
[2,4,6,8]
$$

平方：

$$
[4,16,36,64]
$$


#### Step 2：求平均

$$
\frac1D\sum_{i=1}^{D}x_i^2
$$


例如：

$$
\frac{4+16+36+64}{4}=30
$$



#### Step 3：开根号

$$
RMS(x)
=
\sqrt{
\frac1D\sum_{i=1}^{D}x_i^2
}
$$


实际计算加入：

$$
\epsilon
$$


防止除零：

$$
RMS(x)
=
\sqrt{
\frac1D\sum_{i=1}^{D}x_i^2+\epsilon
}
$$



### 1.2.5. RMSNorm 完整公式

对于：

$$
X\in R^{B\times T\times D}
$$


输出：

$$
Y\in R^{B\times T\times D}
$$


每个元素：

$$
Y_{b,t,i}
=
\frac{
X_{b,t,i}
}
{
\sqrt{
\frac1D
\sum_{j=1}^{D}
X_{b,t,j}^{2}
+\epsilon
}
}
\gamma_i
$$


其中：

| 符号 | 含义 |
|---|---|
| X | 输入 hidden state |
| Y | 输出 hidden state |
| γ | 可学习缩放参数（随训练更新） |
| ε | 数值稳定项（超参数） |


矩阵形式：

$$
\boxed{
Y=
\frac{X}
{
\sqrt{
\frac1D\sum X^2+\epsilon
}
}
\odot\gamma
}
$$




### 1.2.6. 张量维度变化过程

假设：

```
X.shape=(B,T,D)

例如：

(8,2048,4096)
```


####  平方

执行：

```python
X ** 2
```


维度：

```
(B,T,D)
```

不改变。



####  沿 D 维求平均

执行：

```python
torch.mean(
    X**2,
    dim=-1
)
```


变化：

```
(B,T,D)

↓

(B,T)
```


因为：

```
D维被压缩
```


得到：

```
每个 token 一个 RMS 值
```



#### 保留维度

实际实现：

```python
torch.mean(
    X**2,
    dim=-1,
    keepdim=True
)
```


结果：

```
(B,T,1)
```


例如：

```
(8,2048,1)
```



####  归一化

执行：

```python
X / sqrt(rms)
```


广播：

```
(B,T,D)
/(B,T,1)

=

(B,T,D)
```


含义：

每个 token 的 D 个 hidden 值：

除以同一个 RMS。



####  乘可学习参数 gamma

参数：

```
gamma.shape=(D)
```


例如：

```
(4096)
```


广播：

```
(B,T,D)
*
(D)

=

(B,T,D)
```


最终：

```
Output.shape=(B,T,D)
```



### 1.2.7. PyTorch 实现


```python
import torch


def rms_norm(x, weight, eps=1e-6):

    # x:
    # shape = (B,T,D)

    # 每个 token 计算 RMS
    rms = torch.mean(
        x ** 2,
        dim=-1,
        keepdim=True
    )

    # RMS归一化
    x_norm = x / torch.sqrt(
        rms + eps
    )

    # gamma缩放
    output = x_norm * weight

    return output
```


使用：

```python
B = 8
T = 2048
D = 4096


x = torch.randn(
    B,
    T,
    D
)


gamma = torch.ones(D)


y = rms_norm(
    x,
    gamma
)
```


维度：

```
输入:

x
(8,2048,4096)


RMS:

(8,2048,1)


gamma:

(4096)


输出:

y
(8,2048,4096)
```


### 1.2.8. RMSNorm 与 LayerNorm 的区别


#### LayerNorm


公式：

$$
y_i=
\frac{x_i-\mu}
{\sqrt{\sigma^2+\epsilon}}
\gamma_i+\beta_i
$$


步骤：

##### 1. 计算均值

$$
\mu=
\frac1D\sum_i x_i
$$


##### 2. 去均值

$$
x_i-\mu
$$


##### 3. 计算方差

$$
\sigma^2
=
\frac1D
\sum_i(x_i-\mu)^2
$$


##### 4. 缩放和平移

$$
\gamma_i+\beta_i
$$




#### RMSNorm


公式：

$$
y_i=
\frac{x_i}
{
\sqrt{
\frac1D\sum_i x_i^2+\epsilon
}
}
\gamma_i
$$


只需要：

1. 平方
2. 求平均
3. 开根号
4. 缩放


没有：

- 均值计算
- 去中心化
- 方差计算
- bias 参数 β



### 1.2.9. 为什么大模型使用 RMSNorm


####  计算更简单

LayerNorm：

需要：

- mean
- variance
- subtraction


RMSNorm：

只需要：

- square
- mean
- sqrt


计算量更低。




####  更适合 GPU 推理


大语言模型：

例如：

LLaMA-70B：

$$
D=8192
$$


每个 token 都需要归一化。


RMSNorm：

减少：

- kernel 数量
- memory access
- synchronization


更容易做 CUDA kernel fusion。



####  效果接近 LayerNorm


现代 LLM：

- LLaMA
- Qwen
- Mistral


大量使用 RMSNorm。


原因：

- 训练稳定
- 推理效率高
- 参数更少

---


# 1.3 Single-Head Self-Attention / Multi-Head Self-Attention

- `输入维度`：$B\times T\times D$（主通道）
- `输出维度`：$B\times T\times D$（必须回到主通道，才能做残差相加）
- 中间（Q/K/V、加权聚合）：$B\times T\times d$（Attention 内部维）

经过`词嵌入`和`归一化`之后，送入`self-attention`做计算

先解释单头注意力机制

然后解释多头注意力机制

实际上用这两种都可以，但现在一般发现multi-head学习能力更好

# 1.4 Single-Head attention

- 输入：`张量(B, T, D)`
- 输出：`张量(B, T, D)`（经 $W_O$ 投影回主通道后）
- 中间 Q/K/V 与加权结果：`张量(B, T, d)`
- Attention 分数：`张量(B, T, T)`

总步骤如下：
Single-Head Self-Attention  
 - Q,K,V = xW_q,xW_k,xW_v     # (B,T,D) → (B,T,d)
 - RoPE 旋转 Q,K (不旋 V)  
 - scores = QK^T/sqrt(d)      # (B,T,T)
 - Causal Mask (下三角)  
 - out = softmax(s)·V         # (B,T,d)
 - out -> Wo                  # (B,T,d) → (B,T,D)


## 1.4.1. 输入定义

假设 Transformer 某一层的输入为：

$$
X\in\mathbb{R}^{B\times T\times D}
$$

其中：

- $B$：Batch Size，表示一次输入多少个样本
- $T$：Sequence Length，表示每个样本包含多少个 token
- $D$：Hidden Dimension，表示每个 token 的特征维度


例如：

$$
B=8,\quad T=2048,\quad D=4096
$$

表示：
```

8个样本  
每个样本2048个token  
每个token是4096维向量

```


## 1.4.2. Self-Attention 的目标

Self-Attention 的核心思想：

> 对每一个 token，根据它和其他 token 的相关程度，动态聚合其他 token 的信息。
> 假设有T个token，那就生成一个$T\times T$的表格，表格中每一个item表示这两个token之间的关系打分

例如：

输入：
```

我 喜欢 吃 苹果

```


对于 token：
```

苹果

```


模型需要学习：
```

苹果  
↓  
关注  
↓  
吃、喜欢

```


因此每个 token 的表示都会融合整个序列的信息。


## 1.4.3. 生成 Q、K、V

Self-Attention 首先通过三个线性层生成：

- Query（查询）
- Key（键）
- Value（值）


公式：

$$
Q=XW_Q
$$


$$
K=XW_K
$$


$$
V=XW_V
$$


其中：

$$
W_Q,W_K,W_V\in\mathbb{R}^{D\times d}
$$


$d$ 是 Attention 内部维度。


例如：

$$
D=4096
$$

$$
d=128
$$


那么：

$$
W_Q\in\mathbb{R}^{4096\times128}
$$


## 1.4.4. Q、K、V 的 Tensor Shape


输入：

$$
X:
(B,T,D)
$$


经过线性变换：


$$
Q=XW_Q
$$


shape：

$$
(B,T,D)
\times
(D,d)
=
(B,T,d)
$$


因此：

$$
Q\in\mathbb{R}^{B\times T\times d}
$$


同理：


$$
K\in\mathbb{R}^{B\times T\times d}
$$


$$
V\in\mathbb{R}^{B\times T\times d}
$$


最终：
```

X:

(B,T,D)

Linear

Q:

(B,T,d)

K:

(B,T,d)

V:

(B,T,d)

```


## 1.4.5. 计算 Attention Score


核心公式：

$$
S=\frac{QK^T}{\sqrt d}
$$


其中：

- Q 表示查询
- K 表示被查询的信息标签
- QK^T 表示 token 之间的相关程度



## 1.4.5 Tensor Shape计算


Q：

$$
Q\in\mathbb{R}^{B\times T\times d}
$$


K：

$$
K\in\mathbb{R}^{B\times T\times d}
$$


转置最后两个维度：

$$
K^T\in\mathbb{R}^{B\times d\times T}
$$


矩阵乘：


$$
(B,T,d)
\times
(B,d,T)
$$


得到：


$$
S\in\mathbb{R}^{B\times T\times T}
$$


## 1.4.6. 为什么 Attention Score 是 T×T？


因为 Attention 计算的是：

> 每一个 token 对所有 token 的关注程度。


例如：

输入：
```

我 喜欢 吃 苹果

```


有：

$$
T=4
$$


Attention矩阵：

$$
\begin{bmatrix}
a_{11}&a_{12}&a_{13}&a_{14}\\
a_{21}&a_{22}&a_{23}&a_{24}\\
a_{31}&a_{32}&a_{33}&a_{34}\\
a_{41}&a_{42}&a_{43}&a_{44}
\end{bmatrix}
$$


其中：

第 i 行：

表示：

> 第 i 个 token 对所有 token 的关注程度。



## 1.4.7. Scaling：除以 sqrt(d)


公式：

$$
S=\frac{QK^T}{\sqrt d}
$$


原因：

Q 和 K 的维度越大：

点积结果越大。


例如：

$$
q\cdot k
$$


如果：

$$
d=4096
$$


点积可能非常大。


进入 Softmax：

$$
softmax(1000,900,800)
$$


会导致：
```

最大值接近1  
其他值接近0

```


梯度变得不稳定。


因此使用：

$$
\sqrt d
$$


进行缩放。


## 1.4.8. Causal Mask（因果掩码）


GPT 类模型使用 causal mask。


目的：

> 当前 token 不能看到未来 token。


例如：

输入：
```

我 喜欢 吃 苹果

```


预测：
```

吃

```


时：

不能知道：
```

苹果

```


所以 Attention Score 加 Mask：


$$
S'=S+M
$$


其中：


未来位置：

$$
M_{ij}=-\infty
$$


例如：

$$
\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}
$$


Softmax 后：

$$
e^{-\infty}=0
$$


未来 token 权重变为：

$$
0
$$



## 1.4.9. Softmax 得到 Attention 权重


公式：

$$
A=softmax(S')
$$


Shape：

$$
A\in\mathbb{R}^{B\times T\times T}
$$


例如：

某个 token：
```

苹果

```


Attention：
```

我 0.1  
喜欢 0.2  
吃 0.6  
苹果 0.1

```


表示：

苹果这个 token：

60% 信息来自 "吃"

20% 信息来自 "喜欢"


## 1.4.10. 加权求 Value


最终输出：


$$
O=AV
$$


Shape：


Attention：

$$
A:
(B,T,T)
$$


Value：

$$
V:
(B,T,d)
$$


矩阵乘：

$$
(B,T,T)
\times
(B,T,d)
$$


得到：


$$
O:
(B,T,d)
$$



## 1.4.11. 直观理解


假设：

$$
A=
\begin{bmatrix}
0.5&0.3&0.2
\end{bmatrix}
$$


Value：

$$
V=
\begin{bmatrix}
V_1\\
V_2\\
V_3
\end{bmatrix}
$$


输出：

$$
O=
0.5V_1+0.3V_2+0.2V_3
$$


也就是：

> 当前 token 的新表示，是所有 token Value 的加权平均。



## 1.4.12. 输出投影 Wo


Attention 加权聚合后的输出：

$$
O=\mathrm{softmax}(S)V
\in\mathbb{R}^{B\times T\times d}
$$


### $(B,T,d)$ 是什么意义？

此时每个位置仍是一个向量，但这个向量活在 **Attention 内部空间**里：

| 符号 | 含义 |
|------|------|
| $B$ | 多少条样本 |
| $T$ | 每个样本多少个 token |
| $d$ | 每个 token 在 Attention 里的表示长度（由 $W_Q/W_K/W_V$ 投出来） |

直观理解：

> $O_{b,t,:}$ =「第 $b$ 条样本、第 $t$ 个位置，按注意力权重对所有 Value 做加权平均」得到的结果。  
> 它已经融合了上下文，但维度还是 $d$，**还不能直接加回残差**——残差分支上的 $x$ 是 $D$ 维的。

### 为什么还要 $W_O$，变成 $(B,T,D)$？

经过：

$$
Y=OW_O
$$

其中：

$$
W_O\in\mathbb{R}^{d\times D}
$$

得到：

$$
Y\in\mathbb{R}^{B\times T\times D}
$$

意义：

1. **维度对齐**：残差是 $x = x + \mathrm{Attn}(\cdot)$，两边最后一维必须都是 $D$，否则无法相加。
2. **写回主通道**：模型真正在层与层之间传递的「主干表示」始终是 $D$ 维；$d$ 只是 Attention 子层内部的工作空间。
3. **可学习混合**：即使单头，$W_O$ 也可再做一次线性重组；多头时它还负责把各 head 拼起来的结果映射回 $D$。

一句话：

> $(B,T,d)$ = Attention **内部**聚合结果；  
> $(B,T,D)$ = 投影回 **残差主通道** 后的更新量，用来和原来的 $x$ 相加。



## 1.4.13. 完整计算流程


$$
X(B,T,D)
$$


↓

生成 Q,K,V


$$
Q,K,V=XW_Q,XW_K,XW_V
$$


↓

计算 Attention Score


$$
S=\frac{QK^T}{\sqrt d}
$$


↓

Mask


$$
S'=S+Mask
$$


↓

Softmax


$$
A=softmax(S')
$$


↓

聚合 Value


$$
O=AV
$$


↓

输出投影


$$
Y=OW_O
$$


最终：

$$
Y\in\mathbb{R}^{B\times T\times D}
$$



## 1.4.14. 总公式


单头 Self-Attention：

$$
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt d}
+Mask
\right)V
$$


其中：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$


输入：

$$
X\in R^{B\times T\times D}
$$


输出：

$$
Y\in R^{B\times T\times D}
$$



## 1.4.15. 与 Multi-Head Attention 的关系


单头：

$$
Q,K,V\in R^{B\times T\times d}
$$


计算一次：

$$
Attention(Q,K,V)
$$


---

多头：

拆成：

$$
h
$$

个 head：

$$
Q_i,K_i,V_i
$$


每个 head：

$$
Attention(Q_i,K_i,V_i)
$$


然后：

$$
Concat(O_1,O_2,...,O_h)
$$


最后：

$$
Y=Concat(O_1,...,O_h)W_O
$$


所以：

> 单头 Attention 是 Multi-Head Attention 中每一个 head 内部的基本计算单元。



# 1.5 Multi-Head attention（todo）


总步骤如下：
Multi-Head Self-Attention  
 - Q,K,V = xWq,xWk,xWv    
 - RoPE 旋转 Q,K (不旋 V)  
 - scores = QK^T/sqrt(d)  
 - Causal Mask (下三角)  
 - out = softmax(s)·V     
 - out -> Wo      



# 1.6 为什么需要旋转位置编码

上述的attention机制解释中，其实省略了旋转位置编码


## 1. 为什么需要位置编码？

Transformer 的 Self-Attention 本身没有顺序概念。

例如：

$$
X=[x_1,x_2,x_3]
$$

和：

$$
X=[x_3,x_2,x_1]
$$


经过 Attention：

$$
QK^T
$$

计算时，只依赖 token 之间的相似度。


因此：

> Attention 不知道 token 的绝对位置和相对距离。


例如：
```

我 喜欢 吃 苹果

```

和：
```

苹果 喜欢 吃 我

```

对于 Attention 来说，如果没有位置编码，两者结构非常接近。


所以需要加入：

> Position Encoding（位置编码）

告诉模型：

- 当前 token 在第几个位置
- token 之间相距多远


## 2. 传统位置编码的问题


Transformer 原论文使用：

Sinusoidal Position Encoding：

$$
PE(pos,2i)=sin(\frac{pos}{10000^{2i/d}})
$$


$$
PE(pos,2i+1)=cos(\frac{pos}{10000^{2i/d}})
$$


然后：

$$
X'=X+PE
$$


也就是：
```

token embedding

position embedding

```


但是这种方式：

- 直接修改输入 embedding
- 位置信息混入 Q,K,V
- 长文本外推能力有限


因此 LLaMA、Qwen 等模型采用：

$$
RoPE
$$


## 3. RoPE 的核心思想


RoPE 不直接添加位置向量。

而是：

> 通过旋转 Query 和 Key，让 Attention 分数天然包含位置信息。


也就是说：

普通 Attention：

$$
Attention(Q,K)=QK^T
$$


RoPE：

$$
Attention(RoPE(Q),RoPE(K))
$$


其中：

$$
Q'=RoPE(Q)
$$


$$
K'=RoPE(K)
$$


然后：

$$
S=\frac{Q'K'^T}{\sqrt d}
$$


注意：

$$
V
$$

不旋转。




## 4. RoPE 的二维旋转思想


RoPE 的基础来自二维向量旋转。


假设一个二维向量：

$$
x=
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
$$


旋转角度：

$$
\theta
$$


旋转矩阵：

$$
R(\theta)
=
\begin{bmatrix}
cos\theta&-sin\theta\\
sin\theta&cos\theta
\end{bmatrix}
$$


旋转后：


$$
x'=R(\theta)x
$$


展开：

$$
\begin{bmatrix}
x'_1\\
x'_2
\end{bmatrix}
=
\begin{bmatrix}
cos\theta&-sin\theta\\
sin\theta&cos\theta
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
$$


得到：

$$
x'_1=x_1cos\theta-x_2sin\theta
$$


$$
x'_2=x_1sin\theta+x_2cos\theta
$$




## 5. RoPE 如何应用到 Transformer？


假设：

$$
Q,K\in R^{B\times T\times d}
$$


其中：

- B：batch
- T：序列长度
- d：head dimension


RoPE 对每个 token：

根据位置：

$$
m
$$


生成旋转角度：

$$
\theta_m
$$


然后：

$$
q_m'=R_mq_m
$$


$$
k_m'=R_mk_m
$$


其中：

$$
m
$$

表示 token 的位置。

## 6. 旋转角度如何计算？


对于第 i 个二维维度：

$$
\theta_i
=
\frac{1}{10000^{2i/d}}
$$


位置 m：

$$
\theta_{m,i}
=
m\theta_i
$$


因此：

不同位置：

旋转角度不同。


例如：


位置：

$$
m=1
$$


旋转：

$$
\theta
$$


位置：

$$
m=10
$$


旋转：

$$
10\theta
$$


## 7. RoPE 的完整计算过程


假设：

$$
Q=XW_Q
$$


$$
K=XW_K
$$


首先得到：

$$
Q,K\in R^{B\times T\times d}
$$


然后：

对每个位置 m：

$$
Q_m'=R_mQ_m
$$


$$
K_m'=R_mK_m
$$


得到：

$$
Q'=RoPE(Q)
$$


$$
K'=RoPE(K)
$$


然后 Attention：


$$
S=
\frac{Q'K'^T}{\sqrt d}
$$


## 单个 Token 的 RoPE（Rotary Position Embedding）数学过程

假设经过线性映射$Q^{T\times d} = X^{T\times D} \times W_{q}^{D\times d}$以后，一个 token 得到了 Query：

$$
q\in \mathbb{R}^{d}
$$
即$Q$中的一行

其中：

- $d$：attention head dimension
- 通常 $d$ 是偶数，例如 128

假设该 token 的位置为：

$$
m
$$
即这个token是输入X中的第m个token

RoPE 的目标：

> 根据 token 所处的位置 $m$，对 Query 和 Key 的向量空间进行旋转。


---

### 1. 将向量拆成二维子空间


假设：

$$
d=8
$$


那么：

$$
q=
[q_0,q_1,q_2,q_3,q_4,q_5,q_6,q_7]
$$


RoPE 会把相邻两个维度组成一个二维向量：


$$
(q_0,q_1)
$$


$$
(q_2,q_3)
$$


$$
(q_4,q_5)
$$


$$
(q_6,q_7)
$$


也就是：

$$
q=
\{(q_0,q_1),
(q_2,q_3),
(q_4,q_5),
(q_6,q_7)\}
$$


每一对维度看成二维平面上的一个点。

---

### 2. 为每个二维空间设置旋转角度


第 $i$ 个二维空间对应角度频率：


$$
\theta_i=
\frac{1}{10000^{\frac{2i}{d}}}
$$


其中：

$$
i=0,1,...,\frac d2-1
$$


对于位置：

$$
m
$$


实际旋转角度：


$$
\phi_{m,i}=m\theta_i
$$


也就是：

位置越靠后：

旋转角度越大。


---

### 3. 二维旋转公式


对于第 $i$ 个二维向量：


$$
x_i=
\begin{bmatrix}
q_{2i}\\
q_{2i+1}
\end{bmatrix}
$$


旋转矩阵：


$$
R(\phi_{m,i})
=
\begin{bmatrix}
cos\phi_{m,i}&-sin\phi_{m,i}\\
sin\phi_{m,i}&cos\phi_{m,i}
\end{bmatrix}
$$


旋转：


$$
\begin{bmatrix}
q'_{2i}\\
q'_{2i+1}
\end{bmatrix}
=
R(\phi_{m,i})
\begin{bmatrix}
q_{2i}\\
q_{2i+1}
\end{bmatrix}
$$


展开：


$$
q'_{2i}
=
q_{2i}cos\phi_{m,i}
-
q_{2i+1}sin\phi_{m,i}
$$


$$
q'_{2i+1}
=
q_{2i}sin\phi_{m,i}
+
q_{2i+1}cos\phi_{m,i}
$$


---

### 4. 一个具体例子


假设：

$$
d=4
$$


一个 token 的 Query：


$$
q=
[q_0,q_1,q_2,q_3]
$$


拆成两个二维向量：


$$
(q_0,q_1)
$$


和：

$$
(q_2,q_3)
$$


---

第 0 个二维空间：


$$
\phi_{m,0}=m\theta_0
$$


旋转：


$$
q'_0
=
q_0cos(m\theta_0)
-
q_1sin(m\theta_0)
$$


$$
q'_1
=
q_0sin(m\theta_0)
+
q_1cos(m\theta_0)
$$


---

第 1 个二维空间：


$$
\phi_{m,1}=m\theta_1
$$


旋转：


$$
q'_2
=
q_2cos(m\theta_1)
-
q_3sin(m\theta_1)
$$


$$
q'_3
=
q_2sin(m\theta_1)
+
q_3cos(m\theta_1)
$$


最终：


$$
q'=
[q'_0,q'_1,q'_2,q'_3]
$$


这就是该 token 的 RoPE 后 Query。


---

### 5. 对 Key 同样处理


原始：

$$
k\in R^d
$$


同样按照位置 $m$：

$$
k'=RoPE(k,m)
$$


即：

$$
k'_{2i}
=
k_{2i}cos\phi_{m,i}
-
k_{2i+1}sin\phi_{m,i}
$$


$$
k'_{2i+1}
=
k_{2i}sin\phi_{m,i}
+
k_{2i+1}cos\phi_{m,i}
$$


---

### 6. RoPE 后进入 Attention


普通 Attention：

$$
S=qk^T
$$


RoPE 后：


$$
S=(q')^T k'
$$


即：


$$
S=
(R_mq)^T(R_mk)
$$


对于不同位置：


Query：

$$
q_m'=R_mq
$$


Key：

$$
k_n'=R_nk
$$


Attention：


$$
(q_m')^Tk_n'
$$


展开：


$$
(R_mq)^T(R_nk)
$$


因为旋转矩阵满足：


$$
R_m^TR_n=R_{n-m}
$$


所以：


$$
(R_mq)^T(R_nk)
=
q^TR_{n-m}k
$$


因此 Attention 分数依赖：

$$
n-m
$$


即：

> token之间的相对位置距离。


---

### 7. 在实际 LLM 中的位置


完整流程：


$$
X
$$


↓

线性映射


$$
Q=XW_Q
$$


$$
K=XW_K
$$


$$
V=XW_V
$$


↓

RoPE


$$
Q'=RoPE(Q)
$$


$$
K'=RoPE(K)
$$


$$
V'=V
$$


↓

Attention


$$
Attention=
softmax
(
\frac{Q'K'^T}{\sqrt d}
+Mask
)
V
$$


---

### 8. PyTorch实现对应关系


假设：

$$
Q.shape=(B,T,d)
$$


构造：

$$
cos(m\theta_i)
$$


和：

$$
sin(m\theta_i)
$$


得到：

$$
cos,sin\in R^{T\times d}
$$


然后：


$$
Q'=
Q\odot cos
+
rotate(Q)\odot sin
$$


其中：


$$
rotate(q)
=
[-q_1,q_0,-q_3,q_2,...]
$$


所以代码中的经典形式：


```python
q_rot = q * cos + rotate_half(q) * sin


## 8. 为什么只旋转 Q 和 K，不旋转 V？


Attention：

$$
O=softmax(QK^T)V
$$


其中：

Q,K：

决定：

> 关注谁


V：

决定：

> 传递什么内容


位置关系只影响：
```

token之间的匹配关系

```


因此：

需要修改：

$$
Q,K
$$


而 Value：

保存内容信息：

$$
V
$$


不需要旋转。


所以：

$$
Q,K \rightarrow RoPE
$$


$$
V \rightarrow 不变
$$

```


## 9. RoPE 为什么能表示相对位置？


这是 RoPE 最核心的性质。


假设：

两个 token：

位置：

$$
m
$$

和：

$$
n
$$


旋转后的 Query 和 Key：

$$
q_m'=R_mq
$$


$$
k_n'=R_nk
$$


Attention 内积：


$$
(q_m')^Tk_n'
$$


展开：


$$
(R_mq)^T(R_nk)
$$


因为旋转矩阵满足：


$$
R_m^TR_n=R_{n-m}
$$


所以：

$$
(q_m')^Tk_n'
=
q^TR_{n-m}k
$$


对attention结果的影响只依赖：

$$
n-m
$$


也就是：

> token之间的相对距离。




## 10. RoPE 加入 Attention 后完整流程


原始：


```
X

|

Linear

|

Q,K,V

|

QK^T

|

Softmax

|

×V

```


加入 RoPE：

```

X

|

Linear

|

Q,K,V

|

Q ---- RoPE ---- Q'

|

K ---- RoPE ---- K'

|

Q'K'^T

|

Softmax

|

×V

|

Output

```


## 11. 在 LLaMA/Qwen 中的实际形式


以 decoder-only LLM 为例：

输入：

$$
X\in R^{B\times T\times D}
$$


生成：

$$
Q=XW_Q
$$


$$
K=XW_K
$$


$$
V=XW_V
$$


然后：

$$
Q'=RoPE(Q)
$$


$$
K'=RoPE(K)
$$


Attention：

$$
Attention=
softmax
(
\frac{Q'K'^T}{\sqrt d}
+Mask
)
V
$$


最后：

$$
Output=Attention W_O
$$




## 12. 为什么 RoPE 有用：核心逻辑与直觉


前面给出了公式；这里把「它到底解决什么、为什么这样设计」收成一条直觉链。

### 问题在哪？

Self-Attention 的打分是：

$$
S_{ij} \propto q_i^\top k_j
$$

这只看内容相似度，**不看**「$i$ 和 $j$ 隔多远」。  
语言里相对位置极重要（「不 … 很」vs「很 … 不」、指代、局部修饰），所以必须把位置信息写进 $q^\top k$。

### 核心逻辑（一句话）

> 不把位置向量加到 $X$ 上，而是按绝对位置旋转 $q$、$k$，使得内积里**只剩下相对位移** $n-m$。

数学上：

$$
(R_m q)^\top (R_n k) = q^\top R_{n-m} k
$$

绝对位置 $m$、$n$ 各自转完后，点积只依赖差值——这正是语言模型更需要的「相对位置」。

### Intuition：把它想成「指针表」

把每个二维子空间想成一块表盘：

| 对象 | 直觉 |
|------|------|
| 内容向量 $q$、$k$ | 表针指向的「语义方向」 |
| 位置 $m$ | 把表针再转过角度 $m\theta$ |
| $q_m^\top k_n$ | 两根转过不同角度的针，夹角是否对齐 |

因此：

- 内容相似 → 原本就容易对齐
- 再叠上相对转角 → 「隔多远时还该不该对齐」也被编码进去
- 不同频率 $\theta_i$：有的维度转得快（敏感近距离），有的转得慢（还能看见远距离）——多尺度相对位置

### 为什么比「$X + PE$」更贴 Attention？

| | 加性 PE | RoPE |
|--|---------|------|
| 作用位置 | 改输入 $X$ | 只改参与打分的 $Q$、$K$ |
| 进入分数的方式 | 间接（经线性层再混进 Q/K） | 直接进入 $Q'K'^\top$ |
| 相对位置 | 要模型自己从绝对 PE 里学出来 | 几何上天然是相对的 |
| Value | 也被 PE 污染 | 不旋转 $V$，内容载体更干净 |
| 额外参数 | 常有可学习 PE | 旋转无额外参数 |

直觉对照：

> 加性 PE：先给每个词贴位置标签，再希望 Attention「顺便」学会用标签。  
> RoPE：打分公式本身就被改成「内容匹配 × 相对距离滤波」。

### 为什么只旋 $Q$、$K$，不旋 $V$？

- $Q$、$K$ 决定**看谁**（路由 / 寻址）→ 需要位置
- $V$ 决定**取什么内容**（载荷）→ 保持语义本身，避免位置旋转扭曲被聚合的信息

### 对长文本 / 外推的直觉

相对位置编码的好处是：模型学的是「相距 $\Delta$ 时怎么互动」，而不是「第 137 号位置长什么样」。  
训练长度之外的位置，只要相对几何仍有意义，就比死记绝对下标更容易外推（实践中还会调 `rope_theta`、NTK/YaRN 等，但根直觉仍是相对旋转）。

### 收束

RoPE 有用，是因为它同时做到了三件事：

1. **把位置写进 Attention 真正用的运算**（点积），而不是旁路贴标签  
2. **自动变成相对位置**，契合语言结构  
3. **几乎零额外成本**（无位置参数、不改 $V$、实现就是按维成对旋转）


## 13. RoPE 总结


### 普通位置编码

$$
X'=X+PE
$$


特点：

- 修改输入
- 添加位置向量




### RoPE

$$
Q'=R_mQ
$$

$$
K'=R_mK
$$


特点：

- 不改变 Value
- 不增加参数
- 通过旋转引入位置信息
- Attention 分数天然包含相对位置



一句话总结：

> RoPE 是一种通过旋转 Query 和 Key 的表示空间，使 Q 与 K 的点积结果包含 token 相对位置信息的位置编码方法，是 LLaMA、Qwen 等现代大语言模型中 Attention 的核心组件。


---
# 1.7 FFN

- FeedForward (使用 SwiGLU 激活)
- $h = (xW_1) \odot SiLU(xW_3)$
- $y = h W_2$
- 维度变化：$D \to d_{ff} \to D$

在 Decoder Block 中，Attention 之后紧接着就是 FFN。
完整子层形式（Pre-Norm）为：

$$
x = x + \mathrm{FFN}(\mathrm{RMSNorm}(x))
$$

---

## 1.7.1 FFN 在 Decoder 里做什么？

一个 Decoder Block 可以粗分为两半：

| 子层 | 作用 | 是否跨 token 交互 |
|------|------|-------------------|
| Attention | 决定「看谁、吸收谁的信息」 | 是（token 之间混合） |
| FFN | 决定「对当前表示做什么变换」 | 否（逐位置独立） |

直觉：

> Attention 负责**通信**（把相关 token 的信息聚过来）；  
> FFN 负责**计算与加工**（对每个 token 自己做非线性变换）。

因此：

- Attention：跨位置信息路由
- FFN：位置内特征精炼 / 知识调用

研究上也常把 FFN 理解为一种 **key-value 记忆**：大量事实性知识主要编码在 FFN 权重里；Attention 更像在「找相关上下文」，FFN 更像在「根据当前表示检索并写出更新」。

---

## 1.7.2 Position-wise：对每个 token 独立计算

设 Attention + 残差之后的隐藏状态为：

$$
X \in \mathbb{R}^{B \times T \times D}
$$

FFN **不对序列维做混合**，而是对每个位置 $t$ 单独施加同一套 MLP：

$$
\mathrm{FFN}(X)_{b,t,:} = f(X_{b,t,:})
$$

也就是：

```
对每个 token 向量 x ∈ R^D：
    用同一组权重做 FFN
    得到新的向量 y ∈ R^D
```

含义：

- token 之间不会在 FFN 里互相看见
- 跨 token 的信息交换只发生在 Attention
- 因此 FFN 在序列维上高度可并行（非常适合 GPU）

---

## 1.7.3 从标准 FFN 到 SwiGLU

### 标准两层 FFN（原始 Transformer）

$$
\mathrm{FFN}(x) = \sigma(xW_1) W_2
$$

常见设定：

- $\sigma$ = ReLU / GELU
- $d_{ff} = 4D$
- 只有两个矩阵：$W_1 \in \mathbb{R}^{D \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times D}$

结构就是：

```
x -- W1 --> 升维到 d_ff -- 激活 --> W2 -- 降回 D --> y
```

### 现代 LLM 常用 SwiGLU

LLaMA / Qwen 等 decoder-only 模型不再用「线性 → 激活 → 线性」，而改用 **门控** 结构 SwiGLU（来自 GLU 变体）：

$$
h = (xW_1) \odot SiLU(xW_3)
$$

$$
y = h W_2
$$

三个矩阵：

| 矩阵 | 角色 | 形状 |
|------|------|------|
| $W_1$ | 内容投影（up / value 支路） | $D \times d_{ff}$ |
| $W_3$ | 门控投影（gate 支路） | $D \times d_{ff}$ |
| $W_2$ | 输出投影（down） | $d_{ff} \times D$ |

> 命名提醒：本文按「$W_1$=内容、$W_3$=门控、$SiLU$ 作用在 $W_3$」书写。  
> LLaMA 官方代码常写成 `silu(w1(x)) * w3(x)`，即把 **gate 叫 W1、up 叫 W3**，符号对调，公式本质相同。

---

## 1.7.4 SwiGLU 结构

公式：

$$
h=(xW_1)\odot SiLU(xW_3)
$$

$$
y=hW_2
$$

其中：

- $W_1$：内容投影
- $W_3$：门控投影
- $SiLU$：激活函数
- $\odot$：逐元素乘法

结构：

```
              W1 (内容)
               |
x ------------>+------------ 
                              \
                               × ---- W2 ---- y
                              /
x ------------> W3 --> SiLU
                 (门控)
```

更完整的数据流：

```
x ∈ R^D
   |
   |----------------------------|
   |                            |
   v                            v
 x W1                        x W3
 (内容支路)                   (门控支路)
   |                            |
   |                         SiLU(·)
   |                            |
   +---------- ⊙ ---------------+
                |
                h ∈ R^{d_ff}
                |
               W2
                |
                y ∈ R^D
```

### 门控直觉

- 内容支路 $xW_1$：提出「候选特征」
- 门控支路 $SiLU(xW_3)$：决定这些特征「开多大」
- 逐元素相乘：按维度选择性放行 / 抑制
- 再经 $W_2$ 压缩回 $D$ 维，作为对本层残差流的更新量

比「整段 hidden 共用一个激活」更灵活，这也是 SwiGLU 被广泛采用的原因之一。

---

## 1.7.5 SiLU 是什么？

$$
SiLU(z) = z \cdot \sigma(z)
$$

其中 $\sigma$ 是 sigmoid：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

也常写作 Swish：

$$
\mathrm{Swish}(z)=z\cdot\sigma(z)
$$

所以：

> **SwiGLU = Swish/SiLU + GLU（门控线性单元）**

性质简记：

- 光滑、处处可导
- 对大正值近似线性放大
- 对负值不完全截断为 0（比 ReLU 更「软」）

---

## 1.7.6 维度变化：$D \to d_{ff} \to D$

对单个 token：

$$
x \in \mathbb{R}^{D}
\quad\to\quad
h \in \mathbb{R}^{d_{ff}}
\quad\to\quad
y \in \mathbb{R}^{D}
$$

对整段序列：

$$
X:\ (B,T,D)
\to
H:\ (B,T,d_{ff})
\to
Y:\ (B,T,D)
$$

例如（示意）：

```
输入:  (B, T, 4096)
升维:  (B, T, 11008)   # d_ff
降维:  (B, T, 4096)
```

「先扩再压」形成信息瓶颈：中间高维空间做非线性变换，再写回模型主通道维度 $D$。

---

## 1.7.7 为什么 SwiGLU 常用 $d_{ff} \approx \frac{8}{3}D$？

标准 FFN（两矩阵，$d_{ff}=4D$）参数量约：

$$
2 \cdot D \cdot d_{ff} = 2 \cdot D \cdot 4D = 8D^2
$$

SwiGLU 有三个矩阵，若仍取 $d_{ff}=4D$，参数会变多。  
为保持参数量大致不变，把中间维缩小到原来的 $\frac{2}{3}$：

$$
3 \cdot D \cdot d_{ff}^{\mathrm{SwiGLU}}
=
2 \cdot D \cdot (4D)
$$

$$
d_{ff}^{\mathrm{SwiGLU}} = \frac{8}{3}D
$$

实践中还会再对齐到 256 的倍数（方便硬件）。例如 LLaMA-7B：

$$
D=4096,\quad
\frac{8}{3}\times 4096 \approx 10922.67
\to
d_{ff}=11008
$$

因此：

> SwiGLU 不是「参数更多」，而是「同样参数预算下，换成三路门控结构」。

---

## 1.7.8 张量计算过程（逐步）

输入（Norm 之后）：

$$
x \in \mathbb{R}^{B \times T \times D}
$$

### 1）内容投影

$$
u = x W_1,\quad W_1\in\mathbb{R}^{D\times d_{ff}}
$$

形状：

```
(B,T,D) @ (D,d_ff) -> (B,T,d_ff)
```

### 2）门控投影 + SiLU

$$
g = SiLU(x W_3),\quad W_3\in\mathbb{R}^{D\times d_{ff}}
$$

形状同样是 `(B,T,d_ff)`。

### 3）逐元素门控

$$
h = u \odot g
$$

形状不变：`(B,T,d_ff)`。

### 4）降维输出

$$
y = h W_2,\quad W_2\in\mathbb{R}^{d_{ff}\times D}
$$

形状：

```
(B,T,d_ff) @ (d_ff,D) -> (B,T,D)
```

### 5）残差加回

$$
X \leftarrow X + y
$$

这就是总图里的：

```
x = x + FFN(Norm(x))
```

---

## 1.7.9 参数量：为什么说 FFN 是「大头」？

粗略看一层 Decoder（忽略 Norm）：

| 组件 | 参数量（量级） | 约占一层 |
|------|----------------|----------|
| Attention（Q/K/V/O） | $4D^2$ | ~1/3 |
| SwiGLU FFN（$W_1,W_2,W_3$） | $3D\cdot d_{ff}\approx 8D^2$ | ~2/3 |

所以：

- 层内大部分参数在 FFN
- Attention 决定信息怎么流动
- FFN 提供主要表达容量与「记忆」空间

这也解释了为什么后来出现 MoE：用多个专家 FFN + 路由，在几乎不增加单 token 算力的前提下，扩大 FFN 容量。

---

## 1.7.10 在 LLaMA / Qwen 中的实际形式

以 decoder-only LLM 为例，输入经 Pre-RMSNorm 后进入 FFN：

$$
x \in \mathbb{R}^{B\times T\times D}
$$

计算：

$$
u = xW_1
$$

$$
g = SiLU(xW_3)
$$

$$
h = u \odot g
$$

$$
y = hW_2
$$

再残差：

$$
X = X + y
$$

伪代码（对应本文命名）：

```python
# x: (B, T, D)
u = x @ W1                 # (B, T, d_ff)
g = silu(x @ W3)           # (B, T, d_ff)
h = u * g                  # 逐元素
y = h @ W2                 # (B, T, D)
x = x + y                  # 残差（外层通常已做 Norm）
```

LLaMA 源码风格（W1/W3 命名对调）等价写法：

```python
y = W2(silu(W1(x)) * W3(x))
```

---

## 1.7.11 FFN 总结

### 和 Attention 的分工

| | Attention | FFN |
|--|-----------|-----|
| 核心问题 | 看哪些 token？ | 对当前表示做什么？ |
| 交互范围 | 跨位置 | 单位置 |
| 典型结构 | Q/K/V + Softmax | 升维 → 非线性/门控 → 降维 |
| 现代选择 | RoPE + Causal Mask | SwiGLU |

### 关键公式

$$
\boxed{
y=
\big((xW_1)\odot SiLU(xW_3)\big)W_2
}
$$

一句话总结：

> Decoder 里的 FFN 是 **position-wise** 的非线性变换层：Attention 把上下文信息聚到每个 token 后，FFN（现代多用 SwiGLU）在 $D\to d_{ff}\to D$ 的高维空间里对每个 token 独立加工，再经残差写回主路径；它约占层内约 2/3 参数，是模型主要的计算与知识存储部件。


# 1.8 LM-Head

经过 $N$ 层 Decoder Block 后，每个 token 仍是一个 $D$ 维向量。
要把「隐藏表示」变成「下一个词是谁」，还差最后一步线性投影：**LM Head**。

总图对应关系：

```
Decoder 堆叠输出  (B, T, D)
        |
        v
  [ Final RMSNorm ]
        |
        v
  LM Head：线性层  D → V
        |
        v
  Logits  (B, T, V)
```

其中 $V$ = vocabulary size（词表大小）。

---

## 1.8.1 Final RMSNorm

进入 LM Head 之前，通常还有一次整网末尾的归一化：

$$
H = \mathrm{RMSNorm}(X_{\mathrm{final}})
$$

形状不变：

```
(B, T, D) -> (B, T, D)
```

作用与块内 Pre-Norm 类似：稳定最后一层表示的尺度，再投影到词表空间。

---

## 1.8.2 LM Head 在做什么？

本质是一个 **无 bias 的线性分类头**：

$$
\mathrm{Logits} = H W_{\mathrm{LM}}
$$

其中：

$$
H \in \mathbb{R}^{B \times T \times D}
$$

$$
W_{\mathrm{LM}} \in \mathbb{R}^{D \times V}
$$

$$
\mathrm{Logits} \in \mathbb{R}^{B \times T \times V}
$$

对位置 $t$ 的某个 token 向量 $h_t \in \mathbb{R}^{D}$：

$$
z_t = h_t W_{\mathrm{LM}} \in \mathbb{R}^{V}
$$

$z_t$ 的第 $i$ 个分量，就是「下一个 token 是词表第 $i$ 个词」的**未归一化分数（logit）**。

直觉：

> Embedding：词 id → 向量（$V \to D$）  
> LM Head：向量 → 词分数（$D \to V$）

二者正好是「进出词表空间」的两端。

---

## 1.8.3 维度变化

```
Final Norm 后:  (B, T, D)
      ×
LM Head 权重:   (D, V)
      =
Logits:         (B, T, V)
```

例子：

```
d = 4096
V = 32000
B = 1, T = 128

H:      (1, 128, 4096)
Logits: (1, 128, 32000)
```

含义：

- 每个位置各有一套「对全词表的打分」
- 训练时：位置 $t$ 的 logits 用来预测位置 $t+1$ 的真实 token（next-token prediction）
- 推理生成时：通常**只取最后一个位置**的 logits，用来决定「下一个新 token」

---

## 1.8.4 权重共享（Tied Embeddings）

很多模型让：

$$
W_{\mathrm{LM}} = E^{\top}
$$

其中 $E \in \mathbb{R}^{V \times D}$ 是 Token Embedding 矩阵。

也就是：

```
输入查表用 E
输出投影用 E^T
两套权重绑在一起（tied）
```

好处：

- 少存一份 $V \times D$ 大矩阵，省参数 / 显存
- 输入、输出共享同一套词向量空间，有一定正则效果

注意：

- GPT-2、部分早期/中型模型常使用 tied embeddings
- LLaMA 等一些现代开源模型**默认不绑定**，LM Head 是独立矩阵
- 本文总图写「常与 Token Emb 权重共享」，指的是常见设计选择，不是所有 LLM 都必须如此

伪代码对比：

```python
# tied
logits = hidden @ embedding.weight.T   # E: (V,D) -> 用 E^T

# untied
logits = lm_head(hidden)               # 独立 Linear(D, V)
```

---

## 1.8.5 Logits 是什么？

Logits **还不是概率**：

- 可正可负，范围不限
- 各个词上的分数加起来也不等于 1
- 只表示相对强弱：分数越高，越「像」被选为下一个 token

要变成概率，需要 Softmax（见下一节后处理）。

对位置 $t$：

$$
z_t = [z_{t,1}, z_{t,2}, \ldots, z_{t,V}]
$$

例如（示意）：

```
token:   "的"   "是"   "猫"   "狗"  ...
logit:   2.1   5.8   0.3  -1.2  ...
```

这里 `"是"` 的 logit 最高，但最终是否选它，取决于采样策略。

---

## 1.8.6 训练时怎么用 LM Head？

因果语言模型的目标：给定前面的 token，预测下一个 token。

对长度 $T$ 的序列：

- 输入：token $1..T$
- 监督：位置 $t$ 的输出去对齐 token $t+1$（最后一位通常对齐 EOS 或忽略）

损失常用交叉熵：

$$
\mathcal{L}
=
-\sum_t
\log p(y_{t+1}\mid y_{\le t})
$$

其中

$$
p(\cdot)=\mathrm{softmax}(z_t)
$$

因此 LM Head 本质上把每个位置的 hidden state 变成一个 **V 类分类问题**。

---

## 1.8.7 LM Head 总结

$$
\boxed{
\mathrm{Logits}= \mathrm{RMSNorm}(X_{\mathrm{final}})\, W_{\mathrm{LM}}
,\quad
W_{\mathrm{LM}}\in\mathbb{R}^{D\times V}
}
$$

一句话总结：

> LM Head 是 Decoder 输出端的线性投影层，把每个 token 的 $D$ 维表示映到词表大小 $V$ 的 logits；它常与 Embedding 权重共享，是「向量 → 下一个词分数」的最后一跳。


---

# 1.9 后处理，得到唯一 token

LM Head 给出的是整表分数。  
生成时还要把 logits **变成一个具体的 token id**，再拼回序列，循环下去——这就是后处理 / decoding。

完整链路：

```
Logits (对词表打分)
   |
   |  (可选) temperature / mask / top-k / top-p
   v
Softmax -> 概率分布 p ∈ R^V
   |
   |  greedy 或 sampling
   v
下一个 token id  (标量)
   |
   |  append 到序列
   v
再送回模型，继续生成
```

---

## 1.9.1 推理时通常只用「最后一个位置」

设当前已有 $T$ 个 token，LM Head 输出：

$$
\mathrm{Logits}\in\mathbb{R}^{B\times T\times V}
$$

预测「第 $T+1$ 个 token」时，取：

$$
z = \mathrm{Logits}_{:,\, T,\, :} \in \mathbb{R}^{B\times V}
$$

即 **最后一列 / 最后一个时间步** 的 logits。

原因：因果注意力下，位置 $T$ 已经看过 $1..T$，其表示正是用来预测下一个词的。

（Prefill 阶段虽一次算出全部 $T$ 个位置的 logits，生成新词时仍主要用最后一位。）

---

## 1.9.2 Softmax：分数 → 概率

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

性质：

- $p_i \in (0,1)$
- $\sum_i p_i = 1$
- logit 越大，概率越大

得到的是「下一个 token 落在词表各位置上的概率分布」。

---

## 1.9.3 Temperature：调节「自信 / 随机」

修改softmax的过程：
常在 Softmax 前对 logits 除以温度 $\tau$：

$$
p_i = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}
$$

| $\tau$ | 效果 |
|--------|------|
| $\tau \to 0$ | 分布变尖，接近只留最大项（更确定） |
| $\tau = 1$ | 原始分布 |
| $\tau > 1$ | 分布变平，更随机、更多样 |

直觉：

> 温度低 → 模型更「固执」选高分词  
> 温度高 → 更愿意尝试次优但仍合理的词

---

## 1.9.4 从概率到「唯一 token」的常见策略

### 1）Greedy（贪心）

$$
\hat{y} = \arg\max_i z_i
\quad(\text{或 }\arg\max_i p_i)
$$

- 永远选概率最大的那个
- 确定性：同输入同输出
- 简单，但容易重复、缺乏多样性

### 2）多项式采样（Sampling）

按 $p$ 做一次随机抽样：

```python
next_id = sample(probs)   #  multinomial
```

- 高概率词更容易被抽到，但不是必然
- 同 prompt 可得到不同续写

### 3）Top-k

只保留 logit / 概率最高的 $k$ 个词，其余置为不可能（logit $= -\infty$），再 Softmax + 采样。

```
保留前 k 名候选
其余概率清零并重新归一化
再采样
```

作用：去掉长尾噪声词，避免采到极不合理 token。

### 4）Top-p（Nucleus / 核采样）

按概率从高到低累加，取最小集合 $S$，使得：

$$
\sum_{i\in S} p_i \ge p
$$

只在集合 $S$ 内采样。

- $p=0.9$ 表示：覆盖约 90% 概率质量的最小头部队
- 候选数量随分布尖锐/平坦自适应变化（比固定 top-k 更灵活）

实践中常组合：

```
logits
 -> / temperature
 -> top-k 截断（可选）
 -> top-p 截断（可选）
 -> softmax
 -> sample
```

---

## 1.9.5 得到唯一 token 后做什么？

设抽到的 id 为 $y_{T+1}$：

1. **Append**：序列变为 $[y_1,\ldots,y_T,y_{T+1}]$
2. **再前向**：把新 token（或整段，配合 KV Cache 只算新位置）送进模型
3. **再取最后位置 logits → 再采样**
4. 直到：
   - 生成到 EOS
   - 或达到 `max_new_tokens`
   - 或触发停止词 / 停止规则

这就是自回归生成（autoregressive decoding）：

```
prompt tokens
   -> 模型 -> 采样出 token_1
   -> 拼回去
   -> 模型 -> 采样出 token_2
   -> ...
   -> EOS
```

---

## 1.9.6 最小可运行伪代码

```python
# hidden: 最后一层 + Final RMSNorm 后，取最后位置
# hidden: (B, d)

logits = hidden @ W_lm          # (B, V)

# 可选：温度
logits = logits / temperature

# 可选：top-k / top-p 掩码（把不要的位置设为 -inf）
logits = apply_top_k_top_p(logits, k, p)

probs = softmax(logits, dim=-1) # (B, V)

if greedy:
    next_id = argmax(probs, dim=-1)
else:
    next_id = multinomial(probs, num_samples=1)

# next_id: (B,)  —— 这就是「唯一 token」
tokens = concat(tokens, next_id)
```

---

## 1.9.7 和总图的对应关系

总图末尾：

```
Logits (B, T, vocab)
        |
        v
softmax -> 下一个 token 概率
```

补全后的精确理解：

1. Softmax 得到的是 **分布**，还不是最终文本
2. 还要用 greedy / sample（可加 temperature、top-k、top-p）选出 **一个** token id
3. 该 id 经 detokenize 才变成可读字符串
4. 多步重复后，才得到完整回答

---

## 1.9.8 后处理总结

| 步骤 | 输入 | 输出 |
|------|------|------|
| 取末位置 | `(B,T,V)` logits | `(B,V)` |
| Temperature / 截断 | logits | 调整后的 logits |
| Softmax | logits | 概率 $p$ |
| Decode 策略 | $p$ | 唯一 token id |
| Append + 循环 | token id | 更长序列 / 最终文本 |

一句话总结：

> 后处理把 LM Head 的 logits 经 Softmax（及 temperature / top-k / top-p）变成词表上的概率，再用贪心或采样选出**唯一下一个 token**，拼回序列后自回归循环，直到结束。
