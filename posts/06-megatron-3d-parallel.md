# Megatron-LM 3D 并行：数据并行、张量并行、流水线并行的组合艺术

当模型参数达到数十亿甚至数千亿级别时，单张 GPU 的显存已经远远装不下整个模型。以 GPT-3 175B 为例，光是 FP16 权重就需要 350GB 显存，而一张 A100 只有 80GB。要训练这样的大模型，就需要多种并行策略协同工作。

Megatron-LM 提出的 **3D 并行（3D Parallelism）** 将数据并行、张量并行、流水线并行组合在一起，可以在数千张 GPU 上高效训练万亿参数模型。

## 一、为什么需要并行训练

### 1.1 显存墙

训练一个 LLM，显存中需要存储：

| 存储项 | 大小（FP16） | 备注 |
|--------|-------------|------|
| 模型参数 | 2 × N bytes | N = 参数数量 |
| 梯度 | 2 × N bytes | 和参数同规模 |
| 优化器状态（Adam） | 12 × N bytes | m + v（FP32）+ 主参数（FP32） |
| 激活值（Activations） | 高度依赖 batch / seq_len | 可以用 activation checkpointing 换时间 |

以 7B 模型为例：
- 参数：14 GB
- 梯度：14 GB
- Adam 优化器状态：84 GB
- **合计（不算激活）：约 112 GB**

一张 A100 80GB 根本装不下，更不用说更大的模型了。

### 1.2 计算墙

即使显存够，计算量也是巨大的。训练 GPT-3 175B 大约需要 **3.14e23 次浮点运算**，用一张 A100 来算需要数千年。必须用成千上万张 GPU 并行计算。

## 二、数据并行（Data Parallelism, DP）

### 2.1 基本原理

数据并行是最简单、最常用的并行策略：

```
            ┌─────────────┐
            │   输入数据   │
            └──────┬──────┘
                   │
     ┌─────────────┼─────────────┐
     │             │             │
     ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│ GPU 0   │  │ GPU 1   │  │ GPU 2   │
│ 模型副本 │  │ 模型副本 │  │ 模型副本 │
│ batch_0 │  │ batch_1 │  │ batch_2 │
│ 算梯度  │  │ 算梯度  │  │ 算梯度  │
└────┬────┘  └────┬────┘  └────┬────┘
     │             │             │
     └─────────────┼─────────────┘
                   │
            梯度 AllReduce
                   │
                   ▼
           所有 GPU 同步更新
```

**核心思想**：每张 GPU 上都有完整的模型副本，把数据切成多份分给不同 GPU 计算梯度，最后同步梯度更新模型。

### 2.2 通信模式

- **AllReduce**：每一步训练结束后，所有 GPU 同步梯度
- 通信量 = 模型参数大小 × 2（reduce + broadcast）
- 通信量与 DP 度数无关（AllReduce 的特性）

### 2.3 优缺点

| 优点 | 缺点 |
|------|------|
| 实现简单 | 每张 GPU 必须存完整模型，模型太大时用不了 |
| 计算效率高（几乎完美加速比） | 通信开销随 GPU 数增长 |
| 兼容各种模型结构 | 当模型 > 单卡显存时无法使用 |

## 三、张量并行（Tensor Parallelism, TP）

### 3.1 基本原理

张量并行把**单个层的计算**拆分到多张 GPU 上。以 Transformer 的 MLP 和 Attention 为例：

**MLP 层的拆分（列并行 + 行并行）：**

```
输入 X (batch × hidden)
     │
     ├───────────┬───────────┐
     ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ GPU 0   │ │ GPU 1   │ │ GPU 2   │
│ A_1     │ │ A_2     │ │ A_3     │ ← 按列切分权重矩阵 A
│ X @ A_1 │ │ X @ A_2 │ │ X @ A_3 │
└────┬────┘ └────┬────┘ └────┬────┘
     │           │           │
     └───────────┼───────────┘
                 │ AllGather 后 GeLU
                 ▼
┌──────────────────────────────────┐
│         GeLU(Y = XA)            │
└────┬───────────┬───────────┬────┘
     │           │           │
     ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ GPU 0   │ │ GPU 1   │ │ GPU 2   │
│ B_1     │ │ B_2     │ │ B_3     │ ← 按行切分权重矩阵 B
│ Y_1 @ B_1││ Y_2 @ B_2││ Y_3 @ B_3│
└────┬────┘ └────┬────┘ └────┬────┘
     │           │           │
     └───────────┼───────────┘
                 │ ReduceSum (AllReduce)
                 ▼
             输出结果
```

### 3.2 关键通信算子

| 算子 | 用途 | 通信量 |
|------|------|--------|
| AllGather | 列并行的输入分发 | 输入激活大小 |
| ReduceScatter / AllReduce | 行并行的结果聚合 | 输出激活大小 |
| AllReduce | 自注意力的 head 结果聚合 | 输出激活大小 |

### 3.3 优缺点

| 优点 | 缺点 |
|------|------|
| 每层的显存和计算都被拆分 | 通信频率高（每层都要通信） |
| 可以支持非常大的层 | 通信量大（激活值比参数多） |
| 适合层内计算密集的场景 | TP 度数通常 ≤ 8（受限于节点内 NVLink） |

> **经验法则**：张量并行通常只在节点内做（单机 8 卡），因为跨节点的 PCIe / 网络带宽远低于 NVLink。跨节点的并行更多依赖流水线并行和数据并行。

## 四、流水线并行（Pipeline Parallelism, PP）

### 4.1 基本原理

流水线并行把**不同的层**放到不同的 GPU 上，数据像流水线一样逐层流过：

```
GPU 0: Layer 0 → Layer 1 → Layer 2 → Layer 3 → 传给 GPU 1
GPU 1: Layer 4 → Layer 5 → Layer 6 → Layer 7 → 传给 GPU 2
GPU 2: Layer 8 → Layer 9 → Layer 10 → Layer 11 → 传给 GPU 3
GPU 3: Layer 12 → Layer 13 → Layer 14 → Layer 15 → 输出
```

### 4.2 气泡问题（Bubble）

朴素的流水线并行有严重的"气泡"问题：GPU 之间需要等待，很多时间处于空闲状态：

```
朴素流水线（微批量数 = 1）：

GPU 0:  ████░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (前向完成后闲置)
GPU 1:  ░████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
GPU 2:  ░░████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
GPU 3:  ░░░████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
         ↑
         气泡时间
         
█ = 前向计算
░ = 气泡（等待）
▒ = 反向传播

效率 ≈ 1 / PP_degree  →  4 级流水线只有 25% 效率！
```

### 4.3 微批量（Micro-batching）解决方案

通过把一个 batch 切成多个 micro-batch，让流水线"填满"，大幅减少气泡：

```
GPipe 风格（微批量数 = 8，PP = 4）：

GPU 0: ████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
GPU 1: ░███████░█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
GPU 2: ░░██████░░██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
GPU 3: ░░░█████░░░███▒▒▒▒▒▒▒▒▒▒▒▒▒
         ↑↑↑
       气泡只有 3 个 micro-batch 的时间！

气泡比例 = (PP - 1) / num_microbatches
         = (4 - 1) / 8 = 37.5%
```

当 micro-batch 数量足够多时，气泡占比可以很低。Megatron-LM 采用的是 **1F1B（One Forward One Backward）** 调度，进一步优化了气泡和显存。

### 4.4 优缺点

| 优点 | 缺点 |
|------|------|
| 显存按层拆分，适合超深网络 | 存在气泡开销 |
| 通信量小（只传相邻层的激活） | 需要仔细划分层，确保负载均衡 |
| 可以跨节点扩展 | 实现复杂，需要处理流水线调度 |

## 五、3D 并行：三者的组合

### 5.1 整体架构

Megatron-LM 的核心贡献是把 DP、TP、PP 三者无缝结合起来：

```
3D 并行 GPU 拓扑示意图（DP=2, TP=2, PP=2，共 8 张 GPU）：

                    ┌─────────────────────── PP 维度 ───────────────────────┐
                    │  Stage 0 (Layer 0-15)   │  Stage 1 (Layer 16-31)    │
                    ├─────────────────────────┼───────────────────────────┤
  TP 维度 ──┐      │  GPU 0  │  GPU 1        │  GPU 2  │  GPU 3           │
            │      │  TP-0   │  TP-1         │  TP-0   │  TP-1            │
            │      ├─────────┼───────────────┼─────────┼───────────────────┤
            │      │  GPU 4  │  GPU 5        │  GPU 6  │  GPU 7           │
            │      │  TP-0   │  TP-1         │  TP-0   │  TP-1            │
            ▼      │  (DP-1) │  (DP-1)       │  (DP-1) │  (DP-1)          │
                   └──────────────────────────┴────────────────────────────┘
                    ↑ 这一组是 DP-0 的 TP组  ↑ 这一组是 DP-1 的 TP组

说明：
- PP 维度：不同 stage 放不同的层（按深度切分）
- TP 维度：同一 stage 内的 GPU 做张量并行（层内切分）
- DP 维度：相同 stage + 相同 TP 位置的 GPU 做数据并行
```

总 GPU 数量 = DP_size × TP_size × PP_size

### 5.2 通信模式汇总

| 并行维度 | 通信算子 | 通信频率 | 通信内容 | 通信对象 |
|---------|---------|---------|---------|---------|
| TP | AllReduce / AllGather | 每层多次 | 激活值 | 同一 PP stage + 同一 DP rank 的 TP 组 |
| PP | Send/Recv (P2P) | 每 micro-batch 两次 | 层间激活值 | 相邻 PP stage |
| DP | AllReduce | 每 batch 一次 | 梯度/参数 | 同一 TP rank + 同一 PP stage 的 DP 组 |

### 5.3 伪代码：3D 并行的训练步骤

```python
def train_step_3d(
    micro_batches,
    dp_rank, tp_rank, pp_rank,
    dp_size, tp_size, pp_size
):
    """
    3D 并行的一次训练 step（简化版伪代码）
    """
    # ============ 前向传播 ============
    all_activations = []
    
    for i, micro_batch in enumerate(micro_batches):
        # 第一个 stage 接收输入数据
        if pp_rank == 0:
            # 数据并行：每个 DP rank 拿到不同的数据切片
            x = micro_batch[dp_rank]
        else:
            # 从上一个 stage 接收激活值（P2P 通信）
            x = recv_from(pp_rank - 1)
        
        # 当前 stage 的所有层前向计算
        # 每层内部做张量并行
        for layer in current_stage_layers:
            # 层内：TP 组协同计算
            x = tensor_parallel_forward(x, layer, tp_rank, tp_size)
        
        # 最后一个 stage 计算 loss
        if pp_rank == pp_size - 1:
            loss = compute_loss(x, labels)
            all_activations.append(loss)
        else:
            # 传给下一个 stage
            send_to(x, pp_rank + 1)
            all_activations.append(x)  # 保存用于反向
    
    # ============ 反向传播 ============
    for i in reversed(range(len(micro_batches))):
        # 最后一个 stage 从 loss 开始反向
        if pp_rank == pp_size - 1:
            grad = backward_loss(all_activations[i])
        else:
            # 从下一个 stage 接收梯度
            grad = recv_grad_from(pp_rank + 1)
        
        # 当前 stage 的所有层反向计算
        for layer in reversed(current_stage_layers):
            grad = tensor_parallel_backward(grad, layer, tp_rank, tp_size)
        
        if pp_rank == 0:
            # 第一个 stage，反向结束
            pass
        else:
            # 传给上一个 stage
            send_grad_to(grad, pp_rank - 1)
    
    # ============ 参数更新 ============
    # 数据并行：DP 组内做梯度 AllReduce
    allreduce_gradients(dp_group)
    
    # 更新参数
    optimizer.step()
    optimizer.zero_grad()
```

## 六、不同模型规模的配置建议

### 6.1 配置参考表

| 模型规模 | GPU 型号 | 推荐 DP | 推荐 TP | 推荐 PP | 总 GPU 数 | 说明 |
|---------|---------|--------|--------|--------|----------|------|
| 7B | A100 80G | 2-4 | 1 | 1 | 2-4 | 单卡就能装下，DP 即可 |
| 13B | A100 80G | 2-4 | 2 | 1 | 4-8 | TP=2 降低单卡显存压力 |
| 34B | A100 80G | 2 | 4 | 2 | 16 | 需要 TP + PP 组合 |
| 70B | A100 80G | 2 | 8 | 4 | 64 | 单机 TP=8，跨机 PP+DP |
| 175B | A100 80G | 4 | 8 | 8 | 256 | 完整 3D 并行 |
| 500B+ | A100 80G | 4+ | 8 | 12+ | 384+ | 需要更多流水线级 |

### 6.2 配置原则

1. **TP 不超过单机 GPU 数**：TP 通信最频繁，应在 NVLink 域内
2. **PP 层数要均衡**：每个 stage 的计算量尽量一致，避免木桶效应
3. **DP 尽量大**：DP 效率最高，在显存允许的前提下尽量增大 DP
4. **Micro-batch 数量 ≥ 2×PP**：减少气泡开销，一般 4-8 倍 PP 比较合适

## 七、通信开销对比

| 并行策略 | 通信量（每个 step） | 通信次数 | 延迟敏感性 | 带宽敏感性 |
|---------|-------------------|---------|-----------|-----------|
| DP (AllReduce) | 2 × 模型参数大小 | 1 次/step | 低 | 高 |
| TP (AllGather + Reduce) | 每层 × 激活值大小 | 多层/step | 中 | 很高 |
| PP (P2P Send/Recv) | PP × 激活值大小 | 2 × num_mb × PP / step | 高 | 中 |

> **关键洞察**：
> - TP 对带宽要求最高，必须用 NVLink，所以通常只在节点内做
> - PP 对延迟敏感，适合用 RDMA 网络跨节点做
> - DP 通信量固定，可扩展性最好

## 八、总结

3D 并行是大规模 LLM 训练的基石，三种并行策略各有分工：

- **数据并行**：横向扩展数据，效率最高，但受限于单卡显存
- **张量并行**：层内拆分计算，通信密集，适合节点内
- **流水线并行**：层间拆分深度，有气泡开销，适合跨节点

Megatron-LM 的贡献不仅是把这三者组合起来，更在于工程上的极致优化：
- 精心设计的 1F1B 调度最小化气泡
- 高效的通信算子融合
- 灵活的配置接口，适配各种规模

理解了 3D 并行，再去看 DeepSpeed 的 ZeRO、FSDP 等技术，就会发现它们都是在"如何更好地切分模型和数据"这个核心问题上做不同的权衡。

---

**参考资料**：
- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
- Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (SC 2021)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
