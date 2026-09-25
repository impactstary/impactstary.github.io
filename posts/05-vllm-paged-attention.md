# vLLM 核心：PagedAttention 是如何让推理吞吐翻倍的

在 LLM 推理服务中，KV Cache 是提升解码速度的核心技术，但它也是显存的"吞金兽"。传统的 KV Cache 管理方式存在严重的内存碎片问题，导致显存利用率往往不足 50%。vLLM 提出的 **PagedAttention** 借鉴了操作系统虚拟内存的分页思想，将 KV Cache 分割成固定大小的"页"，通过页表进行索引，几乎消除了内存碎片，让推理吞吐提升了 2-4 倍。

## 一、问题背景：KV Cache 的内存浪费

### 1.1 什么是 KV Cache

在自回归解码过程中，每生成一个新 token，都需要用当前 token 的 Q 去和**所有历史 token** 的 K、V 做 attention 计算。如果每次都重新计算所有历史 token 的 K 和 V，那计算量会随着序列长度线性增长。

KV Cache 的做法很简单：**把每一层、每一个位置计算出来的 K 和 V 都缓存起来**，下一次解码时直接复用。

```
第 1 步生成 token_1: 计算 K_1, V_1 → 缓存下来
第 2 步生成 token_2: 只算 K_2, V_2，和缓存的 [K_1] 一起做 attention
第 3 步生成 token_3: 只算 K_3, V_3，和缓存的 [K_1, K_2] 一起做 attention
...
第 t 步生成 token_t: 只算 K_t, V_t，和缓存的 [K_1..K_{t-1}] 一起做 attention
```

### 1.2 传统 KV Cache 的问题

传统做法中，每个请求的 KV Cache 都是**一段连续的内存空间**：

```
请求 A (长度 64):  [K_cache_A | V_cache_A]  ← 连续内存块
请求 B (长度 128): [K_cache_B | V_cache_B]  ← 连续内存块
请求 C (长度 32):  [K_cache_C | V_cache_C]  ← 连续内存块
```

这带来了两个严重问题：

1. **内存碎片（Internal & External Fragmentation）**
   - 内部碎片：为了最坏情况预分配最大长度，实际用不完
   - 外部碎片：请求结束释放后，留下大小不一的空洞
   
2. **显存利用率低**
   - 实际生产环境中，显存利用率通常只有 30%-50%
   - 大量显存被浪费在预留空间和碎片中

下面用 ASCII 图展示传统方式的内存碎片问题：

```
传统连续分配方式：

初始状态（空闲显存）:
[ 空闲空闲空闲空闲空闲空闲空闲空闲空闲空闲 ]

分配请求 A (64 token), B (128 token), C (32 token):
[  A(64)  |    B(128)    | C(32) | 空闲空闲  ]

B 结束释放:
[  A(64)  | (空128碎片)  | C(32) | 空闲空闲  ]

新请求 D 需要 160 token:
[  A(64)  | (空128碎片)  | C(32) |    D(160)    | 空 ]
            ↑ 128 不够用，浪费了！
```

## 二、PagedAttention 的核心洞察

PagedAttention 的灵感来源于操作系统中的**虚拟内存机制**：

> 既然操作系统可以用分页（Paging）来解决物理内存的碎片问题，那 KV Cache 为什么不行？

核心思想：

1. 将每个请求的 KV Cache 逻辑上视为连续的
2. 物理上分割成固定大小的"页"（Block / Page）
3. 通过一个 **Block Table（页表）** 来维护逻辑位置到物理页的映射
4. 页不需要物理连续，可以分散在显存的任意位置

## 三、关键数据结构：Block Table

### 3.1 基本概念

- **Block（页/块）**：KV Cache 的最小分配单元，通常包含 16 或 32 个 token 的 KV
- **Block Table（块表）**：每个请求维护一个数组，索引是逻辑块号，值是物理块号
- **Block Manager（块管理器）**：全局维护空闲块列表，负责分配和回收

### 3.2 ASCII 结构图

```
PagedAttention 内存布局示意:

物理显存中的 KV 块（离散分布）:
+--------+  +--------+  +--------+  +--------+  +--------+
| Block0 |  | Block1 |  | Block2 |  | Block3 |  | Block4 |
| 16 tok |  | 16 tok |  | 16 tok |  | 16 tok |  | 16 tok |
+--------+  +--------+  +--------+  +--------+  +--------+

请求 A 的 Block Table (长度 = 3 个块 = 48 token):
逻辑块号:  [  0   |  1   |  2  ]
物理块号:  [  2   |  0   |  4  ]
            ↓      ↓      ↓
          物理块2  物理块0  物理块4

请求 B 的 Block Table (长度 = 2 个块 = 32 token):
逻辑块号:  [  0   |  1  ]
物理块号:  [  1   |  3  ]
            ↓      ↓
          物理块1  物理块3

内存利用率: 5/5 = 100% （无碎片！）
```

### 3.3 伪代码：Block Table 结构

```python
# KV 块大小：每个块存储多少个 token 的 KV
BLOCK_SIZE = 16

class BlockTable:
    """单个请求的块表，维护逻辑块 -> 物理块的映射"""
    
    def __init__(self, num_layers: int, num_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        # block_table[logical_block_idx] = physical_block_idx
        self.block_table: list[int] = []
        # 当前已用 token 数
        self.current_len = 0
    
    @property
    def num_blocks(self) -> int:
        return len(self.block_table)
    
    def get_physical_block(self, logical_block: int) -> int:
        """根据逻辑块号获取物理块号"""
        return self.block_table[logical_block]


class BlockManager:
    """全局块管理器，负责空闲块的分配与回收"""
    
    def __init__(self, total_blocks: int, block_size: int = BLOCK_SIZE):
        self.block_size = block_size
        self.total_blocks = total_blocks
        # 空闲物理块栈
        self.free_blocks: list[int] = list(range(total_blocks))
    
    def allocate(self, num_blocks: int) -> list[int]:
        """分配 num_blocks 个物理块，返回物理块号列表"""
        if len(self.free_blocks) < num_blocks:
            raise OutOfMemoryError("Not enough KV blocks")
        allocated = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        return allocated
    
    def free(self, physical_blocks: list[int]):
        """回收物理块"""
        self.free_blocks.extend(physical_blocks)
    
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)
```

## 四、PagedAttention 在内核中的实现

### 4.1 注意力计算的挑战

在标准 Attention 中，KV 是连续存储的，直接按偏移访问即可。但在 PagedAttention 中，KV 分散在不同的物理块里，attention kernel 需要能够**根据块表间接寻址**。

关键在于：在计算 attention 时，对于每个 query token，需要遍历所有 key token 位置。由于 key 可能分布在不同的块中，kernel 需要：

1. 根据 token 位置计算出逻辑块号和块内偏移
2. 通过 Block Table 查到物理块号
3. 计算出实际的显存地址
4. 加载 K/V 进行计算

### 4.2 PagedAttention Kernel 的伪代码

```python
def paged_attention_kernel(
    query,           # [num_queries, num_heads, head_dim]
    key_cache,       # [num_blocks, num_layers, num_heads, block_size, head_dim]
    value_cache,     # [num_blocks, num_layers, num_heads, block_size, head_dim]
    block_table,     # [max_num_blocks_per_seq] — 该请求的块表
    seq_len,         # 当前序列长度
    scale,
    output
):
    """
    PagedAttention 的核心逻辑（简化版伪代码）
    实际实现是 CUDA kernel，这里用 Python 描述思路
    """
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    block_size = key_cache.shape[3]
    
    for head_idx in range(num_heads):
        # 当前 head 的 Q
        q = query[0, head_idx, :]  # [head_dim]
        
        # 计算 QK^T，需要遍历所有 key token
        # key token 分布在不同的物理块中
        scores = torch.zeros(seq_len)
        
        for logical_block_idx in range((seq_len + block_size - 1) // block_size):
            # 1. 通过块表找到物理块号
            physical_block_idx = block_table[logical_block_idx]
            
            # 2. 计算这个块内实际有多少个 token
            tokens_in_block = min(
                block_size, 
                seq_len - logical_block_idx * block_size
            )
            
            # 3. 从物理块中加载 K
            k_block = key_cache[physical_block_idx, layer_idx, head_idx, :tokens_in_block, :]
            # k_block: [tokens_in_block, head_dim]
            
            # 4. 计算该块的 attention score
            block_start = logical_block_idx * block_size
            scores[block_start:block_start + tokens_in_block] = (
                q @ k_block.T * scale
            )
        
        # 5. Softmax
        weights = softmax(scores)
        
        # 6. 加权求和 V（同样需要按块遍历）
        output_vec = torch.zeros(head_dim)
        for logical_block_idx in range((seq_len + block_size - 1) // block_size):
            physical_block_idx = block_table[logical_block_idx]
            tokens_in_block = min(
                block_size,
                seq_len - logical_block_idx * block_size
            )
            v_block = value_cache[physical_block_idx, layer_idx, head_idx, :tokens_in_block, :]
            
            block_start = logical_block_idx * block_size
            w_block = weights[block_start:block_start + tokens_in_block]
            output_vec += w_block @ v_block
        
        output[0, head_idx, :] = output_vec
```

> **注意**：实际的 PagedAttention kernel 远不止这么简单。为了性能，vLLM 会做大量优化：
> - 按块的 Tiling，利用 Shared Memory
> - 多个请求的 Batch 处理（batching across sequences）
> - FlashAttention 风格的在线 Softmax
> - 预取块表到 Shared Memory / 寄存器

### 4.3 内存索引的关键公式

理解 PagedAttention 最关键的是理解地址计算：

```
给定: token_pos (第几个 token), layer_idx, head_idx, block_size, head_dim

1. 计算逻辑块号:
   logical_block_idx = token_pos // block_size

2. 计算块内偏移:
   token_offset_in_block = token_pos % block_size

3. 通过块表查物理块号:
   physical_block_idx = block_table[logical_block_idx]

4. 计算 K 的实际地址:
   k_addr = key_cache 
            + physical_block_idx * (num_layers * num_heads * block_size * head_dim)
            + layer_idx * (num_heads * block_size * head_dim)
            + head_idx * (block_size * head_dim)
            + token_offset_in_block * head_dim
```

## 五、连续批处理（Continuous Batching）

PagedAttention 的另一个巨大优势是支持 **Continuous Batching（也叫 Dynamic Batching / Iterative Batching）**。

传统的 Static Batching 需要等一批中所有请求都生成完才能开始下一批，而：

- PagedAttention 让每个请求的 KV Cache 独立管理
- 新请求随时可以加入 batch
- 已完成的请求随时可以从 batch 中移除
- GPU 利用率显著提升

```
Static Batching:
Batch 1: [Req-A(长), Req-B(短), Req-C(中)]
         等待最慢的 Req-A 完成...  → 浪费 GPU 时间
Batch 2: [Req-D, Req-E, Req-F]
         等待最慢的完成...

Continuous Batching:
Time 1: [Req-A, Req-B, Req-C]  ← 同时开始
Time 2: [Req-A, Req-C, Req-D]  ← Req-B 完成，Req-D 加入
Time 3: [Req-A, Req-D, Req-E]  ← Req-C 完成，Req-E 加入
Time 4: [Req-D, Req-E, Req-F]  ← Req-A 完成，Req-F 加入
         ↑ GPU 几乎没有空闲！
```

## 六、性能对比

### 6.1 显存利用率对比

| 指标 | 传统 KV Cache | PagedAttention |
|------|-------------|----------------|
| 内存分配方式 | 连续预分配 | 分页按需分配 |
| 内部碎片 | 严重（预留最大长度） | 极小（最多一个块） |
| 外部碎片 | 严重（释放后空洞） | 几乎为零 |
| 典型显存利用率 | 30% - 50% | 90% - 95% |
| 支持动态批处理 | 困难 | 天然支持 |

### 6.2 吞吐量对比（vLLM 论文数据）

| 模型 | 框架 | 吞吐量 (req/s) | 相对提升 |
|------|------|--------------|---------|
| LLaMA-7B | HuggingFace Transformers | ~15 | 1x |
| LLaMA-7B | Text Generation Inference | ~30 | 2x |
| LLaMA-7B | vLLM (PagedAttention) | ~70 | **4.7x** |
| LLaMA-13B | HuggingFace Transformers | ~8 | 1x |
| LLaMA-13B | vLLM (PagedAttention) | ~35 | **4.4x** |

> 数据来源于 vLLM 论文（OSDI 2023），具体数值取决于硬件、序列长度、batch size 等因素。

## 七、总结

PagedAttention 是 LLM 推理领域的一个里程碑式的优化，它的精妙之处在于：

1. **跨界借鉴**：从操作系统虚拟内存中寻找灵感，用分页解决 KV Cache 碎片问题
2. **结构简单**：Block Table + 分块 KV Cache，实现起来不复杂
3. **效果显著**：显存利用率从 30-50% 提升到 90%+，吞吐翻数倍
4. **生态友好**：天然支持 Continuous Batching，成为后续推理框架的标配

理解 PagedAttention，是理解现代 LLM 推理系统的第一步。后续的 SGLang、TensorRT-LLM 等框架也都采用了类似的分页 KV Cache 设计，并在此基础上做了更多创新（比如 RadixAttention 前缀缓存）。

---

**参考资料**：
- vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention (OSDI 2023)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
