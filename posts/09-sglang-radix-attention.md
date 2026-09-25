# SGLang 杀手锏：RadixAttention 前缀缓存的原理与实现

在 LLM 推理服务中，有一个常被忽视但价值巨大的优化机会：**大量请求共享相同的前缀**。

- 系统提示词（System Prompt）：所有请求都带同样的"你是一个有用的AI助手..."
- Few-shot 示例：同一类任务共享相同的示例
- 并行采样（Parallel Sampling）：同一个输入生成多个输出

vLLM 的 PagedAttention 解决了 KV Cache 的内存碎片问题，但每个请求的 KV Cache 是独立的，共享前缀的 KV 仍然被重复计算和存储。SGLang 提出的 **RadixAttention** 用一棵基数树（Radix Tree）来管理 KV Cache，让不同请求共享相同前缀的 KV，带来了额外的数倍性能提升。

## 一、机会：共享前缀无处不在

### 1.1 典型场景

**场景 1：系统提示词**

```
请求 1: [系统提示: "你是一个专业的翻译官..."] + "翻译这句话: Hello"
请求 2: [系统提示: "你是一个专业的翻译官..."] + "翻译这句话: World"
请求 3: [系统提示: "你是一个专业的翻译官..."] + "翻译这句话: Goodbye"
         ↑ 这部分完全相同，KV Cache 可以共享！
```

**场景 2：Few-shot 学习**

```
请求 1: [示例1] + [示例2] + [示例3] + "现在翻译: Apple"
请求 2: [示例1] + [示例2] + [示例3] + "现在翻译: Banana"
         ↑ 示例1-3 完全相同
```

**场景 3：并行采样（Beam Search / Top-k）**

```
输入: "今天天气真不错，"
输出候选 1: "今天天气真不错，适合出去玩..."
输出候选 2: "今天天气真不错，我们去公园吧..."
输出候选 3: "今天天气真不错，不过有点晒..."
            ↑ 前半句共享前缀
```

### 1.2 浪费有多大？

如果系统提示词有 512 token，同时服务 100 个请求：
- PagedAttention：存储 100 × 512 = 51200 个 token 的 KV
- RadixAttention：只存储 1 × 512 = 512 个 token 的 KV
- **节省 99% 的前缀 KV 显存！**

这还不包括计算的节省：前缀的 KV 只需要计算一次，而不是 100 次。

## 二、RadixAttention 核心思想

### 2.1 什么是 Radix Tree（基数树 / 前缀树）

Radix Tree（也叫 Trie、前缀树）是一种树形数据结构，用于高效存储字符串。共享前缀的字符串会共享树中的路径。

```
Radix Tree 结构示意（存储 "cat", "car", "dog", "dot"）：

                [root]
               /      \
             ca        do
            /  \      /  \
           t    r    g    t
           ↑    ↑    ↑    ↑
         cat  car  dog  dot
         
每个节点存储一段公共前缀，叶子节点是完整的字符串。
```

### 2.2 把 KV Cache 组织成 Radix Tree

RadixAttention 的核心想法很直接：

> 把每个请求的 prompt 看作一个字符串（token 序列），
> 把对应的 KV Cache 存储在 Radix Tree 的节点中，
> 共享前缀的请求共享树中的路径，也就共享了 KV Cache。

```
RadixAttention 中的 KV 前缀树示意：

                          [root]
                            |
                   [系统提示词: 512 token]  ← 所有请求共享
                   /          |          \
          [翻译任务:]   [代码助手:]   [写作助手:]  ← 不同任务的前缀
            /    \        /    \        /    \
       [用户A] [用户B] [用户C] [用户D] [用户E] [用户F]
          ↓      ↓      ↓      ↓      ↓      ↓
         不同的请求内容，各自的 KV Cache
```

每个树节点包含：
- 该节点对应的 token 片段（文本 token）
- 对应的 KV Cache 数据（物理上存在 KV 块中）
- 子节点列表（按 token 索引）
- 引用计数（用于缓存淘汰）

## 三、RadixAttention 的工作流程

### 3.1 请求到达：前缀匹配

当一个新请求到达时：

1. 从根节点出发，沿着 token 序列逐 token 匹配
2. 找到最长的匹配前缀（Cache Hit 的部分）
3. 未匹配的部分（Cache Miss）需要新计算 KV 并插入树中

```
新请求: "你是翻译官。翻译: Hello World"
                       ↑ 这里开始不匹配

Radix Tree 中已有路径:
  root → "你是翻译官。翻译: " → "Apple"
                            ↘ "Banana"

匹配过程:
  root ✓ → "你是翻译官。翻译: " ✓ → "Hello World" ✗ (新分支)
  ↑ 已缓存，直接复用 KV ↑       ↑ 这部分需要新计算 ↑
```

### 3.2 伪代码：前缀匹配与插入

```python
class TreeNode:
    """Radix Tree 节点，对应一段 token 的 KV Cache"""
    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids    # 该节点的 token 片段
        self.kv_blocks = []           # 对应的 KV 物理块号
        self.children: dict[int, TreeNode] = {}  # 子节点，按第一个 token 索引
        self.ref_count = 0            # 引用计数
        self.parent = None


class RadixCache:
    """基于 Radix Tree 的 KV 缓存管理器"""
    
    def __init__(self, block_manager):
        self.root = TreeNode([])
        self.block_manager = block_manager
    
    def match_prefix(self, token_ids: list[int]) -> tuple[TreeNode, int]:
        """
        在树中匹配最长前缀
        返回 (匹配到的最深节点, 匹配的总 token 数)
        """
        node = self.root
        matched_len = 0
        remaining = token_ids
        
        while remaining:
            first_token = remaining[0]
            if first_token not in node.children:
                break
            
            child = node.children[first_token]
            # 比较 child 的 token 和请求的 token 有多少重叠
            common_len = 0
            for i in range(min(len(child.token_ids), len(remaining))):
                if child.token_ids[i] == remaining[i]:
                    common_len += 1
                else:
                    break
            
            if common_len == len(child.token_ids):
                # 完全匹配 child 的全部 token，向下走
                node = child
                matched_len += common_len
                remaining = remaining[common_len:]
            elif common_len > 0:
                # 部分匹配，需要分裂节点
                self._split_node(child, common_len)
                node = child.parent  # 分裂后的父节点（公共前缀）
                matched_len += common_len
                remaining = remaining[common_len:]
                break
            else:
                break
        
        return node, matched_len
    
    def _split_node(self, node, split_pos: int):
        """
        将 node 在 split_pos 处分裂：
        - 新建前缀节点（前 split_pos 个 token）
        - 原节点变成后缀节点（剩余 token）
        """
        # 新的父节点（公共前缀）
        prefix_node = TreeNode(node.token_ids[:split_pos])
        prefix_node.kv_blocks = node.kv_blocks[:split_pos // BLOCK_SIZE + 1]
        prefix_node.parent = node.parent
        
        # 原节点变成后缀
        node.token_ids = node.token_ids[split_pos:]
        # KV blocks 也要对应切分...
        
        # 更新父子关系
        prefix_node.children[node.token_ids[0]] = node
        node.parent.children[prefix_node.token_ids[0]] = prefix_node
        del node.parent.children[node.token_ids[0]]
        node.parent = prefix_node
    
    def insert(self, parent: TreeNode, new_token_ids: list[int], kv_blocks: list[int]):
        """在 parent 节点下插入新的子节点"""
        if not new_token_ids:
            return
        new_node = TreeNode(new_token_ids)
        new_node.kv_blocks = kv_blocks
        new_node.parent = parent
        new_node.ref_count = 1
        parent.children[new_token_ids[0]] = new_node
        return new_node
```

### 3.3 解码阶段：KV Cache 查找

在自回归解码的每一步，需要根据当前已生成的 token 序列，找到对应的 KV Cache。由于 Radix Tree 存储了前缀，解码时可以高效地定位到当前位置：

```python
def get_kv_blocks_for_position(self, token_ids: list[int]) -> list[int]:
    """
    根据完整的 token 序列，返回对应的 KV 物理块号列表
    （用于 PagedAttention kernel 的 block table）
    """
    node = self.root
    kv_blocks = []
    remaining = token_ids
    
    while remaining:
        first_token = remaining[0]
        if first_token not in node.children:
            break
        child = node.children[first_token]
        
        # 累加 KV blocks
        kv_blocks.extend(child.kv_blocks)
        
        # 向下走
        node = child
        remaining = remaining[len(child.token_ids):]
    
    return kv_blocks
```

## 四、缓存命中与未命中

### 4.1 Cache Hit 场景

| 场景 | 命中率 | 说明 |
|------|--------|------|
| 相同系统提示词 | 高 | 所有请求共享系统提示词的 KV |
| 相同 Few-shot 示例 | 高 | 同一任务类型共享示例 |
| 并行采样 | 极高 | 同一个 prompt 生成多个结果 |
| 多轮对话 | 中 | 历史对话部分可能共享 |
| 完全随机 prompt | 低 | 几乎没有共享前缀 |

### 4.2 缓存淘汰（Eviction）

Radix Tree 的节点有引用计数（ref_count）：
- 请求到达并匹配前缀时，路径上的节点 ref_count + 1
- 请求结束时，路径上的节点 ref_count - 1
- 当 ref_count = 0 时，节点可以被回收（LRU 策略）

```
引用计数示意:

          [root: ref=5]
              |
        [系统提示: ref=5]
        /        |        \
    [任务A:2] [任务B:2]  [任务C:1]
    /    \     /    \       |
  [req1][req2][req3][req4] [req5]
   ↑ 这 5 个活跃请求各持有引用
   
当 req1 结束：任务A.ref_count -= 1 → 1
如果任务A.ref_count 变成 0 且内存不够 → 可以回收任务A节点
```

## 五、与 vLLM PagedAttention 的对比

### 5.1 设计对比

| 维度 | PagedAttention (vLLM) | RadixAttention (SGLang) |
|------|----------------------|------------------------|
| 核心问题 | KV Cache 内存碎片 | 共享前缀的重复计算与存储 |
| 数据结构 | Block Table（每个请求独立） | Radix Tree（全局共享） |
| 物理存储 | 分页的 KV 块 | 分页的 KV 块（相同） |
| 前缀共享 | 不支持（每个请求独立） | 支持（树结构共享路径） |
| 适用场景 | 通用推理 | 有大量共享前缀的场景 |
| 实现复杂度 | 中等 | 较高 |

### 5.2 性能对比（相同硬件下）

| 工作负载 | vLLM (PagedAttention) | SGLang (RadixAttention) | 相对提升 |
|---------|----------------------|------------------------|---------|
| 短系统提示 + 随机用户输入 | 基准 | 略高（少量共享） | ~10-20% |
| 长系统提示 (512 token) | 基准 | 显著提升 | **2-3x** |
| Few-shot (10-shot) | 基准 | 大幅提升 | **3-5x** |
| 并行采样 (n=8) | 基准 | 极大提升 | **5-10x+** |
| 完全无共享前缀 | 基准 | 相当（略有 overhead） | ~0.9-1x |

> 数据为定性估计，实际提升取决于前缀长度、共享比例、batch size 等因素。

### 5.3 两者的关系

RadixAttention 不是 PagedAttention 的替代品，而是**在 PagedAttention 基础上的增强**：

- 物理 KV 存储都是分页的（借鉴 PagedAttention）
- RadixAttention 额外增加了前缀树的索引结构
- 可以认为是 PagedAttention + 前缀缓存

实际上，vLLM 后来也加入了前缀缓存功能（Prefix Caching），思路类似。

## 六、SGLang 架构概览

### 6.1 整体架构

```
SGLang 架构图：

┌─────────────────────────────────────────────────────────┐
│                    SGLang Runtime                       │
├─────────────┬───────────────────────────────────────────┤
│             │                                           │
│  Frontend   │  Scheduler + Tokenizer + Router           │
│  (OpenAI    │                                           │
│   API)      │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│             │  │ Schedule│  │ Batch   │  │ Dispatch│  │
│             │  │ Manager │  │ Manager │  │         │  │
├─────────────┴───────────────────────────────────────────┤
│                                                         │
│  Model Worker (每个 GPU 一个)                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Model Runner (Torch / Triton kernels)           │   │
│  │  - Attention kernel with RadixAttention          │   │
│  │  - MLP / Norm layers                             │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  RadixAttention Tree                             │   │
│  │  - 全局 KV 前缀树                                │   │
│  │  - 缓存命中 / 插入 / 淘汰                        │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Memory Manager                                  │   │
│  │  - KV 块分配器（类 PagedAttention）              │   │
│  │  - 显存管理                                      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 6.2 SGLang 的其他亮点

除了 RadixAttention，SGLang 还有一些特色设计：

| 特性 | 说明 |
|------|------|
| **SGLang 语言** | 结构化提示词 DSL，支持控制流、函数调用 |
| **函数调用优化** | 原生支持工具调用，减少 token 开销 |
| **FlashInfer** | 自研的 Attention kernel，性能优秀 |
| **Overlap Schedule** | 调度和计算重叠，减少 CPU overhead |
| **多 GPU 支持** | 支持张量并行和流水线并行 |

## 七、什么时候选 SGLang，什么时候选 vLLM

### 7.1 选择指南

| 场景 | 推荐框架 | 原因 |
|------|---------|------|
| 通用 API 服务，前缀不固定 | vLLM | 生态成熟，稳定性好 |
| 有大量共享前缀（系统提示 / Few-shot） | **SGLang** | RadixAttention 显著提升吞吐 |
| 并行采样、Beam Search | **SGLang** | 前缀缓存收益极大 |
| 需要最丰富的模型支持 | vLLM | 社区更大，模型支持更多 |
| 需要函数调用 / 工具调用 | **SGLang** | 原生支持，效率更高 |
| 生产环境稳定性优先 | vLLM | 经过更多生产验证 |
| 极致性能追求 | 都试试 | 取决于具体 workload |

### 7.2 趋势

两个框架正在互相借鉴、趋同：
- vLLM 加入了 Prefix Caching（前缀缓存）
- SGLang 在生态和模型支持上快速追赶
- 两者都在向更高性能、更易用的方向演进

作为工程师，理解背后的原理比纠结选哪个更重要。

## 八、总结

RadixAttention 是 SGLang 的核心创新，它的价值在于：

1. **洞察精准**：发现了 LLM 服务中"共享前缀"这个被忽视的优化机会
2. **设计优雅**：用 Radix Tree 这种经典数据结构解决 KV 缓存共享问题
3. **效果显著**：在有共享前缀的场景下，带来数倍的吞吐提升
4. **影响深远**：推动了整个推理框架的前缀缓存能力发展

从 PagedAttention 到 RadixAttention，我们可以看到 KV Cache 优化的演进脉络：
- 第一代：连续分配 → 问题：碎片严重
- 第二代：PagedAttention → 解决：碎片问题，支持连续批处理
- 第三代：RadixAttention → 新增：前缀共享，进一步提升利用率

未来还会有怎样的创新？值得期待。

---

**参考资料**：
- SGLang: Efficient Execution of Structured Language Model Programs
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [RadixAttention 技术博客](https://lmsys.org/blog/2024-01-17-sglang/)
