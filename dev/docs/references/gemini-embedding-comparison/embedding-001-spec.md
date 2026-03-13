# gemini-embedding-001 完整规格

> 来源：Google AI 官方文档、arXiv:2503.07891
> 调研日期：2026-03-13

## 1. 模型概览

- **Model ID**: `gemini-embedding-001`
- **状态**: Stable (GA, 2025-06)
- **论文**: "Gemini Embedding: Generalizable Embeddings from Gemini" (arXiv:2503.07891)
- **定位**: 统一模型，合并了 text-embedding-005 (英文/代码) 和 text-multilingual-embedding-002 (多语言) 的能力

## 2. 技术规格

| 属性 | 值 |
|------|-----|
| 输入模态 | 纯文本 |
| 最大输入 | 2,048 tokens/条 |
| 单次请求上限 | 250 条文本 / 20,000 tokens |
| 默认输出维度 | 3,072 |
| 可选维度范围 | 128 ~ 3,072 (MRL) |
| 推荐维度 | 768, 1,536, 3,072 |
| 归一化 | 3,072-D 已预归一化；其他维度需手动 L2 归一化 |
| 距离度量 | Cosine / Dot Product / Euclidean |
| autoTruncate | 默认 true，超长静默截断 |
| 语言支持 | 100+ 语言 |

## 3. 架构与训练

### 架构
- 基于 Gemini LLM 初始化（具体版本/参数量未公开）
- Transformer + **双向注意力**（非因果 LLM 的单向）
- **Mean Pooling** → **线性投影层** → d=3,072
- **Matryoshka Representation Learning (MRL)**: 推理时灵活选择维度，无需重训

### 两阶段训练

**Stage 1 — Pre-finetuning:**
- 十亿级 web 语料，title-passage 正样本对
- 大 batch size，长训练步数

**Stage 2 — Fine-tuning:**
- (query, target, hard_negative) 三元组
- 多任务混合数据集，每 batch 单数据集
- 小 batch size (< 1,024)
- 超参网格搜索

### 其他技术
- **损失函数**: NCE (Noise Contrastive Estimation) + cosine + temperature τ
- **Model Soup**: 多个 fine-tuned checkpoint 参数平均
- **合成数据**: FRet/SWIM-IR 多阶段提示生成
- **数据过滤**: MIRACL 平均提升 +3.9, 分类任务提升 +17.6

## 4. 支持的 Task Types (8 种)

| Task Type | 描述 |
|-----------|------|
| `SEMANTIC_SIMILARITY` | 文本相似度 |
| `CLASSIFICATION` | 文本分类 |
| `CLUSTERING` | 文本聚类 |
| `RETRIEVAL_DOCUMENT` | 文档索引（支持 title 参数）|
| `RETRIEVAL_QUERY` | 搜索查询（默认）|
| `CODE_RETRIEVAL_QUERY` | 自然语言检索代码 |
| `QUESTION_ANSWERING` | QA 文档检索 |
| `FACT_VERIFICATION` | 事实核查证据检索 |

## 5. 性能 Benchmark

### MTEB 各维度分数

| 维度 | MTEB 分数 |
|------|----------|
| 3,072 | 68.32 |
| 2,048 | 68.16 |
| 1,536 | 68.17 |
| 768 | 67.99 |
| 512 | 67.55 |
| 256 | 66.19 |
| 128 | 63.31 |

### MTEB 排名

| 赛道 | 分数 | 排名 |
|------|------|------|
| 多语言 (Task Mean) | 68.32 | #1 (领先第二名 +5.09) |
| 英文 v2 (Task Mean) | 73.30 | #1 |
| 代码 (Mean All) | 74.66~75.5 | #1 |

### 多语言子项

| 任务 | 分数 |
|------|------|
| Bitext Mining | 79.32 |
| Classification | 71.84 |
| Clustering | 54.99 |
| Pair Classification | 83.64 |
| Reranking | 65.72 |
| Retrieval | 67.71 |
| STS | 79.40 |
| Instruction Retrieval | 5.18 (弱项) |

### 跨语言检索

| 模型 | XOR-Retrieve Recall@5k |
|------|----------------------|
| **gemini-embedding-001** | **90.42** |
| Cohere-embed-multilingual-v3.0 | 68.76 |
| Gecko Embedding | 65.67 |

### 低资源语言 (XTREME-UP)

| 模型 | Avg MRR@10 |
|------|-----------|
| **gemini-embedding-001** | **64.33** |
| voyage-3-large | 39.2 |
| Gecko i18n | 35.0 |

### MIRACL 多语言检索
- 18 语言平均: 70.1 (含数据过滤) vs 59.8 (无过滤)

## 6. 定价

| 层级 | 价格 |
|------|------|
| Free | $0（数据可被 Google 用于改进产品）|
| Paid | $0.15 / 1M input tokens |
| Paid Batch | $0.075 / 1M input tokens (50% 折扣) |

## 7. SDK 用法 (Python)

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_API_KEY")

# 基础嵌入
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="What is the meaning of life?"
)

# 批量 + task type + 自定义维度
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["text1", "text2", "text3"],
    config=types.EmbedContentConfig(
        task_type="SEMANTIC_SIMILARITY",
        output_dimensionality=768
    )
)

# 手动归一化 (非 3072-D 时必须)
import numpy as np
vec = np.array(result.embeddings[0].values)
normed = vec / np.linalg.norm(vec)
```

SDK: `google-genai` (v1.67.0+, Python >= 3.10)

## 8. 已知限制

1. **纯文本** — 不支持图片/视频/音频/PDF
2. **2,048 token 上限** — 较短
3. **非 3072-D 需手动归一化**
4. **与 2-preview 嵌入空间不兼容**
5. **Instruction Retrieval 弱** — MTEB 仅 5.18
6. **基座模型不透明** — 未公开具体 Gemini 版本和参数量

## 9. 关键链接

- 官方文档: https://ai.google.dev/gemini-api/docs/embeddings
- 模型页面: https://ai.google.dev/gemini-api/docs/models/gemini-embedding-001
- 论文: https://arxiv.org/abs/2503.07891
- 定价: https://ai.google.dev/pricing
- Vertex AI: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
- Rate Limits: https://ai.google.dev/gemini-api/docs/rate-limits
- PyPI: https://pypi.org/project/google-genai/
