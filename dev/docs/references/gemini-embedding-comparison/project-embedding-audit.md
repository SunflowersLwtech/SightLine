# SightLine 项目 Embedding 使用审计报告

> 审计日期：2026-03-13
> 范围：全代码库 embedding 相关代码、配置、基础设施

## 1. Embedding 系统概览

项目中有两套独立的 embedding 系统：

| 系统 | 模型 | 维度 | API | 认证 |
|------|------|------|-----|------|
| Memory + Entity | gemini-embedding-001 | 2048-D | Google AI API | api_key |
| Face Recognition | InsightFace ArcFace (buffalo_l) | 512-D | 本地推理 | N/A |

## 2. Memory Embedding 系统

### 2.1 核心代码

**文件**: `memory/memory_bank.py`

```python
# Line 20-21: 常量定义
EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = 2048

# Line 24-47: 嵌入计算
def _compute_embedding(text: str) -> Optional[list[float]]:
    from google import genai
    api_key = os.environ.get("_GOOGLE_AI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    client = genai.Client(api_key=api_key, vertexai=False)
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=normalized,
        config={"output_dimensionality": EMBEDDING_DIM},
    )
    return result.embeddings[0].values
```

### 2.2 存储

```python
# memory/memory_bank.py:140-158
from google.cloud.firestore_v1.vector import Vector

doc_data = {
    "content": content,
    "category": category,
    "importance": importance,
    "timestamp": now,
    "memory_layer": memory_layer,
    "entity_refs": entity_refs or [],
    "location_ref": location_ref,
    "half_life_days": half_life,
    "access_count": 0,
    "last_accessed": now,
}
if embedding:
    doc_data["embedding"] = Vector(embedding)
```

存储路径: `user_profiles/{user_id}/memories/{memory_id}`

### 2.3 向量搜索

```python
# memory/memory_bank.py:217-251
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

vector_query = coll.find_nearest(
    vector_field="embedding",
    query_vector=Vector(query_embedding),
    distance_measure=DistanceMeasure.COSINE,
    limit=top_k,
    distance_result_field="vector_distance",
)

# 距离转相似度: Firestore COSINE ∈ [0, 2]
# relevance = 1.0 - (distance / 2.0) → [0, 1]
```

### 2.4 降级方案

当向量搜索不可用时，降级为基于关键词的 Jaccard 相似度文本搜索 (`_text_fallback`, line 253-285)。

## 3. Entity Graph Embedding

**文件**: `context/entity_graph.py:10, 42, 167-169`

```python
embedding: list[float] = field(default_factory=list)  # 2048-D

# 存储
if entity.embedding:
    from google.cloud.firestore_v1.vector import Vector
    data["embedding"] = Vector(entity.embedding)
```

存储路径: `user_profiles/{user_id}/entities/{entity_id}`

## 4. Face Recognition Embedding（不受 Gemini 迁移影响）

**文件**: `agents/face_agent.py`

```python
# Line 80-113: InsightFace 初始化
app = FaceAnalysis(
    name="buffalo_l",
    root=os.path.expanduser("~/.insightface"),
    providers=["CPUExecutionProvider"],
    allowed_modules=["detection", "recognition"],
)

# 嵌入生成
embedding = faces[0].normed_embedding  # (512,) L2-normalized

# 匹配 (dot product == cosine for L2-normed vectors)
similarities = lib_embeddings @ embedding
# MATCH_THRESHOLD = 0.4
```

**文件**: `tools/face_tools.py:122-128`

```python
doc_data = {
    "person_name": person_name,
    "relationship": relationship,
    "embedding": Vector(embedding.tolist()),
}
```

存储路径: `user_profiles/{user_id}/face_library/{face_id}`

## 5. Firestore Vector Index 定义

**文件**: `infrastructure/terraform/firestore.tf`

```terraform
# memories (2048-D, COSINE)
resource "google_firestore_index" "memories_vector" {
  collection = "memories"
  fields {
    field_path = "embedding"
    vector_config {
      dimension = 2048
      flat {}
    }
  }
}

# face_library (512-D, COSINE)
resource "google_firestore_index" "face_library_vector" {
  collection = "face_library"
  fields {
    field_path = "embedding"
    vector_config {
      dimension = 512
      flat {}
    }
  }
}
```

## 6. Memory Ranking（后处理）

**文件**: `memory/memory_ranking.py`

向量搜索找到候选后，使用 5 维复合评分排序：
- **Relevance**: 0.30 权重（来自向量相似度）
- **Recency**: 0.20 权重（指数衰减 + half-life）
- **Importance**: 0.15 权重（用户标注）
- **Location**: 0.20 权重（空间距离）
- **Entity Overlap**: 0.15 权重（场景实体交集）

## 7. 环境变量

| 变量 | 默认值 | 用途 |
|------|--------|------|
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | 嵌入模型 ID |
| `_GOOGLE_AI_API_KEY` / `GOOGLE_API_KEY` | — | Google AI API 认证 |
| `GOOGLE_CLOUD_PROJECT` | `sightline-hackathon` | GCP 项目 |

## 8. 依赖

```
# requirements.txt
google-adk==1.25.1        # 包含 google-genai SDK
google-cloud-firestore==2.23.0  # Vector 支持
insightface==0.7.3        # Face recognition
numpy>=1.24.0,<2.0        # insightface 兼容性约束
```

## 9. 硬编码维度值汇总

| 文件 | 行号 | 维度 | 类型 |
|------|------|------|------|
| `memory/memory_bank.py` | 21 | 2048 | 代码常量 |
| `infrastructure/terraform/firestore.tf` | 52 | 2048 | Terraform |
| `infrastructure/terraform/firestore.tf` | 32 | 512 | Terraform |
| `agents/face_agent.py` | 72, 104 | 512 | 文档注释 |
| `context/entity_graph.py` | 10 | 2048-D | 文档注释 |
| `README.md` | 54, 83, 84 | 2048-D, 512-D | 文档 |
| `assets/context-memory.svg` | 67, 138 | 2048-D, 2048 | SVG 图形 |

## 10. 数据流图

```
用户输入 (文本/图片)
    │
    ├── [Memory 系统]                    [Face 系统]
    │       │                                │
    │   文本内容                          人脸图片 (BGR)
    │       │                                │
    │   _compute_embedding()             InsightFace buffalo_l
    │   (gemini-embedding-001)           ArcFace (512-D)
    │       │                                │
    │   2048-D vector                    512-D L2-normalized
    │       │                                │
    │   Firestore Vector                 Firestore Vector
    │       │                                │
    │   [向量检索]                        [人脸匹配]
    │       │                                │
    │   新上下文 → _compute_embedding()   新人脸 → generate_embedding()
    │       │                                │
    │   find_nearest (COSINE)            dot product
    │       │                                │
    │   top-k memories                   threshold check (0.4)
    │       │                                │
    │   5 维复合排序                      识别/注册
    │       │                                │
    └── 注入对话上下文                    语音播报识别结果
```

## 11. 完整文件列表

### 核心实现
1. `memory/memory_bank.py` — Memory embedding 生成、存储、搜索
2. `agents/face_agent.py` — Face embedding 生成、匹配
3. `tools/face_tools.py` — Face 注册（含 embedding 存储）
4. `context/entity_graph.py` — Entity embedding 存储
5. `memory/memory_extractor.py` — Memory 提取（导入 EMBEDDING_DIM 用于去重）

### 配置 & 基础设施
6. `infrastructure/terraform/firestore.tf` — Vector index 定义
7. `requirements.txt` — 依赖
8. `CLAUDE.md` — 项目说明

### 排序 & 工具
9. `memory/memory_ranking.py` — 后处理排序（使用 embedding 相似度分数）
10. `memory/memory_tools.py` — Function calling 工具

### 测试
11. `tests/test_memory_bank.py` — Embedding mock
12. `tests/test_memory_extractor.py` — 去重 mock
13. `tests/test_face_agent.py` — Face matching mock
14. `tests/test_entity_graph.py` — Entity embedding mock

### 脚本 & 种子
15. `scripts/seed_firestore.py` — Demo 数据

### 文档 & 资产
16. `README.md` — 架构文档
17. `assets/context-memory.svg` — 架构图
18. `assets/system-architecture.svg` — 系统图
19. `memory/__init__.py` — 模块导出
