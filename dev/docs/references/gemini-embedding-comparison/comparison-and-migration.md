# Gemini Embedding 1.0 vs 2.0 对比 & SightLine 迁移分析

> 调研日期：2026-03-13

## 一、模型对比总表

| 属性 | gemini-embedding-001 (1.0) | gemini-embedding-2-preview (2.0) |
|------|---------------------------|----------------------------------|
| 状态 | Stable (GA) | Preview |
| 发布日期 | 2025-07-14 (Stable) | 2026-03-10 (Preview) |
| 输入模态 | 纯文本 | 多模态（文本+图片+视频+音频+PDF）|
| 输入上限 | 2,048 tokens | 8,192 tokens (4x) |
| 输出维度 | 128 ~ 3,072 (默认 3,072) | 128 ~ 3,072 (默认 3,072) |
| MRL | 是 | 是 |
| 语言 | 100+ | 100+ |
| Task Types | 8 种 | 8 种（相同）|
| Batch API | 有 (50% 折扣) | 暂无 |
| Vertex AI | 有 | 暂无 |
| 文本定价 | $0.15/1M tokens | $0.20/1M tokens (+33%) |
| 论文 | arXiv:2503.07891 | 未发表 |
| MTEB 多语言 | 68.32 (#1) | 未公布 |
| MTEB 英文 | 73.30 (#1) | 未公布 |
| 嵌入空间 | 空间 A | 空间 B（**不兼容**）|

## 二、核心区别分析

### 区别大不大？—— 架构差异大，API 接口几乎不变

#### 重大变化 (Breaking)
1. **多模态嵌入** — 2.0 最大突破，文本/图片/视频/音频/PDF 映射到统一向量空间
2. **4x 输入长度** — 2K → 8K tokens
3. **嵌入空间不兼容** — 001 和 2-preview 向量不能混合比较

#### 保持不变
1. API 接口 — 同样的 `embed_content()` 调用
2. 维度范围 — 128 ~ 3,072，MRL 方式相同
3. Task Types — 8 种完全相同
4. SDK — 同一个 `google-genai` 包

#### 退步 / 限制
1. 价格上涨 33% (文本)
2. 无 Batch API（大规模操作不便）
3. 无 Vertex AI（无法用 ADC 认证）
4. Preview 状态，有 breaking change 风险
5. 无公开 benchmark，性能不明

## 三、SightLine 项目迁移成本

### 3.1 当前 Embedding 使用概况

| 系统 | 模型 | 维度 | 存储 | 搜索方式 |
|------|------|------|------|---------|
| Memory | gemini-embedding-001 | 2048-D | Firestore Vector | find_nearest (COSINE) |
| Entity Graph | gemini-embedding-001 | 2048-D | Firestore Vector | — |
| Face Recognition | InsightFace ArcFace | 512-D | Firestore Vector | numpy dot product |

**Face 系统不受影响**（使用本地 InsightFace 模型，与 Gemini 无关）。

### 3.2 受影响的文件

#### 代码文件

| 文件 | 行号 | 改动内容 |
|------|------|---------|
| `memory/memory_bank.py` | 20-21 | EMBEDDING_MODEL 常量 + EMBEDDING_DIM |
| `memory/memory_bank.py` | 34-47 | `_compute_embedding()` 函数（仅改 model 名）|
| `context/entity_graph.py` | 10 | embedding 维度注释 |
| `memory/memory_extractor.py` | 15 | 导入 EMBEDDING_DIM |

#### 基础设施

| 文件 | 行号 | 改动内容 |
|------|------|---------|
| `infrastructure/terraform/firestore.tf` | 52 | memories vector index dimension（仅维度变时）|

#### 测试文件

| 文件 | 改动内容 |
|------|---------|
| `tests/test_memory_bank.py` | mock 维度 `[0.1] * 2048` |
| `tests/test_memory_extractor.py` | 重复检测 mock |
| `tests/test_face_agent.py` | 不受影响 |
| `tests/test_entity_graph.py` | entity embedding mock |

#### 文档/资产

| 文件 | 改动内容 |
|------|---------|
| `README.md:54,83,84` | 维度说明 |
| `CLAUDE.md` | Embedding 模型 ID 和维度 |
| `assets/context-memory.svg:67,138` | 架构图维度标注 |
| `assets/system-architecture.svg` | 系统架构图 |

### 3.3 迁移场景矩阵

| 场景 | 代码改动 | 数据迁移 | 基础设施 | 风险 | 推荐 |
|------|---------|---------|---------|------|------|
| **A: 换模型 + 保持 2048-D** | 改 1 行 | 全量重嵌入 | 无需改 | 低 | -- |
| **B: 换模型 + 改维度** | 改 2 行 + 测试 | 全量重嵌入 + 索引重建 | terraform apply | 中 | -- |
| **C: 暂不迁移 (推荐)** | 无 | 无 | 无 | 无 | **推荐** |

### 3.4 数据迁移工作量

若选择迁移（场景 A 或 B）：

1. **编写迁移脚本** — 遍历 Firestore `memories` 和 `entities` 集合，对每条记录调用新模型重算 embedding
2. **限流处理** — 2-preview 无 Batch API，需自行实现 rate limiting
3. **索引重建** — 如果维度变化，需先删除旧 vector index，创建新 index，等待构建完成
4. **验证** — 对比迁移前后的搜索质量，确保 recall 不下降
5. **回滚方案** — 保留旧 embedding 字段（如 `embedding_v1`）直到验证通过

预估工作量：
- 脚本开发: ~2h
- 数据迁移执行: 取决于数据量（当前规模应 < 1h）
- 验证测试: ~2h
- 总计: ~半天（不含等待索引构建时间）

## 四、建议

### 短期（现在）
**暂不迁移**。原因：
1. 2-preview 仍是 Preview，不适合生产
2. 无 Vertex AI 支持 — 我们的 Live API 走 Vertex AI (ADC)，memory/search 走 Google AI API (api_key)，但保持一致性更好
3. 无 Batch API — 全量重嵌入操作不便
4. 001 在 MTEB 上 #1，性能无问题
5. 2-preview benchmark 未公布，性能未知

### 中期（2-preview GA 后）
**评估迁移**，关注：
1. GA 发布 + Vertex AI 支持
2. Batch API 上线
3. 公开 benchmark 对比
4. 多模态嵌入对 SightLine 的价值（场景图片纳入 memory 语义搜索）

### 长期价值
2.0 的**多模态嵌入**对 SightLine 有潜在重大价值：
- 用户看到的场景图片可以直接嵌入 memory，不再只有文本描述
- 跨模态搜索：用文字描述搜索之前看过的场景
- 音频记忆：对话录音也可纳入向量检索
- 这些都需要 GA + 充分测试后再考虑

### 维度建议
无论何时迁移，**建议保持 2048-D**：
- 001 的 benchmark 显示 2048-D (68.16) 与 3072-D (68.32) 几乎无差异
- 节省 ~33% 存储和计算成本
- 避免 Firestore vector index 重建
