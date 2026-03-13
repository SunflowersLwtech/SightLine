# Gemini Embedding 模型研究资料

> 调研日期：2026-03-13
> 目的：评估 gemini-embedding-001 → gemini-embedding-2-preview 迁移可行性

## 文件索引

| 文件 | 内容 |
|------|------|
| `embedding-001-spec.md` | gemini-embedding-001 完整规格、benchmark、架构 |
| `embedding-2-preview-spec.md` | gemini-embedding-2-preview 完整规格、新特性 |
| `comparison-and-migration.md` | 1.0 vs 2.0 对比 + SightLine 迁移成本分析 |
| `project-embedding-audit.md` | SightLine 代码库 embedding 使用审计报告 |

## 关键结论

- 2.0 核心突破：**多模态嵌入**（文本+图片+视频+音频+PDF）+ **4x 输入上限**
- 嵌入空间**不兼容**，迁移必须全量重嵌入
- 2.0 目前 **Preview**，无 Vertex AI / Batch API，**建议等 GA 再迁移**
