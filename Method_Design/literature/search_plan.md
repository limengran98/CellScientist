# Literature Search Plan (Phase 2)
_Date: 2025-10-16_

## 目标领域（示例）
- BBBC036 / Broad Bioimage Benchmark Collection（细胞显微图像数据集）
- 细胞表型表征、形态学特征工程、图/超图建模
- 剂量-响应（dose-response）、批次效应（batch effect）
- 表型网络/模块、通路富集、可解释性
- 模型基线与评测标准（AUROC/AUPRC、R²/MAE 等）

## 建议检索引擎
- Scholar / PubMed / arXiv（bioRxiv/medRxiv）/ OpenAlex
- 期刊官网（Nature Methods, Bioinformatics, PLoS Comp Bio 等）
- 数据集/工具官方文档（BBBC、CellProfiler、scikit-image、Scanpy 等）

## 关键词拼接（示例）
- "BBBC036" OR "Broad Bioimage Benchmark Collection" OR "Cell Painting"
- "hypergraph" OR "graph neural network" OR "graph-based phenotyping"
- "phenotypic profiling" OR "morphological profiling" OR "cell painting assay"
- "dose response" OR "batch effect" OR "plate effect"
- "feature selection" OR "robust scaling" OR "variance stabilization"

## 纳入/排除标准
- ✅ 与细胞成像表型/图结构/剂量批次相关；提供方法或可复现实证。
- ❌ 无法追溯来源、非学术可信来源、与任务无关评论性文章。

## 记录与落盘
- 统一在 `papers.csv` 录入；对应 BibTeX 存入 `references.bib`；合并摘要写入 `related_work.md`。
- 每条目 ID 如 `R1, R2, ...`，在 MD 与 CSV/BibTeX 中保持一致。
