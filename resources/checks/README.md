# 编码检查与报告

本目录保存 BioBERT 构建、数值一致性和性能检查源码，以及体积有限的验收记录。项目内代码和数据路径均以 `PATHAK/` 为起点。

- [修复报告](reports/BIOBERT_FIX_REPORT.md)：PAD pooling 问题、严格兼容方案与历史验收。
- [性能报告](reports/BIOBERT_PERFORMANCE_REPORT.md)：CPU8 正式配置、剖析、三轮测速与未采用候选。
- [目录整理验收](embedding-layout-reorganization.json)：模型重命名、路径迁移、文件身份和加载复核。
- [模型身份](../../config/biobert_model.json)：唯一正式模型与本地历史模型路径映射。

正式模型是 `models/biobert_pubmed_sentence/`。原模型、dense masked mean 候选和 CPU2 稳定基线保存在 `.backups/biobert-models/`，只在历史复测和来源校验中使用。生产预测不需要读取它们。

在 pathak 环境、项目根目录运行快速复核：

```bash
python resources/checks/check_biobert_parallel_batching.py --report resources/checks/embedding-layout-postcheck.json
```

完整公平测速入口为 `benchmark_biobert_global8.py`，`--prepare-only` 只核对并准备输入。该入口复制本目录 `biobert-performance/` 中适配当前布局的源码，以及原审计目录中的冻结输入和 JAR，到新的 `.backups/` 子目录。原始审计目录不修改。

历史 `embedding-*.json` 保留实测时的路径、时间和校验值，不能将旧路径视为当前生产加载位置；当前路径映射见模型身份文件。大型 trace、编译结果、模型、向量和隔离环境保留在本地，不纳入普通 Git 提交。小型 `.py`、`.java`、报告、验收 JSON 和冻结的 41 条单句数值参考允许提交。
