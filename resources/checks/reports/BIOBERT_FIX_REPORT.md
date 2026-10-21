# BioBERT 批量编码修复：实施与验收报告

> 目录整理：下文代码和数据路径以 `PATHAK/` 为起点；Markdown 证据链接按当前文件位置解析。正式模型为 `models/biobert_pubmed_sentence/`，历史模型归档在 `.backups/biobert-models/`。历史 JSON 保留测量发生时的原始路径和校验值，当前映射见 `config/biobert_model.json`。

执行日期：2026-10-22（Asia/Tokyo）。服务器 stage3；Conda 环境 `pathak`。项目内路径均相对于 `PATHAK/`。

## 当前正式资源配置

当前默认已采用全局 intra-op=8、inter-op=8、句内上限 2、OMP=2，batch=16，模型目录 `models/biobert_pubmed_sentence/`。785 条固定 GO/文献混合语料三轮端到端中位 36.350 → 12.657 秒，约 2.872 倍；全部 batch1/8/16 的 5,117 条定义仍为零差值，标准未放宽。图和权重与 CPU2 稳定版相同。

具体剖析、资源代价、拒绝候选、模型身份与 Notebook 步骤见 [BIOBERT_PERFORMANCE_REPORT.md](BIOBERT_PERFORMANCE_REPORT.md)。下文保留 CPU2 稳定基线的实施与验收记录。此前 batch8/16 的耗时接近，微小差值不作为 16 显著更快的证据。

## CPU2 稳定基线的实施与验收

正式入口已切换为 **batchSize=16**，协议为 `masked_mean_parallel_sentence_v1`。batch=8、16 的全部 **5,117 × 768** 个向量数值均与原模型 batch=1 库相等，最大绝对差为 **0**，行对应全部通过。原阈值仍为 `max_abs <= 1e-5` 且 `cosine >= 0.999999`。

相同线程设置下，256 条代表性 GO 定义的三轮编码中位数从 10.636 秒变为 9.340 秒，吞吐从 24.07 变为 27.41 条/秒，约 **1.139 倍**。这是约 14% 的吞吐提升，幅度有限；不能据此声称有数倍提速。batch=8 与 16 的速度接近，默认 16 依据轮换测速的中位数选择，8 也已通过全库核查。

总体机器记录：[embedding-parallel-fix.json](../embedding-parallel-fix.json)。研究进展记录在 [02 分析进展](../../../docs/02_mouse_embryo_pathak_analysis_PROGRESS.md)。

## 根因：PAD、编码器数值路径及函数封装后的运行时重写

原模型归档内 `saved_model.pb` 实际是 GraphDef。原路径为 `bert/encoder/Reshape_13 → Mean(axis=1) → pooled_output`，平均包含 PAD。Spark NLP 按批内最长 WordPiece 长度补齐并传入 input_mask，attention mask 不会将 PAD 的最终 hidden states 自动置零。这解释了最初很大的单句/混合差异。

仅修正 pooling 后，完整库仍有两条定义超出 1e-5。新一轮逐层探针进一步确认：对敏感句，有效位置的输入 embedding、LayerNorm 均值、方差及加 epsilon 后的值相同，首个差异出现在 `Rsqrt`。单句末尾位置与批量计算的向量化/尾部路径产生约 9.537e-7 的差别，后续层会放大它。重复相同文本 8 次也会出现位置相关差异，故只按有效长度分组仍不充分。

显式设置线程、关闭矩阵乘法融合、关闭 oneDNN 开关，均未消除所测残余差异。提高 LayerNorm、归约或 Rsqrt 精度会改变旧单句结果，未满足全部严格比较。只让 Rsqrt 按句执行也仍有其他超线样本。这些试验均未进入正式实现。

完整逐句函数封装首先实现了新模型单句/批量结果相同，但与旧单句仍有一条超线。对照实际运行图发现，旧管线的 Grappler 会把 GELU 中 Pow、乘法和系数重新结合；未优化的 FunctionDef 没有同样的运算顺序。因此正式函数从**旧管线实际执行的优化后图**提取，保留原来的浮点运算顺序。之后 8 条敏感句及扩展检查均为零差值。

诊断证据：[逐算子首差](../embedding-parallel-first-difference.json)、[逐层比较](../embedding-parallel-layer-differences.json)、[运行时和失败精度试验](../embedding-parallel-runtime-diagnostic.json)。

## 正式图如何执行批量

1. 保留图内 float32 masked mean：`sum(hidden * cast(input_mask)[..., None], axis=1) / sum(cast(input_mask)[..., None], axis=1)`。保留 CLS/SEP，仅排除 PAD。
2. 使用 TensorFlow `MapDefun`，从输入批次中取各句并按 mask 裁掉补齐，函数内保持单句的编码形状。
3. 共享 197 个实际用于前向计算的权重输入；完整原 checkpoint 的 199 个权重张量仍保留，权重数据与索引校验值均与原模型相同。
4. 正常管线的 batchSize 为 16，一次 feed 整批输入，fetch 仍只返回每句 768 维向量。token hidden states 不返回 Python，诊断文件中的 hidden states 只用于定位问题。

本实现是共享权重的图内逐句并行，保留每句计算形状。速度受这种执行方式和当前 Java TensorFlow CPU 后端约束。它没有把不同句子的编码器矩阵合并为一次大矩阵乘法。

运行环境保留 Spark NLP 4.0.2 / NLU 4.0.0 / Java TensorFlow 2.7.1，未升级或重建环境。正常预测仍只加载原来的 Spark NLP assembly JAR。新增 Java 工具用于构建和诊断，不加入日常预测的额外 classpath。

官方实现参照：[Spark NLP TensorflowBert.scala](https://github.com/JohnSnowLabs/spark-nlp/blob/4.0.2/src/main/scala/com/johnsnowlabs/ml/tensorflow/TensorflowBert.scala)、[TensorFlow 2.7.1 MapDefun CPU 内核](https://github.com/tensorflow/tensorflow/blob/v2.7.1/tensorflow/core/kernels/data/map_defun_op.cc)。

## 模型身份和加载代码

| 对象 | 路径或 SHA256 |
| --- | --- |
| 原模型，保留 | `.backups/biobert-models/sent_biobert_pubmed_base_cased/` |
| 正式派生模型 | `.backups/biobert-models/sent_biobert_pubmed_base_cased_parallel_masked_mean/` |
| 原模型归档 | `165a38e20e3b53569947fb3af8803deb8132e75e81ab75d1337b4c7b4d40cf90` |
| 正式派生归档 | `3583a8f750a8891440c1338a46399a9f18a2fc60f5db113506183e0a83c4161b` |
| 正式 GraphDef | `8e5f288ae33ece7f72783989fa83f60ad562e4b1df9f4ada0256de0411f58ba7` |
| 两者共同 variables.data | `0c861e31aee67b085d618f582a5a7fcfe4129753f9578281c45ce86a8ede6c22` |
| 两者共同 variables.index | `0121af1dfa09816805a6fe8185d25565f8b26d979b0b5c658a73995e2b7d0289` |

词表逐文件一致；大小写敏感、最大长度 128、CLS/SEP 与截断规则保留。原模型与仅 masked mean 的失败候选均保留。详细身份记录见 [embedding-parallel-model.json](../embedding-parallel-model.json) 和正式模型的 `pathak_pooling.json`。

`pathak_runtime.py` 的 `load_biobert()` 默认读正式派生模型；`configure_biobert_pipeline()` 根据持久化 storageRef 识别新协议，设置 batch=16，原始导出仍保持安全的 batch=1。保存后重载不会再把新模型强制改回 1。

Spark NLP 在构造 TensorflowBert 时捕获 ConfigProto；加载后只改 Python 参数不会改变已创建的后台配置。本轮将 `[16, 2, 40, 8]` 写入派生模型 metadata 后再加载，分别表示 intra-op=2、inter-op=8。函数内 intra-op 上限为 2；保存重载和新进程均核对了实际捕获的字节。

## 一致性验收

| 检查 | 结果 |
| --- | --- |
| 40 条文本、45 项比较，batch=1/8/16 | 最大绝对差 0 |
| 单句、短长混合、重排、17 次重复敏感句、list/非默认索引 Series、不同调用分块 | 全部通过 |
| 接近与超过最大长度 | 包含 124–127 个 cell 及 600 次重复文本；通过 |
| 保存重载、独立新进程 3 项比较 | 最大绝对差 0 |
| 正式产物的新进程检查 13 项 | 最大绝对差 0，包含全部保存向量与旧单句库比较 |
| 原 `check_embedding_batch_invariance.py` 命令 | 已兼容新 manifest 和 batch 设置，回归通过，最大绝对差 0 |

前两组扩展与序列化检查最初在 2/2 线程下完成；最终 2/8 配置另经过完整库、正式模型、实际捕获配置、保存管线及长度边界的新进程检查。完整记录见 [扩展检查](../embedding-parallel-validation.json)、[序列化新进程](../embedding-parallel-validation-fresh.json)、[正式产物新进程](../embedding-parallel-postcheck.json)。

| batch | 全库形状 | max_abs | 最小 cosine | 完整编码秒 |
| --- | --- | ---: | ---: | ---: |
| 8 | 5117 × 768 | 0 | 0.9999999999999998 | 193.645 |
| 16 | 5117 × 768 | 0 | 0.9999999999999998 | 195.944 |

每条定义与 GO 表、旧库逐行对应，全部值有限且范数非零，拒绝行数均为 0。正式库使用本轮 batch=16 完整编码结果。全库单次时间另列，不用于计算四配置的公平测速加速比。详细结果见 [全库记录](../embedding-parallel-full-library.json)。

旧的仅 masked mean 候选仍在同样标准下被拒绝两条记录（GO:0006303、GO:0090558），负对照见 [embedding-parallel-negative-control.json](../embedding-parallel-negative-control.json)。没有通过放宽阈值使它变成合格模型。

## 速度测量及采用决定

同一 stage3 硬件和软件，Spark local[2]，OMP=2、TF intra-op=2、inter-op=8。原模型后台捕获 ConfigProto 为 None，使用相同环境线程设置；派生模型在加载前显式保存 2/8 配置。256 条定义按长度排名等距抽样后固定打乱，三轮轮换配置顺序。计时包含预热后的 predict、行对齐和向量提取，排除加载和预热。加载是在同一 Spark 会话内的本地加载，首次预测含延迟初始化。

| 配置 | 本地加载秒 | 首次 32 条预测秒 | 256 条编码中位秒 | 条目/秒 | 相对原模型 batch=1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原模型 batch=1 | 6.753 | 4.821 | 10.636 | 24.07 | 1.000× |
| 修正版 batch=1 | 4.575 | 2.145 | 10.687 | 23.96 | 0.995× |
| 修正版 batch=8 | 4.330 | 2.198 | 9.370 | 27.32 | 1.135× |
| 修正版 batch=16 | 4.433 | 1.645 | 9.340 | 27.41 | 1.139× |

上述每轮结果均与原单句输出相等。采用 batch=16；batch=8 同样正确、速度接近。旧 2/2 设置的对照中批量收益很小，完整记录仍保留为 `resources/checks/embedding-parallel-benchmark-threads2.json`。增加到 intra-op=8、函数内上限=1 的激进试验未通过严格比较，没有正式测速结论，未部署。

不能用不同硬件/线程、未完成的失败试验或旧 247 秒历史记录计算加速比。完整样本行号、加载、预热和各轮时间见 [正式公平测速](../embedding-parallel-benchmark.json)。

## 产物、重跑与重现

已更新 `pathak_runtime.py`、相关 Notebook 加载/保存单元、`tutorials/predictions_full_BP.pickle`、`tutorials/pipe_full_BP/` 和 `tutorials/embedding_manifest.json`。manifest 保存 pooling 协议、模型与语料校验值、batch、线程配置及保存管线逐文件校验值。Notebook 2 重建时会重新写清单，Notebook 3 加载时会核对配套关系。

本轮已完成全库重建，使用现有产物时无需重做 GO 准备或再编码一次。已有 Jupyter 内核需要重启，再运行 Notebook 3 的初始化、读取 GO/向量、加载保存管线及需要的查询单元。若主动重新生成，只需执行 Notebook 2 的模型加载、编码、保存 pickle、保存管线及 manifest 单元。修改过的加载/保存单元已清除旧执行输出；没有声称完整文献流程已经重跑。

快速复核：

```bash
/opt/conda/envs/pathak/bin/python resources/checks/check_embedding_batch_invariance.py
/opt/conda/envs/pathak/bin/python resources/checks/check_biobert_parallel_batching.py
```

完整场景与测速重现：

```bash
/opt/conda/envs/pathak/bin/python resources/checks/validate_biobert_parallel.py --work .cache/parallel-recheck
/opt/conda/envs/pathak/bin/python resources/checks/validate_biobert_parallel.py --work .cache/parallel-recheck --fresh
/opt/conda/envs/pathak/bin/python resources/checks/benchmark_biobert_parallel.py --work .cache/parallel-recheck
```

构建工具 `resources/checks/patch_biobert_parallel.py` 使用现有 Java/protobuf 和独立目标目录，不覆盖已有模型，也不自动发布向量。构建出的候选仍须运行上述严格检查。构建源为本地 masked mean 副本，原模型与 checkpoint 校验值会再次核对。

本轮旧配置、向量、保存管线和 manifest 备份在 `.backups/biobert-layer-audit-20261021T150051Z/before/`。原模型继续保留。阶段1目录、计划 00、资源进展 01、环境依赖和 CPU 数据未改变。5,117 条仍是教学筛选库；本轮没有进行小鼠语料适配、完整文献评估或其他研究流程。
