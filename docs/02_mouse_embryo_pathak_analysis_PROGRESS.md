# 02：方法验证与研究分析进展

核查日期：2026-10-22（Asia/Tokyo）。本文件记录方法、验证和分析状态；研究设计见 [00 计划](00_mouse_embryo_pathak_PLAN.md)，资源与输入数量见 [01 进展](01_mouse_embryo_pathak_resources_PROGRESS.md)。项目内路径均相对于 `PATHAK/`。

## 1. 三本本地方法 Notebook 的核查

| Notebook | 代码单元数 | 有保存执行记录的单元 | 当前核查结论 |
| --- | ---: | ---: | --- |
| `tutorials/local_1_GO.ipynb` | 10 | 8 | 保存记录无错误输出；不能据此声称所有单元都完成了完整重跑 |
| `tutorials/local_2_GO_term_embedding.ipynb` | 13 | 5 | 存在独立向量重建与核查报告，详见下节 |
| `tutorials/local_3_PhenoMiner_plant.ipynb` | 17 | 6 | 部分过期输出已清除；完整小鼠文献流程尚未验证 |

这些文件分别提供 GO 语料、编码和文献候选生成的本地方法参考。阶段1的步骤01入口 `notebooks/stage1/01_GO_MELA_resources.ipynb` 的 5 个代码单元均有保存执行记录，资源结果另见 01。

## 2. BioBERT：正式 batch=16，严格一致性与全库验收通过

执行日期：2026-10-22（Asia/Tokyo）；服务器 stage3；环境 `pathak`，Spark NLP 4.0.2 / NLU 4.0.0 / Java TensorFlow 2.7.1。完整实施记录见 [BIOBERT 修复报告](../resources/checks/reports/BIOBERT_FIX_REPORT.md)。

### 2.1 当前正式状态

| 项目 | 已核查结果 |
| --- | --- |
| 模型 | `models/biobert_pubmed_sentence/` |
| 协议 | `masked_mean_parallel_sentence_v1` |
| 内部 batchSize | 默认 16；1、8、16 已核查 |
| 计算方式 | 图内 MapDefun 共享权重并行逐句；保留旧管线优化后的单句运算顺序 |
| pooling | 图内 float32 masked mean；保留 CLS/SEP，仅排除 PAD |
| 参数 | 每条 document 768 维；区分大小写；最大长度 128 |
| 线程 | 全新 JVM 的全局 intra-op=8、inter-op=8；函数内上限 2，OMP=2；metadata 在创建后台前保存配置 |
| 教学向量和管线 | `tutorials/predictions_full_BP.pickle`、`tutorials/pipe_full_BP/` 已重新生成 |
| 清单 | `tutorials/embedding_manifest.json` 核对语料、模型、向量和保存管线 |

原模型保留在 `.backups/biobert-models/sent_biobert_pubmed_base_cased/`，旧产物备份在 `.backups/biobert-layer-audit-20261021T150051Z/before/`。模型图、归档、权重和词表身份见 [模型记录](../resources/checks/embedding-parallel-model.json)。

### 2.2 定位与实际修复

原 Mean 包含 PAD 是主要偏差来源；仅修正 masked mean 后，部分编码器浮点路径仍导致两条定义超线。新探针确认敏感句的输入 embedding、LayerNorm 均值和方差相同，首差出现在 Rsqrt 的向量化/尾部路径；后续层会放大误差。同文本重复成批也有差异，因此只按有效长度分组不充分。

正式方案使用图内逐句并行保持单句形状，并从旧模型的运行时优化图提取函数，保留 Grappler 对 GELU 乘法顺序的重写。权重和 tokenizer 保留，正常输出仍只返回 768 维向量，hidden states 不送回 Python 后处理。提高归一化精度、关闭融合及增大句内线程上限等候选未通过全部严格比较；当前全局8、句内2的性能与验收见 2.6。

### 2.3 CPU2 基线的一致性与全库验收

标准保持 `max_abs <= 1e-5` 且 `cosine >= 0.999999`。

- 40 条文本、45 项比较覆盖 batch=1/8/16、单句与短长混合、敏感句重复、重排、list/Series、分块、长度边界及保存重载，最大绝对差 0。
- 保存后独立新进程 3 项比较最大绝对差 0。
- 最终 2/8 配置下，batch=8 与 16 分别完成全部 5,117 × 768 的重编码，与旧单句库逐元素数值相等，最大绝对差 0，拒绝行数 0，行对应、有限值和非零范数通过。
- 正式默认入口与保存管线的新进程检查 13 项通过；确认保存的 batch 和实际捕获 ConfigProto，最大绝对差 0。
- 旧 masked mean 候选作为负对照，仍被相同阈值拒绝原来的两条定义。

证据：[扩展检查](../resources/checks/embedding-parallel-validation.json)、[全库检查](../resources/checks/embedding-parallel-full-library.json)、[正式产物新进程](../resources/checks/embedding-parallel-postcheck.json)、[总体记录](../resources/checks/embedding-parallel-fix.json)。

### 2.4 CPU2 基线的历史速度记录

相同硬件、2/8 线程设置下，以覆盖长度分布的 256 条 GO 定义预热，轮换配置测三轮。计时包含 predict、行对齐和向量提取，排除加载和预热。

| 配置 | 本地加载秒 | 首次 32 条预测秒 | 256 条编码中位秒 | 条目/秒 | 相对原模型 batch=1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原模型 batch=1 | 6.753 | 4.821 | 10.636 | 24.07 | 1.000× |
| 修正版 batch=1 | 4.575 | 2.145 | 10.687 | 23.96 | 0.995× |
| 修正版 batch=8 | 4.330 | 2.198 | 9.370 | 27.32 | 1.135× |
| 修正版 batch=16 | 4.433 | 1.645 | 9.340 | 27.41 | 1.139× |

此前 CPU2 版本采用 batch=16，公平测速吞吐约为原模型 batch=1 的 1.139 倍，增幅有限；8 和 16 的速度接近。完整库单次编码 batch=8 为 193.645 秒、16 为 195.944 秒，此记录单列，未用于计算四配置的加速比。详细轮次及样本行号见 [测速记录](../resources/checks/embedding-parallel-benchmark.json)。

### 2.5 使用与研究边界

本轮已完成教学库和保存管线重建，无需再重跑 GO 准备。已有内核应重启，重新运行 Notebook 3 的初始化、读取 GO/向量、加载保存管线和需要的查询单元。Notebook 2 重新生成时会同时更新 manifest；Notebook 3 加载时核对产物配套关系。

批量编码技术核查已完成。5,117 条仍是教学候选定义，未进行小鼠完整 BP 库适配、检索阈值确定或 gene–GO 关系评估。阶段1资源、阶段2常规 GO 基准、文献、表达和谱系流程仍按 00 计划分别推进。

### 2.6 性能复验：采用全局 CPU8、句内上限 2

当前正式模型为 `models/biobert_pubmed_sentence/`。默认 batch16，配置 `[16, 8, 40, 8]`，句内上限 2、OMP=2。模型图、权重、tokenizer 和 pooling 协议保留。旧 CPU2 模型已移至 `.backups/biobert-models/sent_biobert_pubmed_base_cased_parallel_masked_mean/`。

三轮交替测试包含 516 条 GO 定义和 269 个真实文献句子。端到端 785 条合计中位由 36.350 秒降为 12.657 秒，21.60 → 62.02 条/秒，约 2.872 倍。平均活跃 CPU 核约 2.46 → 9.11；这是增加计算并发的收益，需要相应 CPU 资源。此前 batch8/16 的微小差别小于轮次波动，本次没有据此声称 16 更快。

原瓶颈为进程级计算池仅有 2 线程。TensorFlow 第一个 Session 决定全局 intra-op 池，因此新配置要求重启内核。缓存后的 NLU 整理约 0.3 秒，函数级追踪主要耗时为矩阵乘法。

batch1、8、16 各自全部 5,117 × 768 个值均与原单句参考相等，最大差值 0；独立进程 raw/保存后加载检查 17 项、发布后 13 项及旧检查命令均通过。269 × 5,117 的查询分数、Top-3/10/20 排名、近似并列和三个现用阈值结果均未改变。阈值仍为 max_abs <= 1e-5、cosine >= 0.999999。

按长度合并的 dense 批量有两条 GO 超线，未采用；扩大句内线程上限也出现超线。隔离 GPU 后端能执行基础矩阵运算，但该 BioBERT 图原生前向崩溃，未获得有效编码速度、未部署。生产仍使用原 `pathak` 依赖与 Java TensorFlow 2.7.1。

完整向量库、保存管线和 manifest 已配套重建。Notebook 重新启动内核并重跑初始化、读取向量和加载保存管线即可。方法及研究阶段边界保持不变，本次不计作小鼠文献评估完成。

详见 [性能报告](../resources/checks/reports/BIOBERT_PERFORMANCE_REPORT.md)、[验收](../resources/checks/embedding-performance-global8-validation.json)、[测速](../resources/checks/embedding-performance-global8-benchmark.json)。

## 3. 静态检查中仍需处理的实现项

以下实现项仍待处理。本轮仅更新编码加载、保存和清单核对单元，未实现这些研究逻辑。对 `tutorials/local_3_PhenoMiner_plant.ipynb` 的静态代码检查可见：

| 实现项 | 检查到的情况 | 小鼠正式检索的处理要求 |
| --- | --- | --- |
| 查询相似度 | 查询向量被重复为多行后形成多余 N×N 矩阵 | 直接计算单查询与 GO 向量的相似度 |
| 多条 GO 命中 | 循环中反复读取第一条命中 | 每次使用当前命中记录 |
| 段落汇总 | 绘图循环上界会遗漏最后一个段落 | 核查全部段落的汇总边界 |
| 基因及候选范围 | 使用植物示例词典和 TAIR 筛选 | 替换为小鼠标识、候选范围和物种判断 |
| 关系证据 | 基因与功能共同出现只能形成候选 | 增加关系判断、实验和发育背景核查 |

这些项应在小鼠方法实现和测试时逐一关闭，不因编码一致性通过而视为全部解决。

## 4. 本研究八阶段状态

| 阶段 | 核查状态 | 下一项具体工作 |
| --- | --- | --- |
| 1 小鼠参考资源 | 步骤01本地资源核查完成；ID 和 GO 注释待准备 | 按步骤02准备小鼠标识与 GAF 输入，完成映射覆盖 |
| 2 程序 GO 基准 | 本项目待开展 | 明确程序候选背景、ORA 和作者解释对照 |
| 3 文献小试 | 正式 batch=16 通过编码、全库及新进程核查；小鼠检索与人工评估待开展 | 使用已验证编码协议，处理实现项，选心脏优先程序并开展证据核查 |
| 4 心脏输入冻结 | 本项目尚未建立固定心脏目标清单 | 核查组织来源、心脏类型、阶段、重复和树叶覆盖 |
| 5 功能时序 | 本项目待开展 | 使用冻结的心脏细胞群进行来源内部分析 |
| 6 MELA 谱系关联 | 本项目待开展 | 末端表达连接目标树叶，明确零模型与重复 |
| 7 Typewriter 核查 | 本项目待开展 | 冻结程序/候选后核查 E13.5 末端支持 |
| 8 候选整合 | 本项目待开展 | 汇总文献、表达和谱系证据，保留反例与缺口 |

### 4.1 下一项执行：阶段1 · 步骤02

当前 `data/reference/mouse/` 尚无准备好的资源文件，步骤02 Notebook 也尚未创建。下一项为计划约定的 `notebooks/stage1/02_mouse_ID_GO_annotations.ipynb`，按以下顺序开展：

1. 固定小鼠标识、GO 注释与现有 GO 本体的来源、版本、获取日期和校验值，核对版本兼容性。
2. 准备 `data/reference/mouse/gene_identifiers.tsv`，连接原始名称、官方符号、MGI 和 Ensembl，保留历史名称、一对多关系、歧义与未映射项。
3. 准备 `data/reference/mouse/mouse_go_annotations.gaf.gz`；保留物种、关系、NOT、证据、参考和可用的上下文扩展，生成可追溯的 BP 注释视图。
4. 对步骤01的 1,322 个不同原始基因名称核查稳定 ID 与 GO 注释覆盖，将歧义和未映射项单列，保留映射前后数量和来源依据。
5. 将标识、注释、映射覆盖及版本核查输出保存到 `results/01_resources/`，资源事实更新至 01 进展文档，验证结论更新至本文件。

阶段1的资源与映射核查完成后，进入研究阶段2的 70 程序常规 GO 基准。开展 ORA 前先追溯程序产生时的候选基因背景；背景未明确时记录限制和敏感性检查，不能直接把 1,322 个程序成员当作通用富集背景。心脏相关程序的文献优先级在基准结果及覆盖核查后确定，后续编码使用本节已验证协议，检索与关系证据仍单独评估。

步骤02的名称和职责已确定；本次仅记录执行顺序，没有创建或执行该 Notebook，也未下载小鼠资源。

### 4.2 后续 CPU 数据接入约束

CPU 既有表达处理、v4 注释和参考投影由其原项目管理；不能把它们自动算成本研究的阶段 4–7 已完成。接入时核对：

- `CPU:seurat_project/data/v4/reference/reference_types_v4.csv`。
- `CPU:seurat_project/results/v4/stage3/decision_log.csv`。
- `CPU:seurat_project/results/v4/stage3/reference_changes.csv`。

这里 `CPU:` 的数据根目录为 `/hy-tmp/`。心脏优先范围采用祖细胞和心肌等来源明确群体，具体纳入规则在阶段 4 固定。Typewriter 的核参考投影和作者祖先推断要记录参考重叠；末端测量、标签转移和祖先推断分开评估。

## 5. 后续更新格式

每次执行记录以下信息，详细运行清单保存到相应 `results/` 阶段目录：

```text
执行日期与阶段：
服务器、环境、代码/Notebook 版本：
输入路径、版本、校验值：
参数与统计单位：
输出路径：
核查证据与适用范围：
问题、未完成项及下一步：
```

计划标准保留在 00；资源数量和覆盖更新到 01；方法结果及阶段完成情况更新到本文件。

### 2.7 目录与模型归档整理

两份 BioBERT 报告已移入 `resources/checks/reports/`；上游说明移入 `tutorials/UPSTREAM.md`。`models/` 仅保留正式模型 `biobert_pubmed_sentence/`，原模型、CPU2 基线和 dense masked mean 候选移到 `.backups/biobert-models/`。加载入口、Notebook、配套 manifest 与本地资源校验表已同步调整，向量和保存管线不需要重新生成。本次路径整理的加载与零差值复核见 [目录检查](../resources/checks/embedding-layout-reorganization.json)。
