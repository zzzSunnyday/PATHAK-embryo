# BioBERT 编码性能：剖析、候选验证与 CPU8 交付

> 目录整理：下文代码和数据路径以 `PATHAK/` 为起点；Markdown 证据链接按当前文件位置解析。正式模型为 `models/biobert_pubmed_sentence/`，历史模型归档在 `.backups/biobert-models/`。历史 JSON 保留测量发生时的原始路径和校验值，当前映射见 `config/biobert_model.json`。

执行日期：2026-10-22（Asia/Tokyo）；服务器 stage3；项目内路径均相对于 `PATHAK/`。

正式配置已更新为 **batch=16、全局 intra-op=8、inter-op=8、句内最大 intra-op=2、OMP=2**。固定 785 条语料的三轮端到端中位耗时由 **36.350 秒**降到 **12.657 秒**，吞吐由 **21.60** 提升到 **62.02 条/秒**，相对已验收 CPU2 稳定版为 **2.872 倍**。

batch=1、8、16 各自全部 5,117 × 768 个值均与原模型单句参考相等，最大绝对差为 0；严格标准保持 `max_abs <= 1e-5`、`cosine >= 0.999999`。图、权重、tokenizer、大小写、截断和 pooling 协议均保留。

## 1. 实际瓶颈

预先分词、预建输入 tensor 后，纯 TensorFlow 编码仍占主要耗时；缓存已经算好的向量后，Spark 结果收集及 NLU 整理只占较小部分。实际进程和原生线程采样显示 CPU2 版约使用 2.4–2.5 个核，GPU 使用率为 0。

| 516 条 GO 定义的剖析项 | 秒 | 含义 |
| --- | ---: | --- |
| 输入转 Spark DataFrame | 0.0195 | 数据转换 |
| 构建 transform 与 cache 计划 | 0.1031 | 惰性计划，尚未计算模型 |
| 首次物化全部缓存列 | 20.066 | 包含模型计算 |
| 缓存后仅收集 document | 0.0528 | 三轮中位；范围 0.0526–0.0678 |
| 缓存后收集全部列 | 0.1047 | 三轮中位；范围 0.0892–0.1054 |
| 缓存后的完整 NLU 整理 | 0.3201 | 三轮中位；范围 0.2813–0.3914 |

缓存结果再次核对为零差值。纯模型和 Spark 调用有不同的调用分组及调度，不能把两条总耗时简单相减当成精确外层开销。Spark task 的 CPU 指标不覆盖全部 TensorFlow 原生线程，本报告以 `/proc` 进程和线程 CPU 计数补充核查。

TensorFlow 2.7.1 默认每进程只有一个 intra-op 池，由第一个 Session 决定，后续 Session 共用。`use_per_session_threads` 针对 inter-op 池，不能重新建立已创建的 intra-op 池。因此在已启动 CPU2 的同一 JVM 中把新会话配置写为 8，实际计算池仍为 2。新方案从全新 JVM 启动全局池 8，并保留每句上限 2。

来源：[TensorFlow 2.7.1 ConfigProto](https://github.com/tensorflow/tensorflow/blob/v2.7.1/tensorflow/core/protobuf/config.proto#L373)、[MapDefun 内核](https://github.com/tensorflow/tensorflow/blob/v2.7.1/tensorflow/core/kernels/data/map_defun_op.cc)。MapDefun 支持并发；本轮结果不能解释为它只能串行。

MapDefun 的外层追踪不暴露全部函数内部算子。为定位算子，使用同一 FunctionDef 的独立单句诊断调用，输出与基线逐值相等。排除容器调用后，矩阵乘法在记录的内核累计耗时中占比：

| 有效 token 长度 | 矩阵乘法占内核累计时间 | 前馈层矩阵乘法占比 |
| ---: | ---: | ---: |
| 8 | 94.98% | 53.52% |
| 32 | 88.52% | 49.25% |
| 128 | 78.92% | 44.89% |

FULL_TRACE 和诊断调用本身有开销，上述时间用于算子归因，不作为吞吐测量；本轮未采集硬件缓存/带宽计数，未进一步断言内存带宽或算力占比。

## 2. 固定输入与公平测速条件

- GO：按定义字符长度分层抽取 512 条，加入 4 条已知敏感定义，共 516 条。文献：两篇本地 XML 摘要和正文的全部 269 个去重句子，无基因筛选。
- 文献输入来自 `tutorials/Example_data/PMC1459476.xml`、`PMC1802096.xml`；用于编码和查询稳定性测试，不代表已完成小鼠文献评估。
- 输入冻结：`.backups/biobert-performance-20261021T161556Z/corpus.jsonl`；SHA256 `b128017f9b70eb679b40aabf30cd6b3a3620886453f8d6105ea5d80fe8922e75`。
- 大小写敏感；同一 Java tokenizer；最大长度 128；保留 CLS/SEP，排除 PAD。纯模型使用预先分词和驻留输入 tensor，两路并发 Session.run；端到端包括输入、predict、行校验和 768 维结果提取。
- CPU2 与 CPU8 使用独立、预热后的 JVM，逐轮交替；Spark 均为 local[2]，batch 均为 16。加载、首次预热不计入编码。没有改变数值阈值。
- Xeon Platinum 8480+；224 逻辑 CPU 可见，容器总 CPU 配额为 48 核。共享服务器上保留了其他作业；原始记录包括逐轮限流、就绪队列等待、原生线程 CPU、RSS 和 GPU 采样。个别窗口出现 2 个容器限流周期，未删除这些轮次。

| 输入 | token 数最小/中位/最大 | 平均 | 达到 128 的条目数 |
| --- | --- | ---: | ---: |
| go | 8/29.5/128 | 33.78 | 1 |
| literature | 10/39/124 | 42.58 | 0 |

## 3. 正式采用方案的实测速度

| 语料与粒度 | CPU2 秒 | CPU8 秒 | CPU2 条/秒 | CPU8 条/秒 | 中位耗时比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 516 条 GO · 纯模型 | 26.022 | 7.609 | 19.83 | 67.81 | 3.420× |
| 516 条 GO · 端到端 | 23.111 | 8.285 | 22.33 | 62.28 | 2.789× |
| 269 个文献句子 · 纯模型 | 13.244 | 4.030 | 20.31 | 66.75 | 3.287× |
| 269 个文献句子 · 端到端 | 13.239 | 4.316 | 20.32 | 62.32 | 3.067× |
| 785 条合计 · 纯模型 | 39.266 | 11.472 | 19.99 | 68.43 | 3.423× |
| 785 条合计 · 端到端 | 36.350 | 12.657 | 21.60 | 62.02 | 2.872× |

合计按同一轮的 GO 与文献时间相加后取中位。各轮原始值如下，未只依据一个小样本中位数作决定。

| 输入 | 粒度 | CPU2 三轮秒 | CPU8 三轮秒 | 逐轮加速比 |
| --- | --- | --- | --- | --- |
| go | pure_tf | 22.314, 27.653, 26.022 | 7.674, 7.609, 6.969 | 2.908×, 3.634×, 3.734× |
| go | end_to_end | 21.747, 23.278, 23.111 | 8.285, 8.341, 7.221 | 2.625×, 2.791×, 3.201× |
| literature | pure_tf | 12.984, 13.273, 13.244 | 4.184, 3.862, 4.030 | 3.103×, 3.436×, 3.287× |
| literature | end_to_end | 13.282, 13.084, 13.239 | 4.517, 4.316, 4.179 | 2.940×, 3.031×, 3.168× |

资源成本：端到端测量窗口内，JVM 平均活跃核数约从 **2.46** 增至 **9.11**。生产仍是一个 JVM 内共享模型权重；没有部署多个模型工作进程。编码测量期间 GPU0 最大使用率为 **0%**。CPU 并发增加是主要资源代价，速度提升不是相同 CPU 用量下的收益。

模型本地加载单列：CPU2 为 7.923 秒，CPU8 worker 的模型加载为 7.453 秒；worker 完整启动为 13.461 秒。两者初始化路径不同，未把加载时间加入吞吐比，也未据此判断加载快慢。

## 4. 其他候选与拒绝依据

| 候选（阶段 A，独立于上述轮次） | GO 端到端条/秒 | 文献端到端条/秒 | 相对同阶段基线 | 结论 |
| --- | ---: | ---: | --- | --- |
| CPU2 稳定基线 | 26.06 | 22.96 | GO 1.000×；文献 1.000× | 参考 |
| 同 JVM 新建 8 线程会话 | 25.90 | 23.38 | GO 0.994×；文献 1.018× | 全库零差值，实际全局池仍为 2；收益在波动内，未采用 |
| 按 WordPiece 长度排序的合并 masked mean 批量 | 45.76 | 37.23 | GO 1.756×；文献 1.621× | 全库 2 条超线，未采用 |

合并批量的最坏绝对差为 `1.52587890625e-05`，超线 GO 原始行号为 [388, 4547]。最小 cosine 为 `0.999999999995577`。269 个真实查询的最大相似度变化为 `2.3103992286e-07`；Top-3/10/20 的集合与顺序均未改变，0.90、0.92、0.935 阈值均无翻转。不同分块下敏感行会变化；这属于剩余浮点兼容性问题，不能按现行标准宣布通过。

另一次资源上限筛选将全局池、OMP 和句内上限均改为 8：516 条用时 6.723 秒，但有 1 条超过标准，最坏 `1.52587890625e-05`。这是单轮筛选，未作为正式公平加速比。本次最终保留 OMP=2 和句内上限 2。

GPU 候选使用完全隔离的 Python 3.12 / TensorFlow 2.16.2 环境，原 checkpoint 和同一 Java tokenizer，FP32、TF32 关闭。L40S 可见，独立 MatMul 的 GPU trace 与数值检查通过；但这个 BioBERT 图在 Session.run 原生层以 SIGSEGV（-11）退出。默认优化、关闭 oneDNN、进一步关闭 meta optimizer 三种受控检查均未获得完整向量。具体失败算子尚未定位，**没有有效的 BioBERT GPU 吞吐或检索精度可报告，未部署**。

GPU 诊断窗口最大采样为 19%、959 MiB；这些是失败尝试及独立矩阵试验的观测，不代表稳定 BioBERT 开销。环境锁定在 `.backups/biobert-performance-20261021T161556Z/gpu-requirements.lock.txt`，原生错误、变体和硬件对照保留在同目录，Conda `pathak` 未安装这些新依赖。

## 5. 严格验收与真实查询稳定性

| 检查 | 结果 |
| --- | --- |
| batch16 全部 5,117 × 768 | max_abs=0.0；行对应通过；单次验收耗时 73.026 秒 |
| batch8 全部 5,117 × 768 | max_abs=0.0；行对应通过；单次验收耗时 81.202 秒 |
| batch1 全部 5,117 × 768 | max_abs=0.0；行对应通过；单次验收耗时 199.728 秒 |
| 独立新进程、raw 与保存后重载 | 17 项；最大差 0.0 |
| 发布后默认入口及保存管线 | 13 项；最大差 0.0 |
| 原检查命令 | list/Series、输入重排、单句/混合、重复、长上下文、新进程/保存结果均为零差值 |
| 新 manifest 检查 | metadata SHA 已加入；篡改 metadata 身份被拒绝；CPU2 基线配置被准确保留 |

用例覆盖真实单句、短长混合、重复 17 次、重排、非默认索引 pandas Series、不同调用分块、124–127 与 600 次 cell 文本。检查了真正的 Java batch 参数与后台捕获 ConfigProto，保持 768 维 document 粒度。全库单次耗时用于验收，batch16 还包含保存管线和 pickle，未拿来判断 batch8/16 的速度优劣。

41 条原模型 batch1 / CPU2 参考已冻结到 `resources/checks/embedding-original-single-cpu2-reference.npz`，配套 JSON 记录源模型、配置、文本和文件校验信息。发布后的 CPU8 进程不能重新代表旧 CPU2 全局池，因此长度边界检查读取该独立参考；验收阈值没有改变。

完整查询比较：269 个文献句子 × 5117 个 GO，共 1,376,473 个相似度。新方案最大分数变化为 0.0；Top-1、Top-3、Top-10、Top-20 的集合、顺序和排名均相同，三个现用阈值的判定均相同。

近似并列样例（从基线 Top-20 邻接排名中选择最小间隔，完整记录见 JSON）：

| 查询 ID | GO 对 | 基线分差 | 新方案分差 | 排名翻转 |
| --- | --- | ---: | ---: | --- |
| PMC1802096:body:6:6 | GO:0000304, GO:0000377 | 2.46106287682e-07 | 2.46106287682e-07 | False |
| PMC1459476:body:3:2 | GO:0000303, GO:0000304 | 3.31113186425e-07 | 3.31113186425e-07 | False |
| PMC1802096:body:18:2 | GO:0000447, GO:0000479 | 5.78454262157e-07 | 5.78454262157e-07 | False |
| PMC1802096:body:6:2 | GO:0000292, GO:0000731 | 6.16993165004e-07 | 6.16993165004e-07 | False |
| PMC1802096:body:18:4 | GO:0000466, GO:0000377 | 8.65652524773e-07 | 8.65652524773e-07 | False |

| 阈值 | 最近实测查询/GO | 距离阈值 | 判定翻转数（全部配对） |
| --- | --- | ---: | ---: |
| 0.9 | PMC1459476:body:26:2 / GO:0000032 | 5.59584368809e-06 | 0 |
| 0.92 | PMC1459476:body:27:0 / GO:0000374 | 4.33549527135e-06 | 0 |
| 0.935 | PMC1459476:body:25:5 / GO:0000390 | 7.37137755682e-05 | 0 |

这是编码和检索数值稳定性检查，不是 gene–GO 关系准确率或小鼠语料评估。原 PAD 平均错误会造成大幅漂移；本次拒绝的 dense/大句内线程候选为微小浮点差异，而最终配置在本轮全部比较中为零差值。

## 6. 模型、产物与保留基线

| 对象 | 路径 | 归档 SHA256 | metadata SHA256 |
| --- | --- | --- | --- |
| 原始导出 | `.backups/biobert-models/sent_biobert_pubmed_base_cased/` | `165a38e20e3b53569947fb3af8803deb8132e75e81ab75d1337b4c7b4d40cf90` | `e9402f16a1dcd1a8eebb9aceedee0769a63b252ab66e87257f095474e990da6e` |
| CPU2 基线 | `.backups/biobert-models/sent_biobert_pubmed_base_cased_parallel_masked_mean/` | `3583a8f750a8891440c1338a46399a9f18a2fc60f5db113506183e0a83c4161b` | `c8313fc423e8cbb2519e95ea9747e9b1b5d2610533a53be70eff1f117a9f28a6` |
| 当前 CPU8 | `models/biobert_pubmed_sentence/` | `3583a8f750a8891440c1338a46399a9f18a2fc60f5db113506183e0a83c4161b` | `a945ec990d0edf1773a90e40ace14cf298db038c318d8a87df63dd063c4f6763` |

CPU2 与 CPU8 的 GraphDef 均为 `8e5f288ae33ece7f72783989fa83f60ad562e4b1df9f4ada0256de0411f58ba7`，归档也相同，差别在 metadata 的线程资源配置。所有候选保留同一权重 data SHA `0c861e31aee67b085d618f582a5a7fcfe4129753f9578281c45ce86a8ede6c22` 与 index SHA `0121af1dfa09816805a6fe8185d25565f8b26d979b0b5c658a73995e2b7d0289`。

- 当前入口：`pathak_runtime.py`；当前 raw 模型目录见上表；默认 batch16，允许已验证的 1/8/16。
- 完整配套：`tutorials/predictions_full_BP.pickle`、`tutorials/pipe_full_BP/`、`tutorials/embedding_manifest.json`。已重新编码并核对行对应；正常预测仍只返回每句 768 维向量，pooling 在 TensorFlow 图中。
- manifest 同时记录 raw 模型 metadata SHA、归档、图、语料、向量和保存管线；加载代码保留显式 CPU2 基线的真实配置，避免用后改的 Python 参数掩盖已捕获后台配置。
- 初始稳定版本完整快照：`.backups/biobert-performance-20261021T161556Z/baseline/`；切换前在线产物另保留在 `before-publication/`。原始模型、CPU2 模型和旧结果保留。
- Spark NLP 4.0.2、NLU 4.0.0、PySpark 3.2.3、Java TensorFlow 2.7.1 与 Conda `pathak` 保持原版本。生产 classpath 仍只有原 Spark NLP assembly JAR。
- `docs/00_mouse_embryo_pathak_PLAN.md`、`docs/01_mouse_embryo_pathak_resources_PROGRESS.md`、`config/project.json`、`environment.yml` 校验值未变；CPU 服务器上的数据和研究流程未操作。

## 7. Notebook 最少重跑步骤

1. **重启 pathak Notebook 内核**，确保 JVM 的第一个 TensorFlow Session 使用 8 线程计算池。只改 Python 参数或重建 SparkSession 不能替换已存在的进程级池。
2. Notebook 2 运行初始化和加载模型单元即可看到新模型目录与 `[16, 8, 40, 8]`；本次完整向量和保存管线已生成。Notebook 3 重跑初始化、读取 GO/向量、manifest 检查和加载保存管线的单元。
3. 重新运行需要的查询。向量库无需由用户再计算一遍。检索矩阵、关系判断及小鼠适配按原计划单独推进。

可选的新进程复核：

```bash
python resources/checks/check_biobert_parallel_batching.py --report resources/checks/embedding-performance-global8-recheck.json
```

完整公平复测入口为 `resources/checks/benchmark_biobert_global8.py`，自动复制冻结输入到新的 `.backups/` 子目录，使用独立 JVM 比较。原执行目录保持只读审计用途，避免覆盖本次证据。

证据：[正式验收](../embedding-performance-global8-validation.json)、[逐轮测速](../embedding-performance-global8-benchmark.json)、[剖析](../embedding-performance-profile.json)、[拒绝候选](../embedding-performance-candidates.json)、[总体机器记录](../embedding-performance-global8.json)。
