# 小鼠胚胎心脏相关 gene–GO 研究计划（E7.5–E13.5）

## 1. 文档范围与路径约定

本文件只规定研究目标、数据结构、执行顺序、产出和验证条件。完成情况、运行参数、资源数量、问题及下一步执行记录分别维护在：

- [01：资源与数据准备进展](01_mouse_embryo_pathak_resources_PROGRESS.md)。
- [02：方法验证与研究分析进展](02_mouse_embryo_pathak_analysis_PROGRESS.md)。

项目内路径均以 **`PATHAK/` 为起点**，例如 `data/raw/mela/SOURCE.json`。Markdown 链接按文档所在目录解析，表格和正文中的路径按项目根目录解析。

CPU 数据位于项目目录之外，使用 `CPU:` 前缀表示；`CPU:` 的数据根目录为 `/hy-tmp/`。例如 `CPU:GSE342607/author_metadata/annotated_programs.csv` 表示 CPU 数据根目录下的对应文件，不属于 stage3 的 `PATHAK/` 目录。

## 2. 研究目标与优先范围

研究小鼠胚胎 E7.5–E13.5 期间，心脏相关细胞谱系的 GO Biological Process 功能程序如何变化，以及哪些基因和程序同时获得文献、表达与谱系关联支持。

优先考察**心脏发生、心脏/心咽祖细胞及心肌细胞分化**；根据阶段、胚胎重复和标签覆盖，决定是否纳入心内膜、心外膜及心脏相关间充质等群体。心脏来源需要组织、标签或其他来源证据支持，不能将全身内皮或间充质都归入心脏研究范围。

主要问题：

1. 心脏相关功能程序在不同发育阶段如何增强、减弱或发生替换？
2. 同一 GO 过程是否由不同阶段的不同基因组合支持？
3. 控制细胞类型、阶段和技术因素后，相关程序是否还具有克隆或谱系关联？
4. 哪些基因具有可追溯的功能依据，适合进一步验证？

第一轮以 MELA 的 70 个作者程序建立比较基准；文献小试优先选择其中与心脏相关的程序，并加入其他功能主题作为比较。若直接相关程序较少，记录筛选依据，不强行给程序添加心脏标签。已有程序解释与复现结果单独标记，新增发现需要另外评估。

## 3. 方法参考、服务器分工与目录

### 3.1 三本本地方法 Notebook

| 本地 Notebook | 参考内容 | 小鼠研究中需要落实的内容 |
| --- | --- | --- |
| [tutorials/local_1_GO.ipynb](../tutorials/local_1_GO.ipynb) | GO 图、术语、定义与语料整理 | 严格 BP namespace 筛选；GO—文本映射；版本和替代 ID |
| [tutorials/local_2_GO_term_embedding.ipynb](../tutorials/local_2_GO_term_embedding.ipynb) | BioBERT 编码及向量保存 | 小鼠候选语料、统一编码协议、显式 ID 和逐行对齐 |
| [tutorials/local_3_PhenoMiner_plant.ipynb](../tutorials/local_3_PhenoMiner_plant.ipynb) | 论文读取、基因匹配、GO 检索与结果汇总 | 小鼠基因词典、物种消歧、关系判断、发育背景及原文出处 |

阶段1的步骤01入口为 [notebooks/stage1/01_GO_MELA_resources.ipynb](../notebooks/stage1/01_GO_MELA_resources.ipynb)。小鼠 Notebook 按研究阶段组织到 `notebooks/stage1/`、`notebooks/stage2/` 等目录，文件以阶段内步骤 `01`、`02` 编号。阶段1的步骤02用于小鼠基因标识与 GO 注释；研究阶段2则是程序 GO 基准，两者含义不同。方法参考中的 TAIR 筛选、基因词典和候选向量需要按小鼠研究范围重新准备。

### 3.2 工作分配

| 位置 | 工作 | 路径约定 |
| --- | --- | --- |
| stage3 | GO、小鼠标识、文本编码、文献检索与关系证据 | `config/`、`notebooks/`、`data/`、`results/` |
| stage3 | 软件、模型和运行依赖 | 环境 `pathak`；`environment.yml`、`pathak_runtime.py`、`models/`、`resources/` |
| CPU | 表达、元数据、心脏目标群、时间趋势和谱系分析 | 读取 `CPU:seurat_project/`、`CPU:GSE342607/`、`CPU:dna_typewriter_mouse_2026/` |
| CPU | 本研究输入快照与结果 | 计划使用 `CPU:embryo_go_project/`，按阶段保存 |

### 3.3 项目内目录

| 路径 | 用途 |
| --- | --- |
| `config/project.json` | 物种、发育窗口、资源路径和来源配置 |
| `docs/` | 研究计划与分阶段进展 |
| `notebooks/` | 按 `stage1/`、`stage2/` 等研究阶段组织 Notebook；各阶段内部按 `01`、`02` 编号 |
| `notebooks/stage1/` | 阶段1的小鼠参考资源与程序成员步骤 |
| `tutorials/` | 三本本地方法 Notebook 及相关教学文件 |
| `data/go/go.obo` | GO 本体 |
| `data/raw/mela/annotated_programs.csv` | 作者程序表 |
| `data/raw/mela/SOURCE.json` | 程序表来源与校验记录 |
| `data/reference/mouse/` | 小鼠标识、GO 注释和版本信息 |
| `results/01_resources/` | BP、程序成员、映射覆盖和资源核查 |
| `results/` | 后续富集、文献候选、评估和汇总结果，按阶段分目录 |
| `models/`、`resources/` | BioBERT、JAR、NLTK 和运行核查输入 |

路径从项目根目录解析，避免依赖 Jupyter 启动目录。资源来源、参数和文件校验值进入运行清单；跨服务器文件保留服务器标识及来源路径。

## 4. 数据是什么：行、列与分析单位

### 4.1 四类表达来源

“完整细胞公共表达数据”指按研究窗口与入口规则选取的 scRNA-seq 数据，包含整胚和组织/器官来源；MELA 单独列出以避免重复计数。具体数据集和数量记录在 01 进展文档，不能把多个对象的数量之和理解成一个已经合并的矩阵。

| 数据类别 | 观察单位 | 表达文件及行、列含义 | 配套信息 |
| --- | --- | --- | --- |
| 完整细胞公共表达数据（单列 MELA） | 一个完整细胞 | Seurat 表达矩阵：**行是基因/feature，列是细胞 ID** | 数据集、样本、阶段、组织/细胞标签、QC、表达尺度 |
| 公共细胞核表达数据（单列 Typewriter） | 一个细胞核 | Seurat 表达矩阵：**行是基因/feature，列是核 ID** | 同上，`unit` 区分 nucleus |
| MELA 表达与谱系数据 | 一个末端测得的细胞 | H5TD 的 `X`：**行是细胞，列是基因/feature**；转成 Seurat 时轴方向相反 | `obs` 中的 embryo、stage、capture、type、clone、tree、cell_type、cell_subtype 等 |
| Typewriter 末端表达与谱系数据 | 一个 E13.5 细胞核 | H5AD 的 `X`：**行是核，列是基因/feature** | H5AD 的细胞 ID 连接外部作者标签及谱系树叶 |

表达矩阵的一个数值代表该观察单位中相应基因的计数或经过处理的表达值。UMI、读段计数和归一化表达按来源分别标记，不能混作同一尺度。feature 编号需要连接真实基因 ID；不按列位置或未验证的同名字符串强行赋名。

### 4.2 表达矩阵、元数据表与汇总表的区别

| 表/矩阵 | 一行代表什么 | 主要列是什么 |
| --- | --- | --- |
| Seurat 表达矩阵 | 一个基因/feature | 每列一个细胞或核；单元格是表达值 |
| H5AD/H5TD 的 `X` | 一个细胞或核 | 每列一个基因/feature；单元格是表达值 |
| 逐细胞/核 metadata | 一个细胞或核 | 来源、样本、阶段、标签、QC 和必要的谱系标识 |
| 数据集/分支汇总表 | 一个数据集与分支的组合 | 数量及 QC 汇总；没有逐基因表达列 |
| MELA 作者程序表 | 一个基因程序 | `program`、`genes`、`size`、`active_in`、`name`、`description` |
| 程序成员长表 | 一个程序中的一个原始基因成员 | `program`、`gene_raw`、`member_position` |
| BP 术语表 | 一个 GO BP 术语 | `GO`、`Name`、`Namespace`、`Definition_OBO`、`Synonyms_OBO`、`is_namespace_root` |
| 小鼠 GO 注释 | 一条来源明确的功能注释 | 基因 ID、GO、关系、参考、证据、物种和可用的扩展信息 |
| 文献证据表 | 一条 gene–GO 候选及其证据 | 基因、GO、原文、文献、发育背景、检索与核查信息 |

CPU 的数据集/分支汇总例子是 `CPU:seurat_project/results/v4/stage1/qc_cells_summary.csv`：`dataset` 和 `branch` 标识来源，`cells`、`in_analysis` 记录数量，其他列记录 QC 标记数量或评分用途。多个 QC 标记可以重叠，不能简单相加当作最终排除数。

### 4.3 元数据需要保留哪些列

CPU 的部分导出 metadata 使用下列字段；它们是来源字段示例，目标群冻结时应记录字段映射与版本，不把历史导出直接当作最终注释。

| 字段 | 含义 |
| --- | --- |
| `cell_id`、`global_cell_id` | 对象内与跨对象唯一标识；用于连接表达、标签和树叶 |
| `source_id`、`dataset` | 来源对象/样本与研究数据集 |
| `unit` | 完整细胞或细胞核 |
| `stage_source`、`day_source`、`source_stage_original` | 阶段、数值日龄及原始写法；另保留时间依据与测量/推断性质 |
| `nCount_RNA`、`nFeature_RNA` | 当前表达轴上的总计数与检出基因数，解释时核对基因范围 |
| `full_native_library_size` | 来源中完整原始基因轴对应的文库量，按来源定义使用 |
| `annotation_author_celltype` | 作者标签，与项目复核标签分列保存 |
| `harmony_batch`、`harmony_batch_code` | 整合使用的批次字段，不自动等于胚胎或生物重复 |

心脏研究的规范 metadata 还需要在可获得时保留 `embryo/sample`、解剖来源、项目心脏细胞类型、标签版本、QC 决策及来源。缺失信息保持缺失，不能从存储块或技术批次推定生物样本。MELA 的 clone/tree/供体宿主信息和 Typewriter 的树叶关系以独立表连接，不要求普通表达数据也具有这些列。

## 5. 执行顺序

```text
1 小鼠基因与 GO 资源 → 2 MELA 程序 GO 基准 → 3 文献小试
                                           ↓
                  4 心脏目标群覆盖核查与输入冻结
                                           ↓
                  5 功能时序 → 6 MELA 谱系关联
                                           ↓
                  7 Typewriter 核查 → 8 候选基因
```

阶段 1–3 与目标群覆盖准备可以同期推进；正式的阶段 5–7 使用阶段 4 冻结的输入。编码验证不满足要求时，先继续资源、常规 GO 和元数据工作，不据此定稿阈值或扩大检索。

### 阶段 1：小鼠参考资源与程序成员

**位置与输入：** stage3；`config/project.json`、`data/go/go.obo`、`data/raw/mela/annotated_programs.csv` 及来源记录。

**阶段内步骤与文件命名：**

| 步骤 | 文件位置 | 职责 |
| --- | --- | --- |
| 步骤01 | `notebooks/stage1/01_GO_MELA_resources.ipynb` | 本地 GO、MELA 程序、版本和校验；导出 BP、程序成员及资源待办 |
| 步骤02 | `notebooks/stage1/02_mouse_ID_GO_annotations.ipynb` | 小鼠官方 ID、历史名称和 GO 注释；映射覆盖、歧义与未映射清单 |

上述是阶段1的步骤命名约定，具体文件是否创建、执行到哪里，记录在 01 进展文档。阶段1后续新增 Notebook 继续放在 `notebooks/stage1/`，按步骤顺序编号。数据与派生结果仍使用 `data/`、`results/`，不随 Notebook 位置移动。

**工作：**

1. 核对本体、程序表和版本，导出 BP、程序成员、成员数核查及资源待办。
2. 使用 `namespace == biological_process`；处理 obsolete、alt ID、名称和同义词，明确用于层级分析的关系边。资源表保留根节点标记，候选检索排除根术语。
3. 保留 GO—文本映射，不以全局文本去重丢失 GO 对应。
4. 准备 `data/reference/mouse/gene_identifiers.tsv`，连接原始写法、Ensembl、MGI、官方符号和历史名称，记录歧义与未映射项。
5. 准备 `data/reference/mouse/mouse_go_annotations.gaf.gz`，保留关系、证据、参考、NOT、物种和上下文扩展。记录发布、获取日期及校验值，核对本体兼容性。
6. BP 基础资源保留完整有效范围；根据小鼠适用性、心脏主题和实验目的确定检索范围，记录范围限制。

**产出与条件：** `results/01_resources/` 中的资源、映射与覆盖表；程序、基因、GO 和来源可追溯后进入常规 GO 分析。缺少注释不作为功能负例。

### 阶段 2：70 程序常规 GO 基准

**位置：** CPU 计算，stage3 保存结果小表。

追溯程序生成时实际进入候选的基因背景，开展 ORA 和多重检验；背景不明时记录限制并检查合理背景的敏感性。保留命中基因、效应量、显著性、直接/传播注释及版本。对照作者的程序名称和描述，标记可解释项、功能混合与缺口。

**产出与条件：** 70 程序 GO 对照、背景说明和人工核查表；统计与基因组成可复查，作者解释和新增解释明确区分。

### 阶段 3：编码协议与心脏相关文献小试

**位置：** stage3，环境 `pathak`。

1. 固定模型、流水线和内部批次协议；测试同文本的单句/批量、重复、输入顺序、长短文本混合、保存/重载及自查询。
2. 使用显式 GO ID、文本和行号保存向量，核对维度、有限值、非零范数和检索排名稳定性。
3. 核查查询计算规模、多条命中逐项读取及段落汇总边界，避免重复或遗漏。
4. 选择约 10 个代表程序，优先心脏主题并保留比较主题；整理可合法获取的原始文献及版本。
5. 先召回 GO 候选，再判断 gene–GO 关系，处理基因歧义、物种、否定、仅共现、实验背景和跨句证据。
6. 人工核查约 200–500 段候选；开发/调参与评估按论文划分，重复发布版本不跨集合。原文未说明的阶段、组织和细胞类型保持未明确。

**产出与条件：** 编码协议、候选证据、人工标注和相对常规 GO 的新增信息评估。阈值由本任务评估确定，余弦相似度不直接称为关系正确的概率。

### 阶段 4：心脏目标群与输入冻结

**位置：** CPU。

统计 `数据集 × 胚胎/样本 × 阶段 × 心脏细胞类型 × cell/nucleus`；核查组织来源、标签、QC、表达尺度和树叶覆盖。按证据选取心脏/心咽祖细胞与心肌细胞等目标群，明确阶段缺口和低覆盖比较。

冻结逐细胞清单、基因轴、阶段依据、标签和参考修订版本、样本信息、QC 规则及唯一 ID。检查共享文库、重复版本与同名 barcode，记录候选选择和主要统计终点。

**产出与条件：** 心脏范围说明、覆盖表、输入快照和连接报告；以胚胎或确认的生物样本为重复，重复不足的比较标为探索性。

### 阶段 5：心脏功能时序分析

**位置：** CPU。

在各数据集与技术内部按胚胎/样本和细胞类型汇总；有合适计数和重复时开展差异分析，分析时间趋势、共表达模块、GO 及阶段相关基因组合。完整细胞与细胞核先分别分析，再比较方向和可重复性。

表达功能分析回到经过 ID 核对的完整基因范围，scVI/Harmony 隐空间用于细胞对应和聚类。区分 UMI、读段与归一化数据；仅使用约 2,000 个建模基因不能代表全部功能范围。

**产出与条件：** 心脏功能时间图、核心基因组合、数据集内效应及跨来源结果；检查组成、技术和研究批次影响。“阶段特异”需要相应比较支持，未覆盖阶段不作实测结论。

### 阶段 6：MELA 功能与谱系关联

**位置：** CPU，按胚胎和目标子树读取。

核对表达、clone、tree、embryo 及供体/宿主范围，先复现一个检查案例。比较同胚胎、同类型细胞的不同克隆及不同克隆反复出现的功能程序，控制阶段、深度、周期和克隆大小，使用尊重树结构的零模型或重采样。

在具备胚胎重复的比较中核查可重复性；不同阶段的不同胚胎不能拼成同一克隆的连续测量。祖先状态和限制时间保留推断算法及不确定性。

**产出与条件：** 谱系关联、胚胎间重复性和敏感性结果；末端表达不冒充祖先实测，关联不直接解释为因果或遗传机制。

### 阶段 7：Typewriter E13.5 核查

**位置：** CPU，在冻结候选和规则后开展。

使用 E13.5 末端表达、来源明确的标签和树叶，检查目标心脏群的程序与基因组合，以及多个克隆后代的重复出现。比较完整树、骨架树、抽样和覆盖变化的结果。

记录标签投影及发育参考重叠；独立表达测量不等于标签和祖先状态完全独立。多个克隆或技术分组不能替代独立胚胎。祖先状态重建如需开展，单列为推断分析。

**产出与条件：** E13.5 支持/不支持清单、树与参考依赖、单胚胎解释边界，不把参考一致性当作独立实验验证。

### 阶段 8：候选基因、证据矩阵与图表

分列数据库注释、文献原文、表达候选和谱系关联；优先整理约 5–10 个证据相对完整的心脏相关候选，写明阶段、类型、反例、缺失证据和待验证问题。标记已经用于注释、筛选或调参的基因、文献和参考，检查循环论证。

**产出与条件：** 可追溯的 gene–GO 表、候选矩阵、功能时序和谱系图，以及后续验证建议；区分已有结果、复现、探索与独立验证。

## 6. 结果字段与复现要求

文献证据表每行至少保留：

```text
record_id, gene_ensembl_id, gene_mgi_id, gene_symbol, go_id,
relation, polarity, species, stage_text, stage_min, stage_max,
tissue, cell_type, evidence_type, evidence_text,
pmid, pmcid, doi, section, text_offset, source_file,
retrieval_score, review_status, reviewer_note,
gene_map_version, go_version, annotation_version, pipeline_version
```

表达与谱系证据独立连接，保留 `dataset`、`embryo/sample`、`cell_id`、标签版本、程序 ID、统计比较、测量/推断状态和来源。不同胚胎的同名 clone/tree 不直接合并。

每阶段保存输入校验值、参数、软件/模型版本、随机种子、输出、验证与限制。百万级数据按对象、胚胎或子树处理，避免建立全体细胞两两矩阵。

## 7. 方法和数据来源

- [PATHAK 方法项目](https://github.com/nilesh-iiita/PATHAK)。
- [MELA 方法与数据](https://github.com/jweissmanlab/MELA-2026)、[作者程序和补充表](https://mela.wi.mit.edu/download/)。
- [Typewriter / NextCell](https://next-cell.org/)。
- [MGI 小鼠 GO 注释](https://www.informatics.jax.org/mgihome/GO/querying.shtml)。
- [GO 证据类型](https://geneontology.org/docs/guide-go-evidence-codes/)、[GAF 与上下文扩展](https://geneontology.org/docs/go-annotation-file-gaf-format-2.2/)。
