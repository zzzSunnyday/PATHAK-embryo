# 01：资源与数据准备进展

核查日期：2026-10-21（Asia/Tokyo）。本文件记录资源和数据准备的事实，研究设计见 [00 计划](00_mouse_embryo_pathak_PLAN.md)，编码与分析状态见 [02 进展](02_mouse_embryo_pathak_analysis_PROGRESS.md)。

项目内路径相对于 `PATHAK/`。`CPU:` 表示 CPU 的 `/hy-tmp/` 数据根目录。数量是指定文件的核查快照，后续更新需记录来源与时间。

## 1. 项目入口与资源状态

| 项目 | 状态 | 证据/路径 |
| --- | --- | --- |
| 项目配置 | 已存在，字段已核对 | `config/project.json` |
| 命名环境 | 项目配置使用 `pathak` | `environment.yml`、`pathak_runtime.py`；本次文档整理未重跑全部软件测试 |
| GO 本体 | 已核查版本及校验值 | `data/go/go.obo` |
| MELA 程序副本 | 已核查 70 程序和来源 | `data/raw/mela/annotated_programs.csv`、`data/raw/mela/SOURCE.json` |
| 步骤01：GO 与 MELA 程序检查 | 已有；保存的 5 个代码单元均有执行记录且无错误输出 | `notebooks/stage1/01_GO_MELA_resources.ipynb` |
| 步骤01资源底表 | 已生成并核查 | `results/01_resources/` |
| 小鼠基因标识 | 待准备 | `data/reference/mouse/gene_identifiers.tsv` |
| 小鼠 GO 注释 | 待准备 | `data/reference/mouse/mouse_go_annotations.gaf.gz` |

步骤01输出不等于完成小鼠 ID 映射、富集或文献验证。当前待办以 `results/01_resources/pending_resources.tsv` 和 `resource_audit.json` 为依据。

阶段1的 Notebook 统一放在 `notebooks/stage1/`，按阶段内步骤编号：

| 步骤 | 文件位置 | 文件状态 |
| --- | --- | --- |
| 步骤01 | `notebooks/stage1/01_GO_MELA_resources.ipynb` | 已整理到阶段1目录，代码与保存输出保留 |
| 步骤02 | `notebooks/stage1/02_mouse_ID_GO_annotations.ipynb` | 已确定命名和位置，Notebook 内容尚未创建 |

本次只整理目录、命名及引用，不执行步骤02。阶段内步骤02不等于 00 计划中的研究阶段2。

## 2. 步骤01的资源核查结果

依据 `results/01_resources/resource_audit.json` 及相应 TSV：

| 核查项 | 结果 |
| --- | ---: |
| GO 版本 | `releases/2026-07-26` |
| 全 GO 图节点 | 38,092 |
| BP 术语，含根节点 | 23,974 |
| MELA 程序 | 70 |
| 程序—基因成员记录 | 1,322 |
| 不同原始基因名称 | 1,322 |
| 程序 size 不一致 | 0 |
| 程序内重复成员行 | 0 |

固定输入 SHA256：

- `data/go/go.obo`：`d3593751d885ca160b2ab7baf6c7eccd88ca3c4599f79436c674bad661095ff0`。
- `data/raw/mela/annotated_programs.csv`：`1489862271bbeb10a1dc9f0eaf7355d131fdfdfdb8455a55b3e2030dca9523c4`。

BP 资源已核查 namespace、GO ID 唯一性及根节点；原始基因写法仍保留，尚未变成官方稳定 ID。

### 输出表的实际行与列

| 路径 | 一行代表什么 | 列 | 核查行数 |
| --- | --- | --- | ---: |
| `results/01_resources/go_bp_terms.tsv` | 一个 BP 术语 | `GO, Name, Namespace, Definition_OBO, Synonyms_OBO, is_namespace_root` | 23,974 |
| `results/01_resources/mela_program_gene_membership.tsv` | 一个程序中的一个成员 | `program, gene_raw, member_position` | 1,322 |
| `results/01_resources/mela_program_audit.tsv` | 一个程序的成员数核查 | `program, name, size, declared_size, parsed_gene_entries, unique_gene_names, matches_declared_size` | 70 |
| `results/01_resources/pending_resources.tsv` | 一项待准备资源 | `resource, path, file_exists, validation_status` | 2 |

## 3. CPU 表达输入的数量快照

依据 `CPU:seurat_project/results/v4/stage1/qc_cells_summary.csv` 的 `in_analysis`。这是进入该入口的观察单位数量，不是独立胚胎数、最终心脏细胞数或最终 QC 后规模。多个对象仍有独立基因轴，不能把总量当作一个统一表达矩阵的列数。

| 来源组 | 观察单位及范围 | 输入数量 |
| --- | --- | ---: |
| 完整细胞公共数据 | 下表列出的 15 个 scRNA-seq 数据集，MELA 另列 | 803,268 个细胞 |
| 公共细胞核数据 | `16_GSE186068`、`18_GSE228590`，Typewriter 另列 | 6,730,863 个核 |
| MELA | `24_GSE342607`，表达与谱系联合来源 | 1,792,442 个细胞 |
| Typewriter | `25_GSE341627`，一个 E13.5 胚胎的末端数据 | 1,584,848 个核 |

### 完整细胞公共数据包含哪些数据集

该表每行是一个数据集，第二列为当前入口计入的细胞数量。

| 数据集 ID | `in_analysis` |
| --- | ---: |
| `01_GSE123046` | 125,036 |
| `02_GSE136689` | 31,268 |
| `03_GSE156125` | 64,366 |
| `04_GSE161143` | 2,712 |
| `05_GSE182135` | 57,840 |
| `06_GSE76118` | 2,859 |
| `07_GSE87038` | 1,904 |
| `08_GSE126128` | 36,753 |
| `09_GSE160136` | 3,549 |
| `10_E-MTAB-6967` | 82,342 |
| `12_GSE121708` | 1,153 |
| `19_E-MTAB-11763` | 314,027 |
| `21_E-MTAB-7320` | 41,025 |
| `22_GSE158820` | 20,706 |
| `23_GSE153162` | 17,728 |

数据集编号不等于胚胎或重复。心脏研究需要进一步依据实际组织、标签、阶段和样本信息筛选，上述数量没有完成该筛选。

### CPU 汇总表的实际列

`qc_cells_summary.csv` 每行是一个 `dataset × branch` 组合；同一数据集可以有多个分支记录。列为：

```text
dataset, branch, cells, in_analysis,
low_library, low_features, high_mito, numeric_doublet_risk,
author_quality_fail, prior_low_quality, prior_maternal, prior_doublet,
female_in_male, male_in_female,
doublet, doublet_borderline, doublet_rate, doublet_role
```

`cells`、`in_analysis` 是数量；其他计数列是对应标记的数量，标记可能重叠。`doublet_role` 说明评分用途，不能只凭某一列求和推导最终排除数。该 CSV 没有逐基因表达列。

表达矩阵和逐细胞 metadata 的轴含义、实际字段示例见 [00 计划第 4 节](00_mouse_embryo_pathak_PLAN.md)。

## 4. MELA 与 Typewriter 的输入位置及范围

| 输入 | CPU 路径 | 说明 |
| --- | --- | --- |
| MELA 作者程序 | `CPU:GSE342607/author_metadata/annotated_programs.csv` | stage3 副本的来源 |
| MELA 胚胎表 | `CPU:GSE342607/author_metadata/embryo_metadata.tsv` | 胚胎、阶段和技术批次关系 |
| MELA 阶段统计 | `CPU:GSE342607/author_metadata/stage_summary.tsv` | 阶段与观察数量 |
| MELA 表达/谱系 | `CPU:GSE342607/downloads/zenodo_h5td/` | 16 个逐胚胎 H5TD |
| MELA 谱系通用格式 | `CPU:GSE342607/downloads/geo_lineage/` | 字符矩阵和树 |
| Typewriter 表达 | `CPU:dna_typewriter_mouse_2026/expression/adata_embryo3.h5ad` | 核 × 基因条目，1,584,848 × 78,258 |
| Typewriter 作者标签 | `CPU:dna_typewriter_mouse_2026/code/dtt-mouse-v1.1.0/support_data/cell_metadata.v8.txt.gz` | 按细胞 ID 连接 |
| Typewriter 完整/骨架树 | `CPU:dna_typewriter_mouse_2026/trees/` | 两个版本分别保留 |

MELA 覆盖 E7.5–E10.0；前五个半日阶段各 3 个胚胎，E10.0 为 1 个。`capture` 是技术批次，donor/host 不直接等于候选基因敲除/对照。

Typewriter H5AD 的 `obs` 仅有细胞 ID，细胞类型从外部标签连接。其 GEO 独立基因列表与 MTX 列数不一致，分析优先使用自带完整基因注释的 H5AD。核查依据为 `CPU:dna_typewriter_mouse_2026/metadata/gene_annotation_check.json`。

## 5. 下一步资源工作

- [x] 项目配置、GO 和 MELA 本地输入核查。
- [x] 步骤01 BP、成员、size 审计和待办表。
- [x] 将阶段1的步骤01整理到 `notebooks/stage1/`，明确步骤02的文件命名和职责。
- [ ] 准备小鼠官方基因标识及历史名称映射。
- [ ] 准备小鼠 GO 注释，保留关系、NOT、证据、参考和版本。
- [ ] 报告 1,322 个原始名称的稳定 ID 覆盖、歧义和未映射项。
- [ ] 追溯程序基因选择背景，为常规 GO 基准提供输入。

阶段 1 的后续结果在本文件更新；八阶段研究设计持续维护在 00，避免把运行结果写回计划正文。
