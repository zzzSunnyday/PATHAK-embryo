# PATHAK
NLP-based method to mine gene and function relationships from published articles.

<img src="https://nilesh-iiita.github.io/PATHAK/_static/PATHAK.png" alt="PATHAK" width="200" height="220" title="PATHAK">


[Demo](https://nilesh-iiita.github.io/PATHAK/intro.html)

## Summary 

Understanding gene function is crucial for advancing our knowledge of biological systems and developing new treatments for diseases. However, it is a complex and challenging task due to the complexity of biological systems, the constantly evolving understanding of gene function, and the specialized language used in research articles. To address this, the authors propose PATHAK, a new method that uses a pre-trained Transformer language model to identify relationships between genes and their potential GO term definition. They applied this method to a large dataset of papers on Arabidopsis thaliana and hope to continue exploring its potential applications in advancing our understanding of gene function and facilitating the discovery of new treatments for diseases.

## stage3 环境与教学

项目目录为 /root/gpufree-data/PATHAK，Conda 环境名为 pathak。教学 Notebook 及生成内容统一保存在 tutorials/。模型位于 models/，运行资源位于 resources/。

### 1. 软件环境

environment.yml 记录当前 Linux x86_64 环境，使用 conda-forge，环境名称为 pathak。

| 组件 | 版本 |
| --- | --- |
| Python / OpenJDK | 3.8.20 / 8.0.472 |
| pip / setuptools / wheel | 24.3.1 / 69.5.1 / 0.45.1 |
| PySpark / Py4J | 3.2.3 / 0.10.9.5 |
| Spark NLP / NLU | 4.0.2 / 4.0.0 |
| NumPy / pandas | 1.23.5 / 1.5.2 |
| SciPy / scikit-learn / PyArrow | 1.9.3 / 1.2.0 / 10.0.1 |
| NetworkX / obonet / NLTK | 2.8.8 / 0.3.1 / 3.8.1 |
| JupyterLab / ipykernel | 3.6.8 / 6.19.4 |

日常使用：

```bash
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate pathak
cd /root/gpufree-data/PATHAK
```

set +u 用于兼容当前 OpenJDK Conda 激活钩子。需要重建环境时，在项目根目录使用：

```bash
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  conda env create --file environment.yml
```

环境文件管理软件包，模型和运行数据使用下述本地资源。

### 2. 当前目录

```text
PATHAK/
├── environment.yml
├── README.md
├── pathak_runtime.py
├── tutorials/
│   ├── local_0_environment_check.ipynb
│   ├── local_1_GO.ipynb
│   ├── local_2_GO_term_embedding.ipynb
│   ├── local_3_PhenoMiner_plant.ipynb
│   ├── GO_BP_Ontology_corups.tsv
│   ├── GO_BP_Ontology_corups_Filtered.tsv
│   ├── predictions_full_BP.pickle
│   └── pipe_full_BP/
├── data/go/go.obo
├── Auxiliary_Files/
│   ├── TAIR_GOTERMS.txt
│   └── 1.2_ATH_Gene_all.lst
├── Example_data/
│   ├── PMC1459476.xml
│   └── PMC1802096.xml
├── models/
│   ├── sent_biobert_pubmed_base_cased/
│   │   ├── bert_sentence_tensorflow
│   │   ├── fields/vocabulary/
│   │   └── metadata/
│   └── biobert-source.json
└── resources/
    ├── jars/spark-nlp-assembly-4.0.2.jar
    ├── nltk_data/tokenizers/punkt/
    ├── archives/punkt.zip
    ├── local-assets-manifest.json
    ├── SHA256SUMS
    └── checks/
```

### 3. 本地输入资源

| 资源 | 项目内路径 | 用途 |
| --- | --- | --- |
| GO 本体 | data/go/go.obo | GO 名称、定义和关系 |
| BioBERT | models/sent_biobert_pubmed_base_cased/ | 768 维、区分大小写的句向量模型 |
| Spark NLP JAR | resources/jars/spark-nlp-assembly-4.0.2.jar | Java 运行库 |
| NLTK punkt | resources/nltk_data/ | 论文分句 |
| TAIR GO 列表 | Auxiliary_Files/TAIR_GOTERMS.txt | 第二课的拟南芥 GO 筛选 |
| 拟南芥基因词典 | Auxiliary_Files/1.2_ATH_Gene_all.lst | 第三课基因名称匹配 |
| 示例论文 | Example_data/ | 本地 XML 输入 |

BioBERT 使用完整目录：bert_sentence_tensorflow 是模型文件，fields/vocabulary/ 是词表，metadata/ 是加载配置。模型来源见 models/biobert-source.json。load_biobert() 从本地目录读取，缺少模型文件时会报错。

```python
MODEL_PATH = PROJECT_ROOT / "models" / "sent_biobert_pubmed_base_cased"
pipe = load_biobert(model_path=MODEL_PATH)
```

当前 GO 版本为 releases/2026-07-26，SHA256：

```text
d3593751d885ca160b2ab7baf6c7eccd88ca3c4599f79436c674bad661095ff0
```

官方 GO 本体发布地址见 [Gene Ontology](https://current.geneontology.org/ontology/go.obo)。资源清单为 resources/local-assets-manifest.json，可在项目根目录执行 sha256sum -c resources/SHA256SUMS 核对本地文件。

### 4. Jupyter 内核与路径

激活 pathak 后注册内核：

```bash
python -m ipykernel install \
  --user \
  --name pathak \
  --display-name "Python (pathak)" \
  --env JAVA_HOME "$CONDA_PREFIX" \
  --env PYSPARK_PYTHON "$CONDA_PREFIX/bin/python" \
  --env PYSPARK_DRIVER_PYTHON "$CONDA_PREFIX/bin/python" \
  --env SPARK_LOCAL_IP "127.0.0.1" \
  --env NLTK_DATA "/root/gpufree-data/PATHAK/resources/nltk_data"
```

选择 Python (pathak) 内核。pathak_runtime.py 配置 Java、Python 和资源路径，Spark 使用 CPU、local[2]、8 GB 驱动内存。

每份教学的启动单元定义：

```python
PROJECT_ROOT = Path("/root/gpufree-data/PATHAK")
TUTORIALS_ROOT = PROJECT_ROOT / "tutorials"
```

模型、GO 本体、词典和论文从 PROJECT_ROOT 读取；教学的 GO 表、向量 pickle、处理流水线和临时日志从 TUTORIALS_ROOT 读写。文件引用不依赖 Jupyter 的启动目录。

### 5. 教学顺序

| 顺序 | Notebook | 主要输入 | 生成内容（tutorials/） |
| --- | --- | --- | --- |
| 0 | [环境与模型检查](tutorials/local_0_environment_check.ipynb) | JAR、BioBERT、NLTK | 两句文本的 768 维向量检查 |
| 1 | [GO 语料](tutorials/local_1_GO.ipynb) | data/go/go.obo | GO_BP_Ontology_corups.tsv |
| 2 | [GO 定义向量](tutorials/local_2_GO_term_embedding.ipynb) | GO 语料、TAIR GO 列表、BioBERT | GO_BP_Ontology_corups_Filtered.tsv、predictions_full_BP.pickle、pipe_full_BP/ |
| 3 | [论文处理](tutorials/local_3_PhenoMiner_plant.ipynb) | 第二课输出、基因词典、论文 XML | Notebook 中的候选关联、表格与网络；temp.del 日志 |

第二课的 pickle 保存预测结果，pipe_full_BP/ 保存可复用的处理流水线。GO 表与 pickle 必须来自同一次向量生成，并保持逐行对应。

可在项目根目录通过命令行执行，执行后的 Notebook 也保存到 tutorials/：

```bash
jupyter nbconvert --to notebook --execute tutorials/local_0_environment_check.ipynb \
  --ExecutePreprocessor.kernel_name=pathak --ExecutePreprocessor.timeout=1800 \
  --output-dir=tutorials --output=local_0_environment_check.executed

jupyter nbconvert --to notebook --execute tutorials/local_1_GO.ipynb \
  --ExecutePreprocessor.kernel_name=pathak --ExecutePreprocessor.timeout=1800 \
  --output-dir=tutorials --output=local_1_GO.executed

jupyter nbconvert --to notebook --execute tutorials/local_2_GO_term_embedding.ipynb \
  --ExecutePreprocessor.kernel_name=pathak --ExecutePreprocessor.timeout=-1 \
  --output-dir=tutorials --output=local_2_GO_term_embedding.executed

jupyter nbconvert --to notebook --execute tutorials/local_3_PhenoMiner_plant.ipynb \
  --ExecutePreprocessor.kernel_name=pathak --ExecutePreprocessor.timeout=-1 \
  --output-dir=tutorials --output=local_3_PhenoMiner_plant.executed
```

timeout=-1 表示不限制单个单元格的运行时间。输出文件名中的 corups 沿用源码拼写。

### 6. 第三课的处理流程

流程：加载 GO 定义与向量 → 加载相同 BioBERT 流水线 → 筛选含基因的段落 → 对句子进行 GO 相似度检索 → 汇总候选关联、表格和网络。

| 函数 | 作用 |
| --- | --- |
| get_sim_df_for_string | 查询句向量与 GO 定义向量的余弦相似度 |
| query_go | 取 Top 条定义记录，再按阈值筛选 |
| para_pass | 从段落中匹配基因词典 |
| read_pubmed | 读取论文、组织元数据和逐段结果 |
| Plot_data | 展示候选表格和 gene–GO 网络 |

相似度表示文本接近程度。段落中出现某个基因和功能描述，只形成候选证据，需要核查原文是否支持该基因参与此功能。Organism 参数记录名称匹配结果，基因识别仍由拟南芥词典决定。

当前源码的实现限制已在第三课中注明：余弦计算形成多余的 N×N 矩阵；多条 GO 命中循环重复读取第一条；绘图汇总循环会漏掉最后一个段落。

### 7. 检查记录

- environment.yml 对应的软件依赖已通过 pip check。
- stage3 的文件完整性和运行检查见 resources/checks/stage3-transfer-verification.json 和 resources/checks/stage3-runtime-validation.json。
- 本地 BioBERT 的两句文本检查得到 (2, 768) 的有限、非零向量，记录见 resources/checks/local-model-smoke-test.json。
- 教学路径、输入对齐和第三课处理流水线检查见 resources/checks/tutorials-layout-check.json。

Java 类加载失败时，重启内核并从启动单元开始；WARN NativeCodeLoader 在已验证的本地运行中不影响计算。

当前编码一致性检查发现批处理差异：单句重编码与已保存批量向量不完全一致，批量输入可重现抽查向量。正式分析前需统一并验证编码方式、校准阈值。记录见 resources/checks/tutorials-embedding-consistency.json。
