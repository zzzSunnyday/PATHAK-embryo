"""Runtime configuration for the PATHAK notebooks on stage3."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PREFIX = Path("/opt/conda/envs/pathak")
JAR_PATH = PROJECT_ROOT / "resources/jars/spark-nlp-assembly-4.0.2.jar"
BIOBERT_PATH = PROJECT_ROOT / "models/biobert_pubmed_sentence"
BIOBERT_BASELINE_PATH = PROJECT_ROOT / ".backups/biobert-models/sent_biobert_pubmed_base_cased_parallel_masked_mean"
BIOBERT_PROTOCOL = "masked_mean_parallel_sentence_v1"
BIOBERT_STORAGE_REF = "pathak_biobert_masked_mean_parallel_sentence_v1"
BIOBERT_BATCH_SIZE = 16  # Retain the validated published default; 1 and 8 remain supported.
BIOBERT_TF_CONFIG = [16, 8, 40, 8]  # Global intra-op 8, inter-op 8; sentence graph limit stays 2.
BIOBERT_BASELINE_TF_CONFIG = [16, 2, 40, 8]
NLTK_DATA_PATH = PROJECT_ROOT / "models/nltk_data"
BIOBERT_REFERENCE = "en.embed_sentence.biobert.pubmed_base_cased"


def configure():
    """Select this project's paths before importing NLU or starting Spark."""
    if Path(sys.prefix).resolve() != ENV_PREFIX.resolve():
        raise RuntimeError("Select the Python (pathak) kernel or activate pathak first.")
    os.environ["JAVA_HOME"] = str(ENV_PREFIX)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["NLTK_DATA"] = str(NLTK_DATA_PATH)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "8")
    java_bin = str(ENV_PREFIX / "bin")
    paths = os.environ.get("PATH", "").split(os.pathsep)
    if java_bin not in paths:
        os.environ["PATH"] = os.pathsep.join([java_bin] + paths)
    os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT


def start_spark():
    """Start a local CPU session with the permanent local Spark NLP assembly JAR."""
    configure()
    if not JAR_PATH.is_file():
        raise FileNotFoundError("Spark NLP JAR is missing: " + str(JAR_PATH))
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("PATHAK")
        .master("local[2]")
        .config("spark.driver.memory", "8g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "2000M")
        .config("spark.jars", str(JAR_PATH))
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.jsl.settings.pretrained.cache_folder", str(PROJECT_ROOT / ".cache/sparknlp"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    from sparknlp.base import DocumentAssembler

    try:
        DocumentAssembler().setInputCol("text").setOutputCol("document")
    except Exception as exc:
        raise RuntimeError(
            "Spark NLP Java classes could not load. Restart this kernel, "
            "then run start_spark() before creating another Spark session."
        ) from exc
    return spark



def configure_biobert_pipeline(pipe, batch_size=None):
    """Select verified graph-native parallel batching or the legacy b1 fallback.

    The new graph maps a shared, runtime-optimized sentence encoder over the
    batch inside TensorFlow. Each sentence retains its unpadded arithmetic;
    CLS/SEP are retained and masked mean pooling stays inside the graph.
    Original exports keep batch=1. storageRef survives saved-pipeline reloads.
    Session configuration must also be in raw model metadata before loading.
    The first TensorFlow session fixes the process-wide intra-op pool; restart
    the notebook kernel when moving from the CPU2 baseline to global CPU8.
    Keep the captured CPU2 or CPU8 setting of explicitly loaded baselines:
    Spark NLP captures it when constructing TensorflowBert, not at predict().
    """
    sentence_models = 0
    for component in pipe.components:
        model = component.model
        if model.__class__.__name__ == "BertSentenceEmbeddings":
            parallel = model.getOrDefault("storageRef") == BIOBERT_STORAGE_REF
            size = (BIOBERT_BATCH_SIZE if parallel else 1) if batch_size is None else batch_size
            if size not in (1, 8, 16):
                raise ValueError("Verified BioBERT batch sizes are 1, 8, and 16.")
            if not parallel and size != 1:
                raise ValueError("The original BioBERT graph requires batch_size=1.")
            if parallel:
                saved_config = list(model.getOrDefault("configProtoBytes"))
                if saved_config not in (BIOBERT_TF_CONFIG, BIOBERT_BASELINE_TF_CONFIG):
                    raise ValueError("Unvalidated BioBERT TensorFlow configuration: " + str(saved_config))
                # Do not mask a saved CPU2 backend by changing its Python Param
                # to CPU8 after TensorflowBert has already captured the setting.
                model.setConfigProtoBytes(saved_config)
                if _captured_biobert_config(model) != saved_config:
                    raise RuntimeError("BioBERT backend and Python configuration differ; reload the validated model in a fresh kernel.")
                pipe._pathak_tf_config_proto_bytes = saved_config
            else:
                model.setConfigProtoBytes([16, 2, 40, 2])
            model.setBatchSize(size)
            pipe._pathak_embedding_protocol = BIOBERT_PROTOCOL if parallel else "legacy_biobert_internal_batch_1_v1"
            sentence_models += 1
    if sentence_models != 1:
        raise ValueError("Expected exactly one BertSentenceEmbeddings component.")
    return pipe

def load_biobert(model_path=None, batch_size=None):
    """Load the complete local model; missing files raise an error without downloading."""
    local_path = BIOBERT_PATH if model_path is None else Path(model_path).expanduser().resolve()
    required = [local_path / "metadata/part-00000", local_path / "bert_sentence_tensorflow"]
    missing = [str(path) for path in required if not path.is_file()]
    if not any((local_path / "fields/vocabulary").glob("part-*")):
        missing.append(str(local_path / "fields/vocabulary/part-*"))
    if missing:
        raise FileNotFoundError("Local BioBERT files are missing: " + ", ".join(missing))
    start_spark()
    import nlu

    pipe = nlu.load(BIOBERT_REFERENCE, path=str(local_path))
    pipe._pathak_model_path = local_path
    return configure_biobert_pipeline(pipe, batch_size=batch_size)


def embedding_column(frame):
    """Find the sentence embedding column produced by NLU or a saved pipeline."""
    columns = [name for name in frame.columns if "sentence_embedding" in name]
    if len(columns) != 1:
        raise ValueError("Expected one sentence embedding column, found: " + str(columns))
    return columns[0]


def _captured_biobert_config(model):
    """Read Spark NLP 4.0.2's backend capture, independently of Python Params."""
    model._transfer_params_to_java()
    backend = model._java_obj.getModelIfNotSet()
    field = backend.getClass().getDeclaredField("configProtoBytes")
    field.setAccessible(True)
    value = field.get(backend)
    return list(value.get()) if value.isDefined() else None


def _sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pipeline_identity(path):
    path = Path(path)
    return {str(file.relative_to(path)): _sha256_file(file)
            for file in sorted(path.rglob("*")) if file.is_file()}


def write_embedding_manifest(pipe, corpus_path, pickle_path, pipeline_path):
    """Record one matched corpus/vector/pipeline set after both outputs are saved."""
    import json
    import pickle
    from datetime import datetime, timezone
    import numpy as np
    import pandas as pd

    corpus_path, pickle_path, pipeline_path = map(Path, (corpus_path, pickle_path, pipeline_path))
    model_path = Path(getattr(pipe, "_pathak_model_path", BIOBERT_PATH))
    models = [c.model for c in pipe.components if c.model.__class__.__name__ == "BertSentenceEmbeddings"]
    if len(models) != 1 or models[0].getOrDefault("storageRef") != BIOBERT_STORAGE_REF:
        raise ValueError("This manifest requires the verified parallel masked-mean model.")
    if list(models[0].getOrDefault("configProtoBytes")) != BIOBERT_TF_CONFIG:
        raise ValueError("The current artifact manifest requires the validated global CPU8 configuration.")
    if _captured_biobert_config(models[0]) != BIOBERT_TF_CONFIG:
        raise ValueError("BioBERT backend has a different captured configuration; reload before publishing artifacts.")
    if models[0].getOrDefault("caseSensitive") is not True or models[0].getOrDefault("maxSentenceLength") != 128:
        raise ValueError("This embedding protocol requires caseSensitive=True and maxSentenceLength=128.")
    table = pd.read_csv(corpus_path, sep="\t")
    with pickle_path.open("rb") as handle:
        frame = pickle.load(handle)
    normalize = lambda value: " ".join(str(value).split())
    if table.Definition.map(normalize).tolist() != frame.document.map(normalize).tolist():
        raise ValueError("GO definitions and saved vectors do not match row by row.")
    vectors = np.stack(frame[embedding_column(frame)].tolist())
    if vectors.shape != (len(table), 768) or not np.isfinite(vectors).all():
        raise ValueError("Expected one finite 768-dimensional vector per GO definition.")
    marker = json.loads((model_path / "pathak_pooling.json").read_text())
    manifest = {
        "protocol": BIOBERT_PROTOCOL,
        "internal_batch_size": models[0].getOrDefault("batchSize"),
        "output_level": "document", "rows": len(table), "dimension": 768,
        "case_sensitive": models[0].getOrDefault("caseSensitive"),
        "max_sentence_length": models[0].getOrDefault("maxSentenceLength"),
        "pooling": "input_mask mean; retain CLS/SEP; exclude PAD; float32",
        "execution": "TensorFlow MapDefun; parallel sentence forwards; shared weights; legacy optimized arithmetic",
        "tf_config_proto_bytes": BIOBERT_TF_CONFIG,
        "go_input_sha256": _sha256_file(corpus_path),
        "pickle_sha256": _sha256_file(pickle_path),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "model_archive_sha256": _sha256_file(model_path / "bert_sentence_tensorflow"),
        "model_metadata_sha256": _sha256_file(model_path / "metadata/part-00000"),
        "graph_sha256": marker["graph_sha256"],
        "sentence_max_intra_op_parallelism": 2,
        "omp_num_threads": 2,
        "requires_fresh_jvm_for_thread_pool": True,
        "pipeline_files": _pipeline_identity(pipeline_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "report": "resources/checks/embedding-performance-global8-validation.json",
        "performance_report": "resources/checks/embedding-performance-global8-benchmark.json",
    }
    target = pickle_path.with_name("embedding_manifest.json")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    return manifest


def check_embedding_manifest(corpus_path, pickle_path, pipeline_path):
    """Reject stale or mixed artifacts before encoding queries against a GO library."""
    import json
    corpus_path, pickle_path, pipeline_path = map(Path, (corpus_path, pickle_path, pipeline_path))
    manifest = json.loads(pickle_path.with_name("embedding_manifest.json").read_text())
    if manifest.get("protocol") != BIOBERT_PROTOCOL:
        raise ValueError("The GO vector library uses a different embedding protocol; rebuild it with Notebook 2.")
    expected = {
        "model_path": str(BIOBERT_PATH.relative_to(PROJECT_ROOT)),
        "tf_config_proto_bytes": BIOBERT_TF_CONFIG,
        "case_sensitive": True,
        "max_sentence_length": 128,
        "go_input_sha256": _sha256_file(corpus_path),
        "pickle_sha256": _sha256_file(pickle_path),
        "model_archive_sha256": _sha256_file(BIOBERT_PATH / "bert_sentence_tensorflow"),
        "model_metadata_sha256": _sha256_file(BIOBERT_PATH / "metadata/part-00000"),
        "pipeline_files": _pipeline_identity(pipeline_path),
    }
    for field, actual in expected.items():
        if manifest.get(field) != actual:
            raise ValueError("Embedding artifact mismatch: " + field + "; rerun Notebook 2's encode/save cells.")
    return manifest
