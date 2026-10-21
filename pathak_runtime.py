"""Runtime configuration for the PATHAK notebooks on stage3."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PREFIX = Path("/opt/conda/envs/pathak")
JAR_PATH = PROJECT_ROOT / "resources/jars/spark-nlp-assembly-4.0.2.jar"
BIOBERT_PATH = PROJECT_ROOT / "models/sent_biobert_pubmed_base_cased"
NLTK_DATA_PATH = PROJECT_ROOT / "resources/nltk_data"
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
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
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


def load_biobert(model_path=None):
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
    for component in pipe.components:
        if hasattr(component.model, "setConfigProtoBytes"):
            # TensorFlow ConfigProto: two intra-op and two inter-op threads.
            component.model.setConfigProtoBytes([16, 2, 40, 2])
    return pipe


def embedding_column(frame):
    """Find the sentence embedding column produced by NLU or a saved pipeline."""
    columns = [name for name in frame.columns if "sentence_embedding" in name]
    if len(columns) != 1:
        raise ValueError("Expected one sentence embedding column, found: " + str(columns))
    return columns[0]
