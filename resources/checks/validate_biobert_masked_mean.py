"""Exercise raw and serialized masked-mean models against original batch=1.

Run with pathak Python. --work holds temporary pipeline, reference vectors and
measured reports; no published tutorial artifacts are changed by this script.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import pandas as pd
    from pathak_runtime import configure_biobert_pipeline, embedding_column, load_biobert, start_spark

    spark = start_spark()
    checks = []
    loads = {}

    def emit(value):
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def measure(name, actual, expected):
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        assert np.isfinite(actual).all() and np.linalg.norm(actual, axis=-1).min() > 0
        absolute = float(np.max(np.abs(actual - expected)))
        cosine = np.sum(actual * expected, axis=-1) / (
            np.linalg.norm(actual, axis=-1) * np.linalg.norm(expected, axis=-1))
        row = {"case": name, "rows": int(actual.shape[0]), "max_abs": absolute,
               "min_cosine": float(np.min(cosine))}
        checks.append(row)
        emit(row)
        assert absolute <= 1e-5 and row["min_cosine"] >= 0.999999, row

    def batch(pipe, size):
        models = [c.model for c in pipe.components
                  if c.model.__class__.__name__ == "BertSentenceEmbeddings"]
        assert len(models) == 1
        models[0].setBatchSize(size)
        assert models[0].getOrDefault("batchSize") == size
        assert models[0].getOrDefault("maxSentenceLength") == 128
        assert models[0].getOrDefault("caseSensitive") is True
        return pipe

    def encode(pipe, values):
        expected_texts = [values] if isinstance(values, str) else list(values)
        frame = pipe.predict(values, output_level="document")
        norm = lambda t: " ".join(str(t).split())
        assert [norm(t) for t in frame.document] == [norm(t) for t in expected_texts], "output row alignment"
        vectors = np.stack(frame[embedding_column(frame)].to_list())
        assert vectors.shape == (len(expected_texts), 768), vectors.shape
        return vectors

    try:
        import nlu
        if args.fresh:
            data = np.load(args.work / "validation_reference.npz", allow_pickle=False)
            texts = data["texts"].tolist()
            reference = data["vectors"]
            pipeline = nlu.load(path=str(args.work / "validation_saved_pipeline"))
            raw_batch = [c.model.getOrDefault("batchSize") for c in pipeline.components
                         if c.model.__class__.__name__ == "BertSentenceEmbeddings"]
            assert raw_batch == [16], raw_batch
            # Do not apply the legacy helper: this is also a check of serialized graph and params.
            measure("fresh_saved_batch16_vs_original_single", encode(pipeline, texts), reference)
            measure("fresh_saved_reversed", encode(pipeline, texts[::-1])[::-1], reference)
            measure("fresh_saved_series", encode(pipeline, pd.Series(texts)), reference)
            report_name = "validation-fresh.json"
        else:
            table = pd.read_csv(ROOT / "tutorials/GO_BP_Ontology_corups_Filtered.tsv", sep="\t")
            indices = np.linspace(0, len(table) - 1, 20, dtype=int).tolist()
            texts = ["Heart development", "The differentiation of cardiac progenitor cells",
                     "A process involving DNA repair, transcription, and chromatin organization",
                     "β-catenin regulates cardiomyocyte differentiation", "Nkx2-5 and GATA4 in E9.5 mouse heart",
                     "cell " * 124, "cell " * 125, "cell " * 126, "cell " * 127,
                     "cell " * 256, "cell " * 600, "Cell division\nwith DNA repair"]
            texts += [str(table.iloc[i].Definition) for i in indices]
            # Strip terminal spaces so document normalization is independent of input containers.
            texts = [t.strip() for t in texts]
            started = time.perf_counter()
            original = batch(load_biobert(model_path=ROOT / ".backups/biobert-models/sent_biobert_pubmed_base_cased"), 1)
            loads["original_load_seconds"] = time.perf_counter() - started
            reference = encode(original, texts)
            for index in [0, 1, 5, 7, 8, 9, 10, 31]:
                measure("original_batch1_vs_single_%d" % index,
                        encode(original, texts[index]), reference[index:index + 1])
            np.savez(args.work / "validation_reference.npz", texts=np.asarray(texts), vectors=reference)
            for size in [1, 8, 16]:
                started = time.perf_counter()
                pipe = batch(load_biobert(model_path=ROOT / ".backups/biobert-models/sent_biobert_pubmed_base_cased_masked_mean"), size)
                loads["patched_batch%d_load_seconds" % size] = time.perf_counter() - started
                measure("patched_batch%d_mixed" % size, encode(pipe, texts), reference)
                measure("patched_batch%d_series" % size,
                        encode(pipe, pd.Series(texts, index=np.arange(len(texts)) * 3 + 7)), reference)
                order = np.random.default_rng(20261021).permutation(len(texts))
                measure("patched_batch%d_shuffled" % size,
                        encode(pipe, [texts[i] for i in order])[np.argsort(order)], reference)
                repeated = [texts[0]] * 17
                measure("patched_batch%d_repeated17" % size, encode(pipe, repeated),
                        np.repeat(reference[0:1], len(repeated), axis=0))
                chunks = []
                for left, right in [(0, 1), (1, 6), (6, 19), (19, len(texts))]:
                    chunks.append(encode(pipe, texts[left:right]))
                measure("patched_batch%d_call_chunks_1_5_13_13" % size,
                        np.concatenate(chunks), reference)
                for index in [0, 7, 8, 10]:
                    measure("patched_batch%d_single_%d" % (size, index),
                            encode(pipe, texts[index]), reference[index:index + 1])
                if size == 16:
                    pipe.save(str(args.work / "validation_saved_pipeline"), overwrite=False)
            reloaded = nlu.load(path=str(args.work / "validation_saved_pipeline"))
            assert [c.model.getOrDefault("batchSize") for c in reloaded.components
                    if c.model.__class__.__name__ == "BertSentenceEmbeddings"] == [16]
            measure("same_process_saved_reload", encode(reloaded, texts), reference)
            report_name = "validation.json"
        report = {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
                  "thresholds": {"max_abs": 1e-5, "min_cosine": 0.999999},
                  "output_level": "document", "dimension": 768,
                  "maximum_absolute_error": max(item["max_abs"] for item in checks),
                  "minimum_cosine": min(item["min_cosine"] for item in checks),
                  "case_count": len(checks), "comparisons": checks, "model_loads": loads,
                  "max_sentence_length": 128, "case_sensitive": True,
                  "threads": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]}}
        (args.work / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        emit({"status": "passed", "report": str(args.work / report_name),
              "max_abs": report["maximum_absolute_error"], "min_cosine": report["minimum_cosine"]})
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
