"""Fair warmed GO-definition benchmark: original b1 versus masked b1/b8/b16."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    assert json.loads((args.work / "validation.json").read_text())["status"] == "passed"
    assert json.loads((args.work / "validation-fresh.json").read_text())["status"] == "passed"
    import numpy as np
    import pandas as pd
    from pathak_runtime import load_biobert, start_spark, embedding_column

    spark_started = time.perf_counter()
    spark = start_spark()
    spark_start_seconds = time.perf_counter() - spark_started
    try:
        table_path = ROOT / "tutorials/GO_BP_Ontology_corups_Filtered.tsv"
        table = pd.read_csv(table_path, sep="\t")
        # Fixed uniform sample in length rank, then deterministic shuffle: covers the GO length distribution.
        lengths = table.Definition.str.split().map(len).to_numpy()
        ranked = np.argsort(lengths, kind="stable")
        indices = ranked[np.linspace(0, len(table) - 1, min(args.rows, len(table)), dtype=int)]
        np.random.default_rng(20261021).shuffle(indices)
        texts = table.iloc[indices].Definition.astype(str).to_list()
        configurations = [("original_batch1", "sent_biobert_pubmed_base_cased", 1),
                          ("masked_batch1", "sent_biobert_pubmed_base_cased_masked_mean", 1),
                          ("masked_batch8", "sent_biobert_pubmed_base_cased_masked_mean", 8),
                          ("masked_batch16", "sent_biobert_pubmed_base_cased_masked_mean", 16)]
        pipelines = {}
        results = {}

        def encode(pipe, values):
            frame = pipe.predict(values, output_level="document")
            norm = lambda text: " ".join(str(text).split())
            assert [norm(t) for t in frame.document] == [norm(t) for t in values]
            values = np.stack(frame[embedding_column(frame)].to_list())
            assert values.shape[1] == 768 and np.isfinite(values).all()
            return values

        for name, model, batch_size in configurations:
            started = time.perf_counter()
            pipe = load_biobert(model_path=ROOT / ".backups/biobert-models" / model)
            models = [c.model for c in pipe.components if c.model.__class__.__name__ == "BertSentenceEmbeddings"]
            assert len(models) == 1
            models[0].setBatchSize(batch_size)
            assert models[0].getOrDefault("batchSize") == batch_size
            load_seconds = time.perf_counter() - started
            started = time.perf_counter()
            encode(pipe, texts[:32])  # includes lazy model/session initialization, excluded from timing below
            first_prediction_seconds = time.perf_counter() - started
            started = time.perf_counter()
            encode(pipe, texts[32:64])
            second_warmup_seconds = time.perf_counter() - started
            pipelines[name] = pipe
            results[name] = {"batch_size": batch_size, "load_seconds": load_seconds,
                             "first_prediction_32_seconds": first_prediction_seconds,
                             "second_warmup_32_seconds": second_warmup_seconds,
                             "encoding_seconds": []}
            print(json.dumps({"prepared": name, **results[name]}), flush=True)
        order = list(pipelines)
        round_orders = []
        baseline = None
        for trial in range(args.rounds):
            # Rotate configurations across rounds to reduce systematic order/load bias.
            trial_order = order[trial:] + order[:trial]
            round_orders.append(trial_order)
            for name in trial_order:
                started = time.perf_counter()
                vectors = encode(pipelines[name], texts)
                elapsed = time.perf_counter() - started
                results[name]["encoding_seconds"].append(elapsed)
                if baseline is None:
                    assert name == "original_batch1"
                    baseline = vectors
                error = float(np.max(np.abs(vectors.astype(float) - baseline.astype(float))))
                cosine = np.sum(vectors.astype(float) * baseline, axis=1) / (
                    np.linalg.norm(vectors.astype(float), axis=1) * np.linalg.norm(baseline.astype(float), axis=1))
                assert error <= 1e-5 and float(cosine.min()) >= 0.999999
                results[name]["vs_original_max_abs"] = max(error, results[name].get("vs_original_max_abs", 0.0))
                results[name]["vs_original_min_cosine"] = min(float(cosine.min()), results[name].get("vs_original_min_cosine", 1.0))
                print(json.dumps({"trial": trial + 1, "configuration": name,
                                  "seconds": elapsed, "items_per_second": len(texts) / elapsed,
                                  "max_abs": error}), flush=True)
        original_seconds = statistics.median(results["original_batch1"]["encoding_seconds"])
        for name, values in results.items():
            median = statistics.median(values["encoding_seconds"])
            values.update(median_encoding_seconds=median, items_per_second=len(texts) / median,
                          speedup_vs_original_batch1=original_seconds / median)
        eligible = [name for name in results if name.startswith("masked_batch")]
        chosen = min(eligible, key=lambda name: results[name]["median_encoding_seconds"])
        report = {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
                  "rows": len(texts), "rounds": args.rounds, "round_orders": round_orders,
                  "go_input_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
                  "row_indices": indices.tolist(), "sample_word_length_quantiles": np.quantile(lengths[indices], [0, .25, .5, .75, .9, 1]).tolist(),
                  "measurement": "wall time of warmed NLU predict + row alignment/vector extraction; excludes model load and warmups",
                  "spark_start_seconds": spark_start_seconds,
                  "spark_master": spark.sparkContext.master,
                  "tf_config_proto_bytes": [16, 2, 40, 2],
                  "threads": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]},
                  "hardware": "Intel Xeon Platinum 8480+; shared host; cgroup CPU quota 48 cores",
                  "results": results, "selected_configuration": chosen,
                  "selected_batch_size": results[chosen]["batch_size"]}
        (args.work / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
