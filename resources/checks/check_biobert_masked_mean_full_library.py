"""Check all candidate GO rows against the published original batch=1 library.

Default: read the candidate pickle made by the repair run. --encode performs
a fresh full candidate encoding. Neither mode replaces published artifacts.
Returns a nonzero exit status when either strict acceptance threshold fails.
"""

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-library", type=Path)
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--batch-size", type=int, choices=[1, 8, 16], default=8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    from pathak_runtime import embedding_column, load_biobert, start_spark

    corpus = ROOT / "tutorials/GO_BP_Ontology_corups_Filtered.tsv"
    table = pd.read_csv(corpus, sep="\t")
    manifest = json.loads((ROOT / "tutorials/embedding_manifest.json").read_text())
    reference_path = ROOT / "tutorials/predictions_full_BP.pickle"
    if manifest["protocol"] != "legacy_biobert_internal_batch_1_v1":
        from pathak_runtime import BIOBERT_PATH
        marker = json.loads((BIOBERT_PATH / "pathak_pooling.json").read_text())
        reference_path = ROOT / marker["legacy_reference_pickle"]
    with reference_path.open("rb") as handle:
        baseline = pickle.load(handle)
    seconds = None
    if args.encode:
        spark = start_spark()
        try:
            pipe = load_biobert(model_path=ROOT / ".backups/biobert-models/sent_biobert_pubmed_base_cased_masked_mean")
            models = [c.model for c in pipe.components if c.model.__class__.__name__ == "BertSentenceEmbeddings"]
            assert len(models) == 1
            models[0].setBatchSize(args.batch_size)
            pipe.predict(table.Definition.iloc[:32].to_list(), output_level="document")
            started = time.perf_counter()
            candidate = pipe.predict(table.Definition, output_level="document")
            seconds = time.perf_counter() - started
        finally:
            spark.stop()
    else:
        path = args.candidate_library
        if path is None:
            run = json.loads((ROOT / "resources/checks/embedding-masked-mean-fix.json").read_text())
            path = ROOT / run["unpublished_candidate_library"]
        with path.open("rb") as handle:
            candidate = pickle.load(handle)
    normalize = lambda value: " ".join(str(value).split())
    expected = table.Definition.map(normalize).to_list()
    assert candidate.document.map(normalize).to_list() == expected == baseline.document.map(normalize).to_list()
    actual = np.stack(candidate[embedding_column(candidate)].to_list()).astype(float)
    reference = np.stack(baseline[embedding_column(baseline)].to_list()).astype(float)
    assert actual.shape == reference.shape == (len(table), 768)
    assert np.isfinite(actual).all() and np.linalg.norm(actual, axis=1).min() > 0
    errors = np.max(np.abs(actual - reference), axis=1)
    cosine = np.sum(actual * reference, axis=1) / (
        np.linalg.norm(actual, axis=1) * np.linalg.norm(reference, axis=1))
    rejected = np.flatnonzero((errors > 1e-5) | (cosine < .999999))
    report = {"status": "passed" if len(rejected) == 0 else "failed",
              "checked_utc": datetime.now(timezone.utc).isoformat(), "rows": len(table), "dimension": 768,
              "batch_size": args.batch_size, "fresh_encoding": args.encode, "encoding_seconds": seconds,
              "row_alignment": "all rows passed", "max_abs": float(errors.max()),
              "min_cosine": float(cosine.min()), "rejected_rows": len(rejected),
              "thresholds": {"max_abs": 1e-5, "min_cosine": .999999},
              "failures": [{"index": int(i), "GO": str(table.iloc[i].GO), "max_abs": float(errors[i]),
                            "cosine": float(cosine[i]), "text": str(table.iloc[i].Definition)} for i in rejected]}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if len(rejected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
