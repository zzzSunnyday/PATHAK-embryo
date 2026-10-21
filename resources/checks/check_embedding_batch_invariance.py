"""Check the published PATHAK vector library in a fresh process.

Run with the pathak Python environment. By default this check only reads files;
--report optionally saves the measured results as JSON.
"""

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    from pathak_runtime import configure_biobert_pipeline, embedding_column, start_spark, BIOBERT_PROTOCOL, check_embedding_manifest

    spark = start_spark()
    try:
        import nlu

        table = pd.read_csv(ROOT / 'tutorials/GO_BP_Ontology_corups_Filtered.tsv', sep='\t')
        with (ROOT / 'tutorials/predictions_full_BP.pickle').open('rb') as handle:
            saved = pickle.load(handle)
        stored_vectors = np.stack(saved[embedding_column(saved)].to_list())
        assert stored_vectors.shape == (len(table), 768)
        assert np.isfinite(stored_vectors).all()
        pipeline = nlu.load(path=str(ROOT / 'tutorials/pipe_full_BP'))
        saved_batch_sizes = [c.model.getOrDefault('batchSize') for c in pipeline.components
                             if c.model.__class__.__name__ == 'BertSentenceEmbeddings']
        manifest = check_embedding_manifest(
            ROOT / 'tutorials/GO_BP_Ontology_corups_Filtered.tsv',
            ROOT / 'tutorials/predictions_full_BP.pickle', ROOT / 'tutorials/pipe_full_BP')
        assert manifest['protocol'] == BIOBERT_PROTOCOL
        assert saved_batch_sizes == [manifest['internal_batch_size']], saved_batch_sizes
        pipeline = configure_biobert_pipeline(pipeline)

        def encode(texts):
            result = pipeline.predict(texts, output_level='document')
            return np.stack(result[embedding_column(result)].to_list())

        indices = sorted(set([0, 1, 7, 15, 101, 388, 647, 1365, 4564, len(table) // 2, len(table) - 1]))
        texts = [str(table.iloc[index].Definition) for index in indices]
        single = encode([texts[0]])[0]
        mixed = encode(texts)
        series = encode(pd.Series(texts))
        reversed_vectors = encode(list(reversed(texts)))[::-1]
        repeated = encode([texts[0]] * 17)
        mixed_long = encode([texts[0], 'The development of neural cells. ' * 100])[0]
        errors = {
            'list_vs_series': float(np.max(np.abs(mixed - series))),
            'order_invariance': float(np.max(np.abs(mixed - reversed_vectors))),
            'single_vs_mixed': float(np.max(np.abs(single - mixed[0]))),
            'single_vs_repeated': float(np.max(np.abs(single - repeated))),
            'single_vs_long_context': float(np.max(np.abs(single - mixed_long))),
            'fresh_process_vs_saved': float(np.max(np.abs(mixed - stored_vectors[indices]))),
        }
        assert max(errors.values()) <= 1e-5, errors
        self_cosines = np.sum(mixed * stored_vectors[indices], axis=1) / (
            np.linalg.norm(mixed, axis=1) * np.linalg.norm(stored_vectors[indices], axis=1)
        )
        assert float(self_cosines.min()) >= 0.999999
        report = {
            'status': 'passed',
            'checked_utc': datetime.now(timezone.utc).isoformat(),
            'rows': len(table),
            'dimension': 768,
            'protocol': BIOBERT_PROTOCOL,
            'saved_internal_batch_sizes': saved_batch_sizes,
            'sample_indices': indices,
            'max_absolute_errors': errors,
            'self_query_cosines': self_cosines.tolist(),
            'maximum_absolute_error': max(errors.values()),
        }
        if args.report:
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
