"""Fresh-process check of the published parallel BioBERT model and GO library."""
import argparse,json,pickle,sys
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'resources/checks/embedding-parallel-postcheck.json')
    args=parser.parse_args()
    import numpy as np,pandas as pd
    from pathak_runtime import (start_spark,load_biobert,configure_biobert_pipeline,embedding_column,
        check_embedding_manifest,BIOBERT_PATH,BIOBERT_BATCH_SIZE,BIOBERT_PROTOCOL,BIOBERT_TF_CONFIG)
    tutorial=ROOT/'tutorials';corpus=tutorial/'GO_BP_Ontology_corups_Filtered.tsv'
    manifest=check_embedding_manifest(corpus,tutorial/'predictions_full_BP.pickle',tutorial/'pipe_full_BP')
    table=pd.read_csv(corpus,sep='\t')
    with (tutorial/'predictions_full_BP.pickle').open('rb') as f:library=pickle.load(f)
    marker=json.loads((BIOBERT_PATH/'pathak_pooling.json').read_text())
    baseline_path=ROOT/marker['legacy_reference_pickle']
    with baseline_path.open('rb') as f:legacy=pickle.load(f)
    normalize=lambda value:' '.join(str(value).split())
    expected_texts=table.Definition.map(normalize).tolist()
    assert library.document.map(normalize).tolist()==legacy.document.map(normalize).tolist()==expected_texts
    stored=np.stack(library[embedding_column(library)].tolist())
    baseline=np.stack(legacy[embedding_column(legacy)].tolist())
    checks=[]
    def compare(name,a,b):
        a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
        assert a.shape==b.shape and a.ndim==2 and a.shape[1]==768
        assert np.isfinite(a).all() and np.linalg.norm(a,axis=1).min()>0
        cosine=np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
        result={'case':name,'rows':len(a),'max_abs':float(np.abs(a-b).max()),'min_cosine':float(cosine.min()),'exact':bool(np.array_equal(a,b))}
        checks.append(result);print(json.dumps(result),flush=True)
        assert result['max_abs']<=1e-5 and result['min_cosine']>=.999999,result
    compare('all_published_vectors_vs_legacy_single',stored,baseline)
    ids=[0,1,7,15,101,388,647,1365,2558,4564,5116]
    texts=table.iloc[ids].Definition.astype(str).tolist();reference=baseline[ids]
    spark=start_spark()
    def verify_config(pipe):
        models=[c.model for c in pipe.components if c.model.__class__.__name__=='BertSentenceEmbeddings']
        assert len(models)==1
        model=models[0]
        assert model.getOrDefault('batchSize')==BIOBERT_BATCH_SIZE
        assert model.getOrDefault('caseSensitive') is True and model.getOrDefault('maxSentenceLength')==128
        model._transfer_params_to_java()
        backend=model._java_obj.getModelIfNotSet()
        field=backend.getClass().getDeclaredField('configProtoBytes');field.setAccessible(True)
        captured=field.get(backend);assert captured.isDefined()
        assert list(captured.get())==BIOBERT_TF_CONFIG
    def encode(pipe,values):
        expected=[values] if isinstance(values,str) else list(values)
        frame=pipe.predict(values,output_level='document')
        assert frame.document.map(normalize).tolist()==list(map(normalize,expected))
        return np.stack(frame[embedding_column(frame)].tolist())
    try:
        import nlu
        raw=load_biobert();verify_config(raw)
        compare('raw_default_batch_vs_legacy',encode(raw,texts),reference)
        for index in [0,5,6,9]:compare('raw_actual_single_%d'%ids[index],encode(raw,texts[index]),reference[index:index+1])
        saved=configure_biobert_pipeline(nlu.load(path=str(tutorial/'pipe_full_BP')));verify_config(saved)
        compare('saved_default_batch_vs_legacy',encode(saved,texts),reference)
        compare('saved_series_nondefault_index',encode(saved,pd.Series(texts,index=np.arange(len(texts))*7+3)),reference)
        compare('saved_reversed_order',encode(saved,texts[::-1])[::-1],reference)
        compare('saved_sensitive_repeated17',encode(saved,[texts[9]]*17),np.repeat(reference[9:10],17,axis=0))
        chunks=np.concatenate([encode(saved,texts[:1]),encode(saved,texts[1:6]),encode(saved,texts[6:])])
        compare('saved_call_chunks_1_5_remainder',chunks,reference)
        long_texts=[('cell '*count).strip() for count in [124,125,126,127,600]]
        # A legacy model loaded in this CPU8 JVM would inherit its global pool.
        # Freeze the actual original-export B1 / CPU2 reference in its own run.
        import hashlib
        reference_file=ROOT/'resources/checks/embedding-original-single-cpu2-reference.npz'
        reference_info=json.loads(reference_file.with_suffix('.json').read_text())
        assert hashlib.sha256(reference_file.read_bytes()).hexdigest()==reference_info['npz_sha256']
        source=ROOT/'.backups/biobert-models/sent_biobert_pubmed_base_cased/bert_sentence_tensorflow'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==reference_info['source_archive_sha256']
        with np.load(reference_file) as frozen:
            by_text=dict(zip(frozen['texts'].tolist(),frozen['vectors']))
            long_reference=np.stack([by_text[text] for text in long_texts])
        compare('raw_truncation_boundary_vs_original_single',encode(raw,long_texts),long_reference)
        compare('saved_truncation_boundary_vs_original_single',encode(saved,long_texts),long_reference)
        report={'status':'passed','checked_utc':datetime.now(timezone.utc).isoformat(),'protocol':BIOBERT_PROTOCOL,
            'batch_size':BIOBERT_BATCH_SIZE,'tf_config_proto_bytes':BIOBERT_TF_CONFIG,
            'case_count':len(checks),'thresholds':{'max_abs':1e-5,'min_cosine':.999999},
            'maximum_absolute_error':max(c['max_abs'] for c in checks),'minimum_cosine':min(c['min_cosine'] for c in checks),
            'comparisons':checks,'manifest_verified':True,'legacy_reference_pickle':str(baseline_path.relative_to(ROOT))}
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':'passed','report':str(args.report),'maximum_absolute_error':report['maximum_absolute_error']}),flush=True)
    finally:spark.stop()

if __name__=='__main__':main()
