"""New-process validation of raw/global8 and staged saved pipeline."""
import os,sys,json,time,pickle,traceback
from pathlib import Path
W=Path(__file__).resolve().parent;R=W.parents[1];sys.path.insert(0,str(R))
os.environ.update(OMP_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='8',TF_NUM_INTEROP_THREADS='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
from run_profile_v2 import dump
from evaluate_retrieval import numerical
def main():
    import numpy as np,pandas as pd
    import pathak_runtime as rt
    rt.configure()
    import nlu
    spark=rt.start_spark();reference=np.load(W/'global8-validation-reference.npz');texts=reference['texts'].tolist();expected=reference['vectors'];checks=[]
    def check(name,actual,ref):
        c=numerical(actual,ref);checks.append(dict(case=name,**c));assert c['violating_rows']==0,(name,c)
    try:
        for name,path in [('raw',W/'candidate-cpu-global8-map2-omp2-model'),('saved',W/'staged_global8/pipe_full_BP')]:
            pipe=nlu.load(rt.BIOBERT_REFERENCE,path=str(path)) if name=='raw' else nlu.load(path=str(path))
            model=next(c.model for c in pipe.components if c.model.__class__.__name__=='BertSentenceEmbeddings')
            assert model.getOrDefault('caseSensitive') is True and model.getOrDefault('maxSentenceLength')==128
            def encode(values,batch=16):
                model.setBatchSize(batch);frame=pipe.predict(values,output_level='document');wanted=[values] if isinstance(values,str) else list(values)
                assert frame.document.tolist()==wanted;assert model._java_obj.getBatchSize()==batch
                b=model._java_obj.getModelIfNotSet();f=b.getClass().getDeclaredField('configProtoBytes');f.setAccessible(True);assert list(f.get(b).get())==[16,8,40,8]
                return np.stack(frame[rt.embedding_column(frame)].tolist())
            for batch in [1,8,16]:check(name+'_batch%d'%batch,encode(texts,batch),expected)
            check(name+'_actual_single',encode(texts[0]),expected[:1])
            check(name+'_series',encode(pd.Series(texts,index=np.arange(len(texts))*7+3)),expected)
            check(name+'_reverse',encode(texts[::-1]),expected[::-1])
            check(name+'_repeat17',encode([texts[0]]*17),np.repeat(expected[:1],17,axis=0))
            actual=np.concatenate([encode(texts[:1]),encode(texts[1:6]),encode(texts[6:])]);check(name+'_call_chunks',actual,expected)
        bank0=np.load(W/'go-bank-baseline.npy')
        with (W/'staged_global8/predictions_full_BP.pickle').open('rb') as f:library=pickle.load(f)
        table=pd.read_csv(R/'tutorials/GO_BP_Ontology_corups_Filtered.tsv',sep='\t')
        assert library.document.map(lambda s:' '.join(str(s).split())).tolist()==table.Definition.map(lambda s:' '.join(str(s).split())).tolist()
        check('all_staged_vectors',np.stack(library[rt.embedding_column(library)].tolist()),bank0)
        result={'status':'passed','cases':checks,'case_count':len(checks),'maximum_absolute_error':max(x['max_abs'] for x in checks),'minimum_cosine':min(x['min_cosine'] for x in checks),'actual_captured_config':[16,8,40,8],'actual_java_batch_sizes_checked':[1,8,16],'reference':'Original export batch1 in independent CPU2 baseline process','rows':5117}
        dump(W/'global8-staged-fresh-validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
    finally:spark.stop()
if __name__=='__main__':
    try:dump(W/'global8-fresh-status.json',{'status':'running'});main();dump(W/'global8-fresh-status.json',{'status':'passed'})
    except Exception as ex:dump(W/'global8-fresh-status.json',{'status':'failed','error':repr(ex)});traceback.print_exc();raise
