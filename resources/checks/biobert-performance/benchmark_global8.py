"""Paired CPU2 vs fresh global8/map2/OMP2; strict all-library validation."""
import os,sys,json,time,subprocess,pickle,traceback
from pathlib import Path
from datetime import datetime,timezone
W=Path(__file__).resolve().parent;R=W.parents[1];sys.path.insert(0,str(R))
os.environ.update(OMP_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
from run_profile_v2 import dump,Monitor
from evaluate_retrieval import numerical,retrieval
def main():
    import numpy as np,pandas as pd
    import pathak_runtime as rt
    rt.configure()
    from pyspark.sql import SparkSession
    spark=(SparkSession.builder.appName('PATHAK-global8-benchmark').master('local[2]').config('spark.driver.memory','8g').config('spark.driver.maxResultSize','2g')
      .config('spark.jars',str(rt.JAR_PATH)+','+str(W/'profile-v3.jar')).config('spark.ui.enabled','false').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR');helper=spark._jvm.pathak.performance.ProfileBioBert;pid=spark.sparkContext._gateway.proc.pid
    t=time.perf_counter();pipe=rt.load_biobert();model=next(c.model for c in pipe.components if c.model.__class__.__name__=='BertSentenceEmbeddings');model._transfer_params_to_java();load=time.perf_counter()-t
    child=None;result={'status':'running','records':[],'protocol_cases':{},'full_library':{},'baseline_load_seconds':load,'started_utc':datetime.now(timezone.utc).isoformat()}
    try:
        t=time.perf_counter();child=subprocess.Popen([sys.executable,'-u',str(W/'cpu_budget8_worker.py'),'--map-intra','2','--omp','2'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=(W/'global8-worker.log').open('ab',buffering=0),text=True,bufsize=1)
        def receive():
            line=child.stdout.readline()
            if not line:raise RuntimeError('global8 worker exit '+str(child.wait(timeout=5)))
            d=json.loads(line)
            if 'error' in d:raise RuntimeError(d['error'])
            return d
        def rpc(command,**kwargs):child.stdin.write(json.dumps(dict(command=command,**kwargs),separators=(',',':'))+'\n');child.stdin.flush();return receive()
        result['candidate']=receive();result['candidate']['startup_total_seconds']=time.perf_counter()-t
        rows=[json.loads(l) for l in (W/'corpus.jsonl').read_text().splitlines()]
        def stable(values):
            frame=pipe.predict(values,output_level='document');assert frame.document.tolist()==list(values);return np.stack(frame[rt.embedding_column(frame)].tolist())
        stable([r['text'] for r in rows[:32]]);rpc('encode',texts=[r['text'] for r in rows[:32]])
        for kind in ['go','literature']:
            texts=[r['text'] for r in rows if r['kind']==kind];reference=np.load(W/(kind+'-baseline.npy'))
            engine=helper.prepare(model._java_obj,texts,16,'input');rpc('prepare',texts=texts)
            engine.runConcurrent('',2);rpc('pure');stable(texts[:64]);rpc('encode',texts=texts[:64])
            for round_no in range(3):
                names=['stable','global8_map2'] if round_no%2==0 else ['global8_map2','stable']
                for name in names:
                    for mode in (['pure_tf','end_to_end'] if round_no%2==0 else ['end_to_end','pure_tf']):
                        with Monitor(pid if name=='stable' else result['candidate']['jvm_pid']) as monitor:
                            t=time.perf_counter();detail={}
                            if mode=='pure_tf':
                                f=W/(kind+'-'+name+'-pure.bin');detail=json.loads(engine.runConcurrent(str(f),2)) if name=='stable' else rpc('pure',filename=str(f));seconds=detail['wall_seconds'];vectors=np.fromfile(f,dtype='<f4').reshape(-1,768)
                            else:
                                if name=='stable':vectors=stable(texts)
                                else:
                                    f=W/(kind+'-global8-map2-e2e.bin');detail=rpc('encode',texts=texts,filename=str(f));vectors=np.fromfile(f,dtype='<f4').reshape(-1,768)
                                seconds=time.perf_counter()-t
                        check=numerical(vectors,reference);assert check['violating_rows']==0,check
                        np.save(W/(kind+'-'+name+'-'+mode+'-r%d.npy'%round_no),vectors)
                        if name=='global8_map2':np.save(W/(kind+'-global8-map2.npy'),vectors)
                        r={'kind':kind,'config':name,'mode':mode,'round':round_no,'rows':len(texts),'wall_seconds':seconds,'rows_per_second':len(texts)/seconds,'detail':detail,'monitor':monitor.result,'compatibility':check}
                        result['records'].append(r);dump(W/'global8-benchmark.json',result)
                        print(json.dumps({k:r[k] for k in ['kind','config','mode','round','wall_seconds','rows_per_second','compatibility']}),flush=True)
            engine.close()
        # A real original-export B=1 reference is made in this CPU2 process.
        cases=[r['text'] for r in rows[:24]]+[r['text'] for r in rows if r['kind']=='literature'][:12]+[('cell '*n).strip() for n in [124,125,126,127,600]]
        old=rt.load_biobert(model_path=R/'.backups/biobert-models/sent_biobert_pubmed_base_cased');frame=old.predict(cases,output_level='document');reference=np.stack(frame[rt.embedding_column(frame)].tolist())
        np.savez(W/'global8-validation-reference.npz',texts=np.array(cases),vectors=reference)
        def check_case(name,texts,expected,**kwargs):
            f=W/'global8-protocol-output.bin';rpc('encode',texts=texts,filename=str(f),**kwargs);v=np.fromfile(f,dtype='<f4').reshape(-1,768);c=numerical(v,expected);result['protocol_cases'][name]=c;assert c['violating_rows']==0,(name,c);return v
        for b in [1,8,16]:check_case('batch%d_vs_original_single'%b,cases,reference,batch_size=b)
        check_case('series_nondefault_index',cases,reference,input_type='series');check_case('actual_single_string',[cases[0]],reference[:1],input_type='string')
        check_case('reversed_order',cases[::-1],reference[::-1]);check_case('repeated17',[cases[0]]*17,np.repeat(reference[:1],17,axis=0))
        chunks=np.concatenate([check_case('chunk_first',cases[:1],reference[:1]),check_case('chunk_middle',cases[1:6],reference[1:6]),check_case('chunk_rest',cases[6:],reference[6:])]);result['protocol_cases']['combined_call_chunks']=numerical(chunks,reference)
        dump(W/'global8-benchmark.json',result)
        table=pd.read_csv(R/'tutorials/GO_BP_Ontology_corups_Filtered.tsv',sep='\t');texts=table.Definition.map(lambda s:' '.join(str(s).split())).tolist();bank0=np.load(W/'go-bank-baseline.npy')
        stage=W/'staged_global8';stage.mkdir(exist_ok=True)
        # B16 creates a complete aligned vector/pipeline pair before any promotion.
        for b in [16,8,1]:
            f=W/('go-bank-global8-b%d.bin'%b);options={'pickle_path':str(stage/'predictions_full_BP.pickle'),'pipeline_path':str(stage/'pipe_full_BP')} if b==16 else {}
            t=time.perf_counter();detail=rpc('encode',texts=texts,batch_size=b,filename=str(f),**options);seconds=time.perf_counter()-t
            bank=np.fromfile(f,dtype='<f4').reshape(-1,768);c=numerical(bank,bank0);result['full_library']['batch%d'%b]={'numeric':c,'seconds_including_staging_if_b16':seconds,'rows_aligned':True};assert c['violating_rows']==0,(b,c)
            if b==16:np.save(W/'go-bank-global8-map2.npy',bank)
            dump(W/'global8-benchmark.json',result);print('FULL',b,json.dumps(result['full_library']['batch%d'%b]),flush=True)
        bank=np.load(W/'go-bank-global8-map2.npy');queries=np.load(W/'literature-global8-map2.npy');q0=np.load(W/'literature-original-single.npy')
        result['retrieval']=retrieval(bank0,q0,bank,queries,table.GO.tolist(),[r['id'] for r in rows if r['kind']=='literature'])
        with (stage/'predictions_full_BP.pickle').open('rb') as handle:published=pickle.load(handle)
        assert published.document.tolist()==texts;result['staged_pickle_numeric']=numerical(np.stack(published[rt.embedding_column(published)].tolist()),bank0)
        result['status']='passed';dump(W/'global8-benchmark.json',result);rpc('close');child.wait(timeout=30)
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)
        spark.stop()
if __name__=='__main__':
    try:dump(W/'global8-status.json',{'status':'running'});main();dump(W/'global8-status.json',{'status':'passed'})
    except Exception as ex:dump(W/'global8-status.json',{'status':'failed','error':repr(ex)});traceback.print_exc();raise
