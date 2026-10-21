"""Bounded profiling of accepted BioBERT: frozen corpora, pure TF and NLU."""
import os,sys,json,time,hashlib,threading,collections,traceback,subprocess
from pathlib import Path
from datetime import datetime,timezone

W=Path(__file__).resolve().parent
R=W.parents[1]
sys.path.insert(0,str(R))
os.environ.update(OMP_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')

def dump(p,data):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)

def build_corpus():
    import pandas as pd,numpy as np
    import xml.etree.ElementTree as ET
    import pathak_runtime
    pathak_runtime.configure()
    from nltk.tokenize import sent_tokenize
    go=pd.read_csv(R/'tutorials/GO_BP_Ontology_corups_Filtered.tsv',sep='\t')
    order=np.argsort(go.Definition.astype(str).str.len().values,kind='stable')
    ids=sorted(set(order[np.linspace(0,len(order)-1,512,dtype=int)].tolist()+[388,647,1365,4564]))
    rng=np.random.RandomState(20261022);rng.shuffle(ids)
    rows=[{'id':'go:%d'%i,'kind':'go','source':'tutorials/GO_BP_Ontology_corups_Filtered.tsv','row':int(i),'go':go.iloc[i].GO,'text':' '.join(str(go.iloc[i].Definition).split())} for i in ids]
    papers=[];seen=set();sources={}
    for xml in sorted((R/'tutorials/Example_data').glob('*.xml')):
        sources[str(xml.relative_to(R))]=hashlib.sha256(xml.read_bytes()).hexdigest()
        root=ET.fromstring(xml.read_bytes())
        for area in root.findall('.//abstract')+root.findall('.//body'):
            for pi,p in enumerate(area.iter('p')):
                for si,sentence in enumerate(sent_tokenize(''.join(p.itertext()))):
                    text=' '.join(sentence.split())
                    if len(text)<25 or text in seen:continue
                    seen.add(text);papers.append({'id':'%s:%s:%d:%d'%(xml.stem,area.tag,pi,si),'kind':'literature','source':str(xml.relative_to(R)),'text':text})
    # All unique body/abstract sentences of the two already-local papers.
    rng.shuffle(papers);rows+=papers
    (W/'corpus.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    dump(W/'corpus-manifest.json',{'seed':20261022,'go_rows':len(ids),'literature_rows':len(papers),'sources_sha256':sources,'corpus_sha256':hashlib.sha256((W/'corpus.jsonl').read_bytes()).hexdigest(),'selection':'GO length-stratified 512 plus four known sensitive rows; all unique sentences of local abstract/body paragraphs; no gene filter','thresholds':{'max_abs':1e-5,'min_cosine':.999999},'rounds':3,'threads':{'intra':2,'inter':8,'OMP':2,'Spark_tasks':2},'measurement':'Precomputed resident input tensors for pure TF; full NLU predict to aligned numpy for end-to-end; alternating order each round; file writes outside timing'})
    return rows

class Monitor:
    def __init__(self,pid):
        import psutil
        self.p=psutil.Process(pid);self.stop=threading.Event();self.samples=[]
    def snapshot(self):
        out={}
        for path in Path('/proc/%d/task'%self.p.pid).iterdir():
            try:
                fields=(path/'stat').read_text().rsplit(')',1)[1].split()
                out[path.name]={'ticks':int(fields[11])+int(fields[12]),'state':fields[0]}
            except (OSError,IndexError):pass
        return out
    def sample(self):
        while not self.stop.is_set():
            try:
                threads=self.snapshot()
                self.samples.append({'t':time.monotonic(),'cpu':sum(self.p.cpu_times()[:2]),'rss':self.p.memory_info().rss,'states':dict(collections.Counter(v['state'] for v in threads.values()))})
            except Exception:break
            self.stop.wait(.5)
    def __enter__(self):
        self.start=self.snapshot();self.cgroup_start=Path('/sys/fs/cgroup/cpu.stat').read_text();self.wall=time.perf_counter();self.cpu0=sum(self.p.cpu_times()[:2])
        self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start();return self
    def __exit__(self,*unused):
        self.stop.set();self.thread.join();wall=time.perf_counter()-self.wall;cpu=sum(self.p.cpu_times()[:2])-self.cpu0;end=self.snapshot()
        ticks=os.sysconf('SC_CLK_TCK');active=[]
        for tid,v in end.items():
            delta=(v['ticks']-self.start.get(tid,{'ticks':0})['ticks'])/ticks
            if delta>0:active.append({'tid':tid,'cpu_seconds':delta,'fraction_of_wall':delta/wall})
        active.sort(key=lambda x:x['cpu_seconds'],reverse=True)
        waits=collections.Counter()
        for tid in list(end)[:1000]:
            try:waits[(Path('/proc/%d/task'%self.p.pid)/tid/'wchan').read_text().strip()]+=1
            except OSError:pass
        cg=lambda s:dict((k,int(v)) for k,v in (l.split() for l in s.splitlines()))
        a,b=cg(self.cgroup_start),cg(Path('/sys/fs/cgroup/cpu.stat').read_text())
        self.result={'wall_seconds':wall,'jvm_cpu_seconds':cpu,'average_jvm_cores':cpu/wall,'max_rss_bytes':max([s['rss'] for s in self.samples] or [0]),'samples':self.samples,'active_threads':active,'end_wait_channels':dict(waits),'cgroup_cpu_stat_delta':{k:b[k]-a[k] for k in a}}

def main():
    import numpy as np,pandas as pd
    import pathak_runtime as runtime
    runtime.configure()
    rows=build_corpus()
    from pyspark.sql import SparkSession
    spark=(SparkSession.builder.appName('PATHAK-performance-profile').master('local[2]')
      .config('spark.driver.memory','8g').config('spark.driver.maxResultSize','2g')
      .config('spark.serializer','org.apache.spark.serializer.KryoSerializer')
      .config('spark.jars',str(runtime.JAR_PATH)+','+str(W/'profile.jar'))
      .config('spark.ui.enabled','false').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR');pid=spark.sparkContext._gateway.proc.pid
    helper=spark._jvm.pathak.performance.ProfileBioBert
    start=time.perf_counter();pipe=runtime.load_biobert();load=time.perf_counter()-start
    model=next(c.model for c in pipe.components if c.model.__class__.__name__=='BertSentenceEmbeddings');model._transfer_params_to_java()
    meter=helper.meter(spark.sparkContext._jsc.sc())
    results={'status':'running','jvm_pid':pid,'load_seconds':load,'records':[],'plans':{},'traces':{}}
    def compare(a,b):
        a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
        cos=np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
        d=np.max(np.abs(a-b),axis=1)
        return {'max_abs':float(d.max()),'min_cosine':float(cos.min()),'violating_rows':int(((d>1e-5)|(cos<.999999)).sum()),'exact':bool(np.array_equal(a,b))}
    def nlu_predict(texts,instrument=False):
        spans={}
        if instrument:
            from nlu.pipe.utils.data_conversion_utils import DataConversionUtils as D
            original_convert=D.to_spark_df;original_py=pipe.pythonify_spark_dataframe
            def conversion(*a,**k):
                t=time.perf_counter();v=original_convert(*a,**k);spans['input_conversion_seconds']=time.perf_counter()-t;return v
            def pythonify(*a,**k):
                t=time.perf_counter();v=original_py(*a,**k);spans['compute_collect_and_pythonify_seconds']=time.perf_counter()-t;return v
            D.to_spark_df=staticmethod(conversion);pipe.pythonify_spark_dataframe=pythonify
        try:
            t=time.perf_counter();frame=pipe.predict(texts,output_level='document');prediction=time.perf_counter()-t
            assert frame.document.map(lambda x:' '.join(str(x).split())).tolist()==texts
            t=time.perf_counter();v=np.stack(frame[runtime.embedding_column(frame)].tolist());spans['vector_stack_seconds']=time.perf_counter()-t;spans['predict_seconds']=prediction
            return v,spans
        finally:
            if instrument:D.to_spark_df=staticmethod(original_convert);pipe.pythonify_spark_dataframe=original_py
    try:
        # Warm all layers, sessions, Spark workers and output conversion separately.
        nlu_predict([r['text'] for r in rows[:32]])
        for kind in ['go','literature']:
            selected=[r for r in rows if r['kind']==kind];texts=[r['text'] for r in selected]
            engine=helper.prepare(model._java_obj,texts,16,'input')
            results['plans'][kind]=json.loads(engine.describe());engine.exportInputs(str(W/(kind+'-tokens.json')))
            engine.run('');nlu_predict(texts)
            reference=None
            for round_no in range(3):
                for mode in (['pure_tf','nlu'] if round_no%2==0 else ['nlu','pure_tf']):
                    meter.reset()
                    with Monitor(pid) as mon:
                        t=time.perf_counter()
                        if mode=='pure_tf':
                            detail=json.loads(engine.run(str(W/(kind+'-pure.bin'))));elapsed=time.perf_counter()-t
                            vectors=np.fromfile(W/(kind+'-pure.bin'),dtype='<f4').reshape(-1,768)
                        else:
                            vectors,detail=nlu_predict(texts,instrument=True);elapsed=time.perf_counter()-t
                    # Drain already-finished listener events, outside elapsed time.
                    spark.sparkContext._jsc.sc().listenerBus().waitUntilEmpty(10000)
                    if reference is None:reference=vectors.copy()
                    check=compare(vectors,reference);assert not check['violating_rows'],check
                    if mode=='nlu':np.save(W/(kind+'-baseline.npy'),vectors)
                    record={'kind':kind,'mode':mode,'round':round_no,'rows':len(texts),'wall_seconds':detail['wall_seconds'] if mode=='pure_tf' else elapsed,'rows_per_second':len(texts)/(detail['wall_seconds'] if mode=='pure_tf' else elapsed),'detail':detail,'monitor':mon.result,'spark_tasks':json.loads(meter.report()),'compatibility':check}
                    results['records'].append(record);dump(W/'profile.json',results)
                    print(json.dumps({k:record[k] for k in ['kind','mode','round','rows','wall_seconds','rows_per_second','compatibility']}),flush=True)
            results['traces'][kind]=json.loads(engine.trace(str(W/(kind+'-trace'))));engine.close()
        results['status']='passed';dump(W/'profile.json',results)
    finally:spark.stop()

if __name__=='__main__':
    try:
        dump(W/'profile-status.json',{'status':'running','started_utc':datetime.now(timezone.utc).isoformat()});main()
        dump(W/'profile-status.json',{'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        dump(W/'profile-status.json',{'status':'failed','error':repr(e)});traceback.print_exc();raise
