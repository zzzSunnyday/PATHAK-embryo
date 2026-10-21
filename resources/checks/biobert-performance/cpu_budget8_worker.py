"""One bounded full-thread-budget candidate: OMP/TF/map intra=8, inter=8."""
import os,sys,json,time,zipfile,shutil,subprocess,traceback,hashlib,argparse,pickle
from pathlib import Path
W=Path(__file__).resolve().parent;R=W.parents[1];sys.path.insert(0,str(R))
parser=argparse.ArgumentParser();parser.add_argument('--map-intra',type=int,choices=[2,8],default=8);parser.add_argument('--omp',type=int,choices=[2,8],default=8);args=parser.parse_args()
os.environ.update(OMP_NUM_THREADS=str(args.omp),TF_NUM_INTRAOP_THREADS='8',TF_NUM_INTEROP_THREADS='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS=str(args.omp))
import numpy as np
import pathak_runtime as rt
rt.configure()
from pyspark.sql import SparkSession
import nlu

def send(value):print(json.dumps(value,separators=(',',':')),flush=True)

target=W/('candidate-cpu-budget8-model' if (args.map_intra,args.omp)==(8,8) else 'candidate-cpu-global8-map%d-omp%d-model'%(args.map_intra,args.omp))
if not target.exists():
    shutil.copytree(rt.BIOBERT_PATH,target,copy_function=os.link)
    archive=target/'bert_sentence_tensorflow'
    with zipfile.ZipFile(archive) as z:(W/'budget8-source.pb').write_bytes(z.read('saved_model.pb'))
    cp=str(W/'profile-v3.jar')+':'+str(rt.JAR_PATH)
    graphfile=target/'experiment-graph.pb'
    if args.map_intra==2:graphfile.write_bytes((W/'budget8-source.pb').read_bytes())
    else:
        subprocess.run([str(rt.ENV_PREFIX/'bin/java'),'-cp',cp,'SetMapParallelism',str(W/'budget8-source.pb'),str(graphfile),str(args.map_intra)],check=True,stdout=sys.stderr)
        temporary=target/'bert_sentence_tensorflow.tmp'
        with zipfile.ZipFile(archive) as source,zipfile.ZipFile(temporary,'w') as out:
            for info in source.infolist():out.writestr(info,graphfile.read_bytes() if info.filename=='saved_model.pb' else source.read(info.filename))
        temporary.replace(archive);(target/'.bert_sentence_tensorflow.crc').unlink(missing_ok=True)
    meta=target/'metadata/part-00000';value=json.loads(meta.read_text());meta.unlink()
    value['paramMap']['configProtoBytes']=[16,8,40,8];value['paramMap']['batchSize']=16;meta.write_text(json.dumps(value,separators=(',',':'))+'\n')
    (target/'metadata/.part-00000.crc').unlink(missing_ok=True)
    marker=target/'pathak_pooling.json';value=json.loads(marker.read_text());marker.unlink();value.update(status='isolated_performance_experiment',tf_config_proto_bytes=[16,8,40,8],sentence_max_intra_threads=args.map_intra,graph_sha256=hashlib.sha256(graphfile.read_bytes()).hexdigest(),model_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest());marker.write_text(json.dumps(value,indent=2)+'\n')
spark=(SparkSession.builder.appName('PATHAK-budget8-worker').master('local[2]').config('spark.driver.memory','8g')
 .config('spark.serializer','org.apache.spark.serializer.KryoSerializer').config('spark.driver.maxResultSize','2g')
 .config('spark.jars',str(rt.JAR_PATH)+','+str(W/'profile-v3.jar')).config('spark.ui.enabled','false').getOrCreate())
spark.sparkContext.setLogLevel('ERROR');helper=spark._jvm.pathak.performance.ProfileBioBert
t=time.perf_counter();pipe=nlu.load(rt.BIOBERT_REFERENCE,path=str(target));load=time.perf_counter()-t
model=next(c.model for c in pipe.components if c.model.__class__.__name__=='BertSentenceEmbeddings');model.setBatchSize(16);model._transfer_params_to_java()
b=model._java_obj.getModelIfNotSet();f=b.getClass().getDeclaredField('configProtoBytes');f.setAccessible(True);assert list(f.get(b).get())==[16,8,40,8]
with zipfile.ZipFile(target/'bert_sentence_tensorflow') as z:graph_sha=hashlib.sha256(z.read('saved_model.pb')).hexdigest()
send({'ready':True,'jvm_pid':spark.sparkContext._gateway.proc.pid,'load_seconds':load,'model':str(target),'OMP_NUM_THREADS':args.omp,'intra':8,'inter':8,'map_intra':args.map_intra,'archive_sha256':hashlib.sha256((target/'bert_sentence_tensorflow').read_bytes()).hexdigest(),'graph_sha256':graph_sha})
engine=None
try:
    for line in sys.stdin:
        try:
            message=json.loads(line);command=message['command']
            if command=='close':send({'closed':True});break
            if command=='prepare':
                if engine is not None:engine.close()
                engine=helper.prepare(model._java_obj,message['texts'],16,'input');send(json.loads(engine.describe()));continue
            if command=='pure':send(json.loads(engine.runConcurrent(message.get('filename',''),2)));continue
            if command=='encode':
                model.setBatchSize(message.get('batch_size',16));values=message['texts']
                if message.get('input_type')=='series':
                    import pandas as pd
                    values=pd.Series(values,index=np.arange(len(values))*7+3)
                elif message.get('input_type')=='string':values=values[0]
                t=time.perf_counter();frame=pipe.predict(values,output_level='document')
                assert frame.document.map(lambda s:' '.join(str(s).split())).tolist()==message['texts']
                vectors=np.stack(frame[rt.embedding_column(frame)].tolist());elapsed=time.perf_counter()-t
                if message.get('filename'):vectors.astype('<f4').tofile(message['filename'])
                if message.get('pickle_path'):
                    with Path(message['pickle_path']).open('wb') as handle:pickle.dump(frame,handle,protocol=4)
                if message.get('pipeline_path'):
                    import contextlib
                    with contextlib.redirect_stdout(sys.stderr):pipe.save(message['pipeline_path'])
                send({'rows':len(vectors),'wall_seconds':elapsed,'rows_per_second':len(vectors)/elapsed});continue
            raise ValueError(command)
        except Exception as e:traceback.print_exc(file=sys.stderr);send({'error':repr(e)})
finally:
    if engine is not None:engine.close()
    spark.stop()
