"""Isolated TF 2.16 FP32 worker. Inputs use the unchanged Java tokenizer."""
import os,sys,json,time,zipfile,hashlib,traceback,argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','gpu'],default='gpu');args=parser.parse_args()
W=Path(__file__).resolve().parent;R=W.parents[1]
os.environ['CUDA_VISIBLE_DEVICES']='0' if args.device=='gpu' else ''
os.environ['NVIDIA_TF32_OVERRIDE']='0';os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.config.experimental.enable_tensor_float_32_execution(False)

def send(d):print(json.dumps(d,separators=(',',':')),flush=True)

archive=R/'.backups/biobert-models/sent_biobert_pubmed_base_cased_masked_mean/bert_sentence_tensorflow'
checkpoint=W/'gpu-checkpoint';checkpoint.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    graphdef=tf.compat.v1.GraphDef();graphdef.ParseFromString(z.read('saved_model.pb'))
    for n in ['variables.index','variables.data-00000-of-00001']:
        p=checkpoint/n
        if not p.exists():p.write_bytes(z.read(n))
    source_sha=hashlib.sha256(z.read('saved_model.pb')).hexdigest()
for n in graphdef.node:n.device=''
graph=tf.Graph()
with graph.as_default(),tf.device('/GPU:0' if args.device=='gpu' else '/CPU:0'):
    tf.import_graph_def(graphdef,name='')
config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=8,allow_soft_placement=True)
config.graph_options.rewrite_options.disable_meta_optimizer=True
config.gpu_options.allow_growth=True
if args.device=='cpu':config.device_count['GPU']=0
start=time.perf_counter();session=tf.compat.v1.Session(graph=graph,config=config)
session.run(graph.get_operation_by_name('save/restore_all'),feed_dict={'save/Const:0':str(checkpoint/'variables')})
input_tensors=[graph.get_tensor_by_name(n+':0') for n in ['input_ids','input_mask','segment_ids']]
output=graph.get_tensor_by_name('pooled_output:0')
devices=[{'name':d.name,'type':d.device_type,'memory_limit_bytes':getattr(d,'memory_limit_bytes',None)} for d in session.list_devices()]
if args.device=='gpu' and not any(d['type']=='GPU' for d in devices):raise RuntimeError('GPU requested but no GPU is visible; CPU fallback is rejected')
send({'ready':True,'tensorflow_version':tf.__version__,'device':args.device,'devices':devices,'load_restore_seconds':time.perf_counter()-start,'graph_sha256':source_sha,'tf32_enabled':tf.config.experimental.tensor_float_32_execution_enabled(),'build_info':tf.sysconfig.get_build_info()})
cached=None;cached_cpu=None;cpu_session=None;cpu_inputs=None;cpu_output=None
def prepare(ids,batch_size,grouping,handles=None):
    handles=input_tensors if handles is None else handles
    order=np.arange(len(ids));lengths=np.array([len(x) for x in ids])
    if grouping!='input':order=np.argsort(lengths,kind='stable')
    batches=[]
    for i in range(0,len(ids),batch_size):
        positions=order[i:i+batch_size];length=int(lengths[positions].max());values=np.zeros((len(positions),length),dtype=np.int32)
        for j,p in enumerate(positions):values[j,:len(ids[p])]=ids[p]
        batches.append((positions,{handles[0]:values,handles[1]:(values!=0).astype(np.int32),handles[2]:np.zeros_like(values)}))
    return len(ids),batches
def infer(plan,filename='',trace_prefix=None,fetch=None,backend_session=None,backend_output=None):
    selected_session=session if backend_session is None else backend_session
    selected_output=output if backend_output is None else backend_output
    count,batches=plan;vectors=np.empty((count,768),dtype=np.float32);runner=0;cpu=time.process_time();start=time.perf_counter();trace_devices={}
    for bi,(positions,feed) in enumerate(batches):
        t=time.perf_counter()
        if os.environ.get('PATHAK_GPU_DEBUG')=='1':print('DEBUG before Session.run batch='+str(bi)+' shape='+str(next(iter(feed.values())).shape),file=sys.stderr,flush=True)
        if trace_prefix is not None and bi==0:
            metadata=tf.compat.v1.RunMetadata();options=tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            values=selected_session.run(selected_output if fetch is None else fetch,feed_dict=feed,options=options,run_metadata=metadata)
            Path(trace_prefix+'.pb').write_bytes(metadata.SerializeToString())
            for d in metadata.step_stats.dev_stats:
                trace_devices[d.device]={'nodes':len(d.node_stats),'summed_op_us':sum(n.op_end_rel_micros-n.op_start_rel_micros for n in d.node_stats)}
        else:values=selected_session.run(selected_output if fetch is None else fetch,feed_dict=feed)
        if os.environ.get('PATHAK_GPU_DEBUG')=='1':print('DEBUG after Session.run '+str(values.shape),file=sys.stderr,flush=True)
        runner+=time.perf_counter()-t;vectors[positions]=values
    wall=time.perf_counter()-start;cpu=time.process_time()-cpu
    result={'rows':count,'wall_seconds':wall,'tf_runner_seconds':runner,'rows_per_second':count/wall,'cpu_seconds':cpu,'average_cpu_cores':cpu/wall,'trace_devices':trace_devices}
    if filename:vectors.astype('<f4',copy=False).tofile(filename)
    return result
try:
    for line in sys.stdin:
        try:
            message=json.loads(line);cmd=message['command']
            if cmd=='close':send({'closed':True});break
            if cmd=='init_cpu_control':
                t=time.perf_counter();cpu_graph=tf.Graph()
                with cpu_graph.as_default(),tf.device('/CPU:0'):tf.import_graph_def(graphdef,name='')
                cc=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=8,allow_soft_placement=False,device_count={'GPU':0})
                cc.graph_options.rewrite_options.disable_meta_optimizer=True
                cpu_session=tf.compat.v1.Session(graph=cpu_graph,config=cc)
                cpu_session.run(cpu_graph.get_operation_by_name('save/restore_all'),feed_dict={'save/Const:0':str(checkpoint/'variables')})
                cpu_inputs=[cpu_graph.get_tensor_by_name(n+':0') for n in ['input_ids','input_mask','segment_ids']];cpu_output=cpu_graph.get_tensor_by_name('pooled_output:0')
                send({'ready':True,'load_seconds':time.perf_counter()-t,'device':'CPU:0','intra':2,'inter':8,'oneDNN':False});continue
            control=message.get('backend')=='cpu_control'
            handles=cpu_inputs if control else input_tensors
            kwargs={'backend_session':cpu_session,'backend_output':cpu_output} if control else {}
            if cmd=='prepare':
                t=time.perf_counter();plan=prepare(message['input_ids'],message.get('batch_size',16),message.get('grouping','sorted'),handles)
                if control:cached_cpu=plan
                else:cached=plan
                send({'prepared':True,'seconds':time.perf_counter()-t});continue
            if cmd=='run':
                send(infer(cached_cpu if control else cached,message.get('filename',''),message.get('trace_prefix'),**kwargs));continue
            if cmd=='encode':
                t=time.perf_counter();plan=prepare(message['input_ids'],message.get('batch_size',16),message.get('grouping','sorted'),handles);prep=time.perf_counter()-t
                fetch=graph.get_tensor_by_name('Mean:0') if message.get('unmasked') else None
                result=infer(plan,message.get('filename',''),fetch=fetch,**kwargs);result['batch_prepare_seconds']=prep;send(result);continue
            raise ValueError(cmd)
        except Exception as e:
            traceback.print_exc(file=sys.stderr);send({'error':repr(e)})
finally:
    session.close()
    if cpu_session is not None:cpu_session.close()
