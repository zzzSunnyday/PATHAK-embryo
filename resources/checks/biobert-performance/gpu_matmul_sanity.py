"""Diagnostic CUDA FP32 MatMul; this is not a text-encoding benchmark."""
import os,json,time,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.environ.update(CUDA_VISIBLE_DEVICES='0',NVIDIA_TF32_OVERRIDE='0',TF_ENABLE_ONEDNN_OPTS='0')
import numpy as np,tensorflow as tf
tf.compat.v1.disable_eager_execution();tf.config.experimental.enable_tensor_float_32_execution(False)
g=tf.Graph()
with g.as_default(),tf.device('/GPU:0'):
    x=tf.compat.v1.placeholder(tf.float32,[None,768],name='x');w=tf.compat.v1.placeholder(tf.float32,[768,3072],name='w')
    y=tf.reduce_mean(tf.matmul(x,w,name='diagnostic_matmul'))
c=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=8,allow_soft_placement=False)
c.gpu_options.allow_growth=True;c.graph_options.rewrite_options.disable_meta_optimizer=True
with tf.compat.v1.Session(graph=g,config=c) as session:
    feed={x:np.ones((800,768),dtype=np.float32),w:np.ones((768,3072),dtype=np.float32)}
    for _ in range(2):value=session.run(y,feed_dict=feed)
    meta=tf.compat.v1.RunMetadata();t=time.perf_counter();value=session.run(y,feed_dict=feed,options=tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE),run_metadata=meta);seconds=time.perf_counter()-t
    assert float(value)==768.0
    devices={d.device:len(d.node_stats) for d in meta.step_stats.dev_stats};assert any('GPU:0' in k and v for k,v in devices.items())
    result={'status':'passed','tensorflow':tf.__version__,'value':float(value),'shape':[800,768,3072],'devices':devices,'traced_seconds':seconds,'tf32_enabled':tf.config.experimental.tensor_float_32_execution_enabled(),'scope':'Hardware MatMul availability only; no claim about BioBERT encoding speed.'}
    print(json.dumps(result),flush=True)
