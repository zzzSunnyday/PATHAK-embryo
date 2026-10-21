"""Reject silent CPU fallback and verify the isolated worker before timing."""
import subprocess,os,json,sys,traceback,time
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent
try:
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='8',OPENBLAS_NUM_THREADS='1')
    libs=[str(p) for p in (W/'gpu-env/lib/python3.12/site-packages/nvidia').glob('*/lib')]
    env['LD_LIBRARY_PATH']=':'.join(['/lib/x86_64-linux-gnu','/usr/lib/x86_64-linux-gnu']+libs)
    env['PATHAK_GPU_DEBUG']='1'
    p=subprocess.Popen([str(W/'gpu-env/bin/python'),'-X','faulthandler','-u',str(W/'gpu_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(W/'gpu-preflight-worker.log').open('ab',buffering=0),text=True,env=env)
    def read():
        line=p.stdout.readline()
        if not line:raise RuntimeError('worker exit '+str(p.wait(timeout=5))+': '+(W/'gpu-preflight-worker.log').read_text(errors='replace')[-5000:])
        d=json.loads(line)
        if 'error' in d:raise RuntimeError(d['error'])
        return d
    info=read();ids=json.loads((W/'go-tokens.json').read_text())['input_ids'][:16]
    request={'command':'encode','input_ids':ids,'batch_size':16,'grouping':'sorted','filename':str(W/'gpu-preflight.bin')}
    p.stdin.write(json.dumps(request)+'\n');p.stdin.flush();measurement=read()
    v=np.fromfile(W/'gpu-preflight.bin',dtype='<f4').reshape(-1,768)
    assert v.shape==(16,768) and np.isfinite(v).all()
    from evaluate_retrieval import numerical
    check=numerical(v,np.load(W/'go-baseline.npy')[:16])
    p.stdin.write(json.dumps({'command':'close'})+'\n');p.stdin.flush();read();p.wait(timeout=30)
    result={'status':'passed','worker':info,'measurement':measurement,'numeric':check,'note':'Passing preflight verifies hardware execution and shape, not formal numeric acceptance.'}
    (W/'gpu-preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
except Exception as e:
    (W/'gpu-preflight.json').write_text(json.dumps({'status':'failed','error':repr(e)},indent=2)+'\n');traceback.print_exc();raise
finally:
    if 'p' in globals() and p.poll() is None:p.terminate();p.wait(timeout=30)
