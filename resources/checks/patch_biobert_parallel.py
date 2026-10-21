"""Build an independent graph-native parallel model using existing Java TF.

Run in a fresh pathak Python process. Does not replace an existing target or
publish embeddings. Run the validation and benchmark commands before promotion.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def members(path):
    result={}
    with zipfile.ZipFile(path) as archive:
        for name in ['saved_model.pb','variables.data-00000-of-00001','variables.index']:
            h=hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            result[name]=h.hexdigest()
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'.backups/biobert-models/sent_biobert_pubmed_base_cased_masked_mean')
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,choices=[1,8,16],default=16)
    parser.add_argument('--intra-threads',type=int,default=2)
    parser.add_argument('--inter-threads',type=int,default=8)
    parser.add_argument('--sentence-intra-threads',type=int,default=2)
    args=parser.parse_args()
    source=args.source.expanduser().resolve();target=args.target.expanduser().resolve()
    if target.exists():raise FileExistsError('Target already exists: '+str(target))
    if not target.parent.is_dir():raise FileNotFoundError(target.parent)
    from pathak_runtime import configure,JAR_PATH,BIOBERT_REFERENCE
    configure()
    import pyspark
    from pyspark import SparkContext
    if SparkContext._active_spark_context is not None:raise RuntimeError('Use a fresh standalone Python process')
    compiler=Path(sys.prefix)/'bin/javac';java=Path(sys.prefix)/'bin/java';jar=Path(sys.prefix)/'bin/jar'
    source_archive=source/'bert_sentence_tensorflow'
    source_members=members(source_archive)
    original=ROOT/'.backups/biobert-models/sent_biobert_pubmed_base_cased/bert_sentence_tensorflow'
    original_members=members(original)
    for name in ['variables.data-00000-of-00001','variables.index']:
        if original_members[name]!=source_members[name]:raise ValueError('Weights differ from original: '+name)
    cache=ROOT/'.cache';cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='biobert-graph-build-',dir=str(cache)) as directory:
        work=Path(directory);classes=work/'classes';classes.mkdir()
        cp=str(JAR_PATH)+':'+str(Path(pyspark.__file__).parent/'jars/*')
        scripts=Path(__file__).resolve().parent
        subprocess.run([str(compiler),'-cp',cp,'-d',str(classes),str(scripts/'ProbeBertRuntime.java'),str(scripts/'PatchParallelSentenceEncoder.java')],check=True)
        tooljar=work/'audit.jar'
        subprocess.run([str(jar),'cf',str(tooljar),'-C',str(classes),'.'],check=True)
        with zipfile.ZipFile(source_archive) as archive:(work/'masked.pb').write_bytes(archive.read('saved_model.pb'))
        from pyspark.sql import SparkSession
        spark=(SparkSession.builder.appName('PATHAK-BioBERT-Graph-Build').master('local[2]')
            .config('spark.driver.memory','8g').config('spark.jars',str(JAR_PATH)+','+str(tooljar))
            .config('spark.ui.enabled','false').getOrCreate())
        spark.sparkContext.setLogLevel('ERROR')
        try:
            import nlu
            import pandas as pd
            pipe=nlu.load(BIOBERT_REFERENCE,path=str(source))
            model=next(c.model for c in pipe.components if c.model.__class__.__name__=='BertSentenceEmbeddings')
            model.setBatchSize(1);model._transfer_params_to_java()
            text=str(pd.read_csv(ROOT/'tutorials/GO_BP_Ontology_corups_Filtered.tsv',sep='\t').iloc[4564].Definition)
            spark._jvm.pathak.audit.ProbeBertRuntime.probe(model._java_obj,[text],'explicit2',str(work/'capture'))
        finally:spark.stop()
        execution=work/'capture/single_optimized_0.pb'
        output=work/'parallel.pb'
        subprocess.run([str(java),'-cp',str(classes)+':'+str(JAR_PATH),'PatchParallelSentenceEncoder',
            str(work/'masked.pb'),str(output),str(execution),str(args.sentence_intra_threads)],check=True)
        shutil.copytree(source,target)
        archive_path=target/'bert_sentence_tensorflow';temporary=target/'bert_sentence_tensorflow.tmp'
        with zipfile.ZipFile(source_archive) as old,zipfile.ZipFile(temporary,'w') as new:
            for info in old.infolist():new.writestr(info,output.read_bytes() if info.filename=='saved_model.pb' else old.read(info.filename))
        temporary.replace(archive_path)
        config=[16,args.intra_threads,40,args.inter_threads]
        metadata_path=target/'metadata/part-00000';metadata=json.loads(metadata_path.read_text())
        metadata['paramMap'].update(storageRef='pathak_biobert_masked_mean_parallel_sentence_v1',batchSize=args.batch_size,configProtoBytes=config)
        metadata_path.write_text(json.dumps(metadata,separators=(',',':'))+'\n')
        for modified in [archive_path,metadata_path]:
            crc=modified.parent/('.'+modified.name+'.crc')
            if crc.exists():crc.unlink()
        identity=members(archive_path)
        for name in ['variables.data-00000-of-00001','variables.index']:
            assert identity[name]==original_members[name]
        marker={
            'status':'candidate_requires_validation','protocol':'masked_mean_parallel_sentence_v1',
            'pooling':'float32 input_mask mean; retain CLS/SEP; exclude PAD',
            'execution':'MapDefun over sentences; shared checkpoint; legacy runtime arithmetic',
            'graph_sha256':identity['saved_model.pb'],'archive_sha256':sha(archive_path),
            'source_archive_sha256':sha(source_archive),'original_archive_sha256':sha(original),
            'members_sha256':identity,'original_members_sha256':original_members,
            'legacy_runtime_graph_sha256':sha(execution),'batch_size':args.batch_size,
            'tf_config_proto_bytes':config,'sentence_intra_threads':args.sentence_intra_threads,
            'case_sensitive':True,'max_sentence_length':128,'dimension':768,
        }
        (target/'pathak_pooling.json').write_text(json.dumps(marker,indent=2)+'\n')
        print(json.dumps({'created_model':str(target),'identity':marker},indent=2),flush=True)

if __name__=='__main__':main()
