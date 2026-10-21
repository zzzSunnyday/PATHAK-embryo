"""Reproduce the CPU2/global8 comparison in a new audit directory."""
import argparse,hashlib,json,os,runpy,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
FROZEN=ROOT/'.backups/biobert-performance-20261021T161556Z'
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,help='A NEW direct child of PATHAK/.backups')
    parser.add_argument('--prepare-only',action='store_true',help='Check inputs and stage tools without running the long benchmark')
    args=parser.parse_args()
    work=(args.work or ROOT/'.backups'/('biobert-global8-recheck-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    if work.parent!=(ROOT/'.backups').resolve() or work.exists():raise SystemExit('--work must be a new direct child of PATHAK/.backups')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(ROOT/'tutorials/GO_BP_Ontology_corups_Filtered.tsv')!='3d872b81692702f966aa7241afa9b2be922d36b2ea8105832095431e99bc944f':raise SystemExit('GO corpus changed; freeze and review a new benchmark corpus first')
    files=['benchmark_global8.py','cpu_budget8_worker.py','run_profile_v2.py','evaluate_retrieval.py','validate_global8_staged.py','profile-v3.jar','corpus.jsonl','corpus-manifest.json','go-baseline.npy','literature-baseline.npy','go-bank-baseline.npy','literature-original-single.npy']
    for name in files:
        if not (FROZEN/name).is_file():raise SystemExit('Frozen input/tool missing: '+str(FROZEN/name))
    work.mkdir();identities={};sources={}
    for name in files:
        source = ROOT/'resources/checks/biobert-performance'/name if name.endswith('.py') else FROZEN/name
        if not source.is_file():raise SystemExit('Current audit source missing: '+str(source))
        shutil.copy2(source,work/name);identities[name]=sha(work/name);sources[name]=str(source.relative_to(ROOT))
    (work/'recheck-inputs.json').write_text(json.dumps({'source':str(FROZEN.relative_to(ROOT)),'files_sha256':identities,'file_sources':sources,'baseline_model':'.backups/biobert-models/sent_biobert_pubmed_base_cased_parallel_masked_mean','candidate_global_intra':8,'candidate_sentence_max_intra':2,'OMP_NUM_THREADS':2,'batch_size':16},indent=2)+'\n')
    print(json.dumps({'prepared':str(work),'encoding_started':not args.prepare_only}),flush=True)
    if args.prepare_only:return
    os.environ.update(OMP_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(work))
    import pathak_runtime as runtime
    # Keep the original producer frozen. Its default loader must resolve to
    # CPU2 for this controller, while the separate worker bootstraps CPU8.
    runtime.BIOBERT_PATH=runtime.BIOBERT_BASELINE_PATH
    runpy.run_path(str(work/'benchmark_global8.py'),run_name='__main__')
    subprocess.run([sys.executable,'-u',str(work/'validate_global8_staged.py')],cwd=str(ROOT),check=True)
if __name__=='__main__':main()
