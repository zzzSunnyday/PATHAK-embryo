"""Build a separate masked-mean BioBERT model without changing the environment.

Uses javac and TensorFlow's Java protobuf definitions from the existing NLP JAR.
No Python TensorFlow installation is needed; all encoder weights remain intact.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "biobert_masked_mean_cls_sep_v1"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    original = ROOT / ".backups/biobert-models/sent_biobert_pubmed_base_cased"
    target = args.work / "masked-mean-model"
    if target.exists():
        raise FileExistsError(target)
    args.work.mkdir(parents=True, exist_ok=True)
    jar = ROOT / "resources/jars/spark-nlp-assembly-4.0.2.jar"
    java_home = Path("/opt/conda/envs/pathak")
    java_source = Path(__file__).with_name("MaskedMeanGraph.java")
    classes = args.work / "java_classes"
    classes.mkdir()
    subprocess.run([str(java_home / "bin/javac"), "-cp", str(jar), "-d", str(classes),
                    str(java_source)], check=True)
    archive = original / "bert_sentence_tensorflow"
    original_hash = sha256(archive)
    original_graph = args.work / "original_graph.pb"
    revised_graph = args.work / "masked_mean_graph.pb"
    members = {}
    with zipfile.ZipFile(archive) as source:
        assert set(source.namelist()) == {"saved_model.pb", "variables.index", "variables.data-00000-of-00001"}
        original_graph.write_bytes(source.read("saved_model.pb"))
    command = [str(java_home / "bin/java"), "-cp", str(classes) + os.pathsep + str(jar),
               "MaskedMeanGraph", str(original_graph), str(revised_graph)]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    (args.work / "graph_patch_inspection.txt").write_text(result.stdout, encoding="utf-8")
    print(result.stdout, flush=True)
    staging = target.with_name(target.name + ".staging-" + uuid.uuid4().hex)
    shutil.copytree(original, staging)
    changed_crc = staging / ".bert_sentence_tensorflow.crc"
    if changed_crc.exists():
        changed_crc.unlink()  # This checksum describes the original archive, not the patched copy.
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(staging / archive.name, "w") as output:
        for info in source.infolist():
            data = source.read(info.filename)
            before_hash = hashlib.sha256(data).hexdigest()
            if info.filename == "saved_model.pb":
                data = revised_graph.read_bytes()
            output.writestr(info, data)
            members[info.filename] = {"original_sha256": before_hash,
                                      "patched_sha256": hashlib.sha256(data).hexdigest(),
                                      "bytes": len(data)}
    for name, identity in members.items():
        if name != "saved_model.pb":
            assert identity["original_sha256"] == identity["patched_sha256"], name
    vocabulary = {str(p.relative_to(original)): sha256(p)
                  for p in sorted((original / "fields/vocabulary").rglob("*")) if p.is_file()}
    for name, value in vocabulary.items():
        assert sha256(staging / name) == value
    assert (original / "metadata/part-00000").read_bytes() == (staging / "metadata/part-00000").read_bytes()
    assert sha256(archive) == original_hash
    report = {
        "status": "patched_not_yet_validated",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "formula": "sum(hidden * cast(input_mask)[...,None], axis=1) / sum(mask, axis=1)",
        "special_tokens": "CLS and SEP retained; only PAD excluded",
        "original_model": str(original.relative_to(ROOT)),
        "patched_model": str(target.relative_to(ROOT)),
        "original_archive_sha256": original_hash,
        "patched_archive_sha256": sha256(staging / archive.name),
        "archive_members": members,
        "vocabulary_sha256": vocabulary,
        "metadata_sha256": sha256(original / "metadata/part-00000"),
        "tokenizer_and_metadata": "unchanged; caseSensitive=true; maxSentenceLength=128",
        "implementation": "8 graph nodes added; pooled_output input rewired; Java TensorFlow executes pooling",
        "source_reference": "https://github.com/JohnSnowLabs/spark-nlp/blob/4.0.2/src/main/scala/com/johnsnowlabs/ml/tensorflow/TensorflowBert.scala",
    }
    (staging / "pathak_pooling.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    staging.rename(target)
    (args.work / "model-patch.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
