"""Download the pinned, permissively licensed encoder before offline computation."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODEL="ibm-granite/granite-embedding-97m-multilingual-r2"
REVISION="835ad14087e140460703cf0fae09f97d469d65c2"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--kind",choices=["encoder","reranker"],default="encoder")
    a=p.parse_args()
    model,revision=(MODEL,REVISION) if a.kind=="encoder" else ("Qwen/Qwen3-Reranker-0.6B","2925c98b11f00b3364acaeb0a669f498ac45bf54")
    info=HfApi(token=False).model_info(model,revision=revision)
    license_name=(info.card_data or {}).get("license")
    if info.sha!=revision or license_name!="apache-2.0":
        raise ValueError("Model revision or license differs from the experiment declaration")
    snapshot_download(model,revision=revision,local_dir=a.output,token=False,
                      allow_patterns=["*.json","*.safetensors","*.txt","*.model","README.md","LICENSE"],
                      ignore_patterns=["onnx/*","openvino/*"])
    h=hashlib.sha256()
    with (a.output/"model.safetensors").open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):
            h.update(b)
    (a.output/"provenance.json").write_text(json.dumps({"model":model,"revision":revision,
        "license":license_name,"weights_sha256":h.hexdigest()},indent=2)+"\n")


if __name__=="__main__":
    main()
