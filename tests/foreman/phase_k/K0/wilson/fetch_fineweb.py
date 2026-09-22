"""Download FineWeb-Edu sample/10BT parquet shards 000-003 at a pinned revision, verify LFS sha256."""
import hashlib, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download, HfApi

REPO, REV = "HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
RAW = "C:/Users/seal/datasets/fineweb_edu/raw"
FILES = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 4)]

meta = {s.rfilename: s for s in HfApi().dataset_info(REPO, revision=REV, files_metadata=True).siblings}

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

def one(fn):
    t = time.time()
    p = hf_hub_download(REPO, fn, repo_type="dataset", revision=REV, local_dir=RAW)
    dt = time.time() - t
    s = sha256(p)
    rec = {"file": fn, "path": p.replace("\\", "/"), "bytes": meta[fn].size, "sha256": s,
           "sha256_expected_lfs": meta[fn].lfs.sha256, "sha_ok": s == meta[fn].lfs.sha256,
           "url": f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{fn}", "download_s": round(dt, 1)}
    print(json.dumps(rec), flush=True)
    return rec

with ThreadPoolExecutor(len(FILES)) as ex:
    recs = list(ex.map(one, FILES))
json.dump(recs, open(RAW + "/../source_files.json", "w"), indent=1)
print("DONE", all(r["sha_ok"] for r in recs), flush=True)
