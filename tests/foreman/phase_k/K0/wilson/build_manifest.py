"""Assemble tests/foreman/phase_k/DATA_MANIFEST.json: sources (re-hashed vs HF LFS sha256), shards (re-hashed from disk), tokenizer, license."""
import hashlib, json, os
from concurrent.futures import ThreadPoolExecutor

OUT = "C:/Users/seal/datasets/fineweb_edu"
REPO_MANIFEST = "C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_k/DATA_MANIFEST.json"
WD = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
REPO, REV = "HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
LFS = {  # from HfApi().dataset_info(REPO, files_metadata=True) at REV, 2026-09-23 01:38
    "000_00000.parquet": (2152819114, "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871"),
    "001_00000.parquet": (2152222432, "3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693"),
    "002_00000.parquet": (2151796315, "547ae182d132c9f06b6ce63149567208ea9f57630bfd9b1a2938e504f0c9ebd7"),
    "003_00000.parquet": (2152437524, "22184e6eb25759ddd97783751ffc73e1705dfa2542e630dae1f2a8bac8ee6ddb"),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


tok = json.load(open(OUT + "/DATA_MANIFEST.json"))
srcs = [f"{OUT}/raw/sample/10BT/{n}" for n in LFS]
shards = tok["val_shards"] + tok["train_shards"]
with ThreadPoolExecutor(8) as ex:
    src_sha = list(ex.map(sha, srcs))
    shard_sha = list(ex.map(sha, [f"{OUT}/{r['file']}" for r in shards]))
sources = [{"file": f"sample/10BT/{n}", "bytes": os.path.getsize(p), "sha256": s, "hf_lfs_sha256": LFS[n][1],
            "hf_lfs_bytes": LFS[n][0], "match": s == LFS[n][1] and os.path.getsize(p) == LFS[n][0],
            "url": f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/sample/10BT/{n}"}
           for n, p, s in zip(LFS, srcs, src_sha)]
for r, s in zip(shards, shard_sha):
    r["sha256_disk_matches_write"] = (s == r["sha256"])
    assert os.path.getsize(f"{OUT}/{r['file']}") == r["bytes"] == 2 * r["tokens"]
readme = WD + "/fineweb_edu_README.md"
man = {
    "dataset": {"repo": REPO, "revision": REV, "subset": "sample-10BT (files 000-003 of 000-013)",
                "card_url": f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/README.md",
                "card_sha256": sha(readme),
                "license_field": "odc-by",
                "license_text_verbatim": "The dataset is released under the **Open Data Commons Attribution License (ODC-By) v1.0** [license](https://opendatacommons.org/licenses/by/1-0/). The use of this dataset is also subject to [CommonCrawl's Terms of Use](https://commoncrawl.org/terms-of-use).",
                "license_text_source": "dataset card README.md line 632 (section 'Licensing Information'); front matter line 2: 'license: odc-by'"},
    "sources": sources,
    "tokenizer": tok["tokenizer"], "split": tok["split"], "shard_dir": OUT, "shard_tokens": tok["shard_tokens"],
    "val_shards": tok["val_shards"], "train_shards": tok["train_shards"],
    "tokens_val": tok["tokens_val"], "tokens_train": tok["tokens_train"], "tokens_total": tok["tokens_total"],
    "tokenise_wall_s": tok["tokenise_wall_s"],
    "scripts": {n: {"path": f"{WD}/{n}", "sha256": sha(f"{WD}/{n}")} for n in ("fetch_curl.sh", "tokenize_fineweb.py", "build_manifest.py")},
}
json.dump(man, open(REPO_MANIFEST, "w"), indent=1)
print(json.dumps({"sources_match": [s["match"] for s in sources], "shards": len(shards),
                  "shard_disk_ok": all(r["sha256_disk_matches_write"] for r in shards),
                  "tokens_total": man["tokens_total"], "tokens_train": man["tokens_train"], "tokens_val": man["tokens_val"]}))
