"""Tokenise FineWeb-Edu parquet -> uint16 shards (GPT-2 BPE, tiktoken "gpt2", EOT 50256 before every document).
Validation = the first VAL_DOCS documents (row order) of the first source file, whole documents only.
Train = every other document, in source order, cut into SHARD-token files (documents may span shard edges).
Writes DATA_MANIFEST.json (shards + sources + tokenizer); progress to stdout.
"""
import glob, hashlib, json, os, sys, time
from multiprocessing import Pool
import numpy as np
import pyarrow.parquet as pq
import tiktoken

OUT = os.environ.get("FW_OUT", "C:/Users/seal/datasets/fineweb_edu")
RAW = os.environ.get("FW_RAW", OUT + "/raw/sample/10BT")
MANIFEST = sys.argv[1] if len(sys.argv) > 1 else OUT + "/DATA_MANIFEST.json"
SHARD = int(os.environ.get("FW_SHARD", 100_000_000))
VAL_DOCS = int(os.environ.get("FW_VAL_DOCS", 10_000))
EOT = 50256


def work(unit):
    path, rg = unit
    enc = tiktoken.get_encoding("gpt2")
    texts = pq.ParquetFile(path).read_row_group(rg, columns=["text"]).column("text").to_pylist()
    toks = enc.encode_ordinary_batch(texts, num_threads=1)
    lens = np.fromiter((len(t) + 1 for t in toks), dtype=np.int64, count=len(toks))
    arr = np.empty(int(lens.sum()), dtype=np.uint16)
    i = 0
    for t in toks:
        arr[i] = EOT
        a = np.asarray(t, dtype=np.int64)
        assert a.size == 0 or a.max() < EOT, "token id >= 50256 inside a document"
        arr[i + 1:i + 1 + a.size] = a
        i += 1 + a.size
    return arr, lens


class Writer:
    def __init__(self, prefix):
        self.prefix, self.buf, self.n, self.k, self.recs = prefix, np.empty(SHARD, np.uint16), 0, 0, []

    def add(self, arr):
        while arr.size:
            take = min(SHARD - self.n, arr.size)
            self.buf[self.n:self.n + take] = arr[:take]
            self.n += take
            arr = arr[take:]
            if self.n == SHARD:
                self.flush()

    def flush(self):
        if not self.n:
            return
        name = f"{self.prefix}_{self.k:06d}.bin"
        data = self.buf[:self.n].astype("<u2", copy=False).tobytes()
        with open(f"{OUT}/{name}", "wb") as f:
            f.write(data)
        self.recs.append({"file": name, "bytes": len(data), "tokens": self.n, "sha256": hashlib.sha256(data).hexdigest()})
        print(json.dumps({"shard": self.recs[-1], "t": round(time.time() - T0, 1)}), flush=True)
        self.k, self.n = self.k + 1, 0


if __name__ == "__main__":
    T0 = time.time()
    files = sorted(glob.glob(RAW + "/*.parquet"))
    units = [(f.replace("\\", "/"), rg) for f in files for rg in range(pq.ParquetFile(f).num_row_groups)]
    print(json.dumps({"files": files, "units": len(units)}), flush=True)
    tr, va = Writer("train"), Writer("val")
    docs_seen = docs_val = docs_tr = 0
    with Pool(int(os.environ.get("NPROC", 24))) as pool:
        for u, (arr, lens) in enumerate(pool.imap(work, units, chunksize=1)):
            if docs_seen < VAL_DOCS:
                nv = min(VAL_DOCS - docs_seen, lens.size)
                cut = int(lens[:nv].sum())
                va.add(arr[:cut]); tr.add(arr[cut:])
                docs_val += nv; docs_tr += lens.size - nv
            else:
                tr.add(arr); docs_tr += lens.size
            docs_seen += lens.size
            if u % 50 == 0:
                done = sum(r["tokens"] for r in tr.recs) + tr.n
                print(json.dumps({"unit": u, "of": len(units), "docs": docs_seen, "train_tokens": done,
                                  "t": round(time.time() - T0, 1)}), flush=True)
    va.flush(); tr.flush()
    man = {"tokenizer": {"library": "tiktoken", "version": tiktoken.__version__, "encoding": "gpt2",
                         "eot_id": EOT, "rule": "EOT prepended to every document (so one EOT between consecutive documents)",
                         "dtype": "uint16 little-endian, no header"},
           "split": {"rule": f"validation = first {VAL_DOCS} documents in row order of {os.path.basename(files[0])}; "
                             "train = all remaining documents in source order", "docs_val": docs_val, "docs_train": docs_tr},
           "shard_dir": OUT, "shard_tokens": SHARD,
           "val_shards": va.recs, "train_shards": tr.recs,
           "tokens_val": sum(r["tokens"] for r in va.recs), "tokens_train": sum(r["tokens"] for r in tr.recs),
           "tokenise_wall_s": round(time.time() - T0, 1)}
    man["tokens_total"] = man["tokens_val"] + man["tokens_train"]
    json.dump(man, open(MANIFEST, "w"), indent=1)
    print("DONE", json.dumps({k: man[k] for k in ("tokens_val", "tokens_train", "tokens_total", "tokenise_wall_s")}), flush=True)
