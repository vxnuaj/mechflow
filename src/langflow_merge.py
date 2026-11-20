"""Merge LangFlow decomposition chunks (<root>/chunk_<start>/) in start order into <root>/merged/ (same schema as one run)."""
import glob, json, os, sys
root = sys.argv[1]; ch = sorted(glob.glob(f"{root}/chunk_*"), key=lambda p: int(p.rsplit("_", 1)[1]))
cells, plain, ident, n = None, [], [], 0
for c in ch:
    C = json.load(open(f"{c}/cells.json")); j = next(iter(C)); m = json.load(open(f"{c}/meta.json")); assert m["start"] == n, (c, m["start"], n)
    if cells is None: cells = {k: [] for k in C[j]}; J = j
    for k in cells: cells[k] += C[j][k]
    plain += json.load(open(f"{c}/plain_on.json"))["tokens"]; ident.append(m["identity_min"]); n += m["n"]
os.makedirs(f"{root}/merged", exist_ok=True)
json.dump({J: cells}, open(f"{root}/merged/cells.json", "w")); json.dump({"tokens": plain}, open(f"{root}/merged/plain_on.json", "w"))
json.dump({"n": n, "chunks": len(ch), "identity_min": min(ident)}, open(f"{root}/merged/meta.json", "w")); print("merged", len(ch), "chunks,", n, "samples")
