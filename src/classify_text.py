"""Text-level hybrid classifier at token boundaries (phase4 §2.3), robust to leading-space
tokenization variants and trailing punctuation. Used for re-scoring saved outputs."""
import re


def norm(s):
    s = s.strip().split("\n")[0].strip()
    s = re.sub(r"[\s\.\!\?\,\;\:\"\'\)\(]+$", "", s)
    s = re.sub(r"^[\"\'\(]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def pieces(tok, ids):
    """prefixes/suffixes of the mode string at its token boundaries (decoded, normalized)."""
    pre = [norm(tok.decode(ids[:i])) for i in range(1, len(ids))]
    suf = [tok.decode(ids[j:]) for j in range(1, len(ids))]
    return pre, suf


def classify_text(tok, out_text, a_name, b_name, tok_a, tok_b):
    o = norm(out_text)
    if o.lower() == a_name.lower():
        return "mode_a"
    if o.lower() == b_name.lower():
        return "mode_b"
    pa, sa = pieces(tok, tok_a)
    pb, sb = pieces(tok, tok_b)
    cands = set()
    for p in pa:
        for s in sb:
            cands.add(norm(p + s)); cands.add(norm(p + " " + s.lstrip()))
    for p in pb:
        for s in sa:
            cands.add(norm(p + s)); cands.add(norm(p + " " + s.lstrip()))
    cands -= {a_name, b_name}
    if o in cands:
        return "hybrid"
    # word-level hybrid: first word from one mode, rest from the other
    aw, bw = a_name.split(), b_name.split()
    ow = o.split()
    if len(ow) == 2 and ((ow[0] == aw[0] and ow[1] == bw[1]) or (ow[0] == bw[0] and ow[1] == aw[1])) and o not in (a_name, b_name):
        return "hybrid"
    return "other"
