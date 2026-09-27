"""Additional label-free evidence for a frozen dense owner's winner decision.

The base features, candidate identities and pair-model winner remain unchanged.
Transliteration is an extra view, never a replacement for the original script.
"""
import math
import re
from functools import lru_cache

import numpy as np
from anyascii import anyascii
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from pair_features import LEGAL, normalized


@lru_cache(maxsize=60000)
def views(name, address):
    n, a = normalized(anyascii(name)), normalized(anyascii(address))
    core = " ".join(t for t in n.split() if t not in LEGAL)
    alpha = " ".join(re.findall(r"[a-z]+", a))
    skeleton = re.sub(r"[aeiou\s]", "", core)
    numbers = tuple(str(int(v)) for v in re.findall(r"\d+", a) if len(v) < 19)
    return n, core, core.replace(" ", ""), skeleton, a, alpha, numbers


TEXT_METRICS = ["ratio", "sort", "set", "jaro", "edit", "exact", "source_coverage", "target_coverage"]
VIEW_NAMES = ["roman_name", "roman_core", "compact_core", "consonants", "roman_address", "address_alpha"]
PAIR_NAMES = [f"rich_{v}_{m}" for v in VIEW_NAMES for m in TEXT_METRICS] + [
    "rich_missing_name_tokens", "rich_extra_name_tokens", "rich_missing_name_chars", "rich_extra_name_chars",
    "rich_name_missing_fuzzy_mean", "rich_name_missing_fuzzy_min", "rich_name_extra_fuzzy_mean",
    "rich_name_extra_fuzzy_min", "rich_target_name_prefix", "rich_source_name_prefix",
    "rich_name_initials_equal", "rich_number_both", "rich_number_equal", "rich_number_edit",
    "rich_number_same_length", "rich_number_hamming", "rich_number_subsequence", "rich_number_reversed",
    "rich_number_log_delta", "rich_number_delta_per_length", "rich_number_target_in_source",
    "rich_number_source_in_target", "rich_numeric_count_diff", "rich_numeric_best_edit",
    "rich_name_digit_equal", "rich_name_digit_conflict", "rich_postcode_overlap",
    "rich_name_residual_address_overlap", "rich_address_residual_name_overlap",
]
RIVAL_NAMES = ["rich_dense_rival_name", "rich_dense_rival_address", "rich_dense_rival_alpha",
               "rich_dense_rival_number_equal", "rich_name_vs_dense_rival", "rich_address_vs_dense_rival",
               "rich_alpha_vs_dense_rival", "rich_number_vs_dense_rival"]
PEER_NAMES = ["rich_peer_exists", "rich_peer_name", "rich_peer_address", "rich_peer_alpha",
              "rich_peer_number_equal", "rich_peer_pair_probability", "rich_peer_source_agreement"]
EXTRA_NAMES = PAIR_NAMES + RIVAL_NAMES + PEER_NAMES
# These aggregates were not exported by the completed test run. Omitting them
# lets deployment reuse the exact frozen pair winners without rescoring 100M edges.
OMIT_BASE = {'pair_sum', 'pair_count_05', 'pair_count_01'}


def text_evidence(a, b):
    aa, bb = set(a.split()), set(b.split())
    shared = len(aa & bb)
    return [fuzz.ratio(a,b)/100, fuzz.token_sort_ratio(a,b)/100,
            fuzz.token_set_ratio(a,b)/100, JaroWinkler.normalized_similarity(a,b),
            Levenshtein.normalized_similarity(a,b), float(bool(a) and a==b),
            shared/max(1,len(aa)), shared/max(1,len(bb))]


def fuzzy_missing(a, b):
    if not a:
        return [1., 1.]
    values = [max((fuzz.ratio(x,y)/100 for y in b), default=0.) for x in sorted(a)]
    return [sum(values)/len(values), min(values)]


def is_subsequence(a, b):
    it = iter(b)
    return bool(a) and all(any(c==v for v in it) for c in a)


def pair_evidence(source, target):
    s, t = views(*source), views(*target)
    out = [value for a,b in zip(s[:6],t[:6]) for value in text_evidence(a,b)]
    sw,tw=set(s[1].split()),set(t[1].split())
    missing,extra=sw-tw,tw-sw
    sn,tn=s[6],t[6]
    sf,tf=(sn[0] if sn else ""),(tn[0] if tn else "")
    both=bool(sf and tf)
    delta=abs(int(sf)-int(tf)) if both else 0
    same_len=both and len(sf)==len(tf)
    name_s=set(re.findall(r"\d+",s[0]));name_t=set(re.findall(r"\d+",t[0]))
    out += [len(missing),len(extra),sum(map(len,missing)),sum(map(len,extra)),
            *fuzzy_missing(missing,tw),*fuzzy_missing(extra,sw),
            float(bool(t[2]) and s[2].startswith(t[2])),float(bool(s[2]) and t[2].startswith(s[2])),
            float(bool(sw and tw) and ''.join(x[0] for x in s[1].split())==''.join(x[0] for x in t[1].split())),
            float(both),float(both and sf==tf),Levenshtein.normalized_similarity(sf,tf) if both else -1,
            float(same_len),sum(a!=b for a,b in zip(sf,tf)) if same_len else -1,
            float(both and (is_subsequence(sf,tf) or is_subsequence(tf,sf))),float(both and sf==tf[::-1]),
            math.log1p(delta) if both else -1,delta/max(1,int(sf),int(tf)) if both else -1,
            float(bool(tf) and tf in sn),float(bool(sf) and sf in tn),len(tn)-len(sn),
            max((Levenshtein.normalized_similarity(a,b) for a in sn for b in tn),default=-1),
            float(bool(name_s) and name_s==name_t),float(bool(name_s and name_t) and not name_s&name_t),
            len({x for x in sn if len(x)>=5}&{x for x in tn if len(x)>=5}),
            len(extra&set(s[5].split()))/max(1,len(extra)),
            len((set(t[5].split())-set(s[5].split()))&sw)/max(1,len(sw))]
    if len(out)!=len(PAIR_NAMES):
        raise ValueError('Rich pair schema mismatch')
    return out


def quick_evidence(source, target):
    s,t=views(*source),views(*target)
    return [fuzz.ratio(s[1],t[1])/100,fuzz.ratio(s[4],t[4])/100,
            fuzz.token_sort_ratio(s[5],t[5])/100,float(bool(s[6] and t[6]) and s[6][0]==t[6][0])]


def enrich_row(row):
    source,target,rival,peer,peer_probability,peer_source_agreement=row
    own=quick_evidence(source,target);other=quick_evidence(rival,target)
    peer_features=([1.,*quick_evidence(peer,target),peer_probability,peer_source_agreement]
                   if peer is not None else [0.,-1.,-1.,-1.,-1.,-1.,0.])
    values=pair_evidence(source,target)+other+[a-b for a,b in zip(own,other)]+peer_features
    if len(values)!=len(EXTRA_NAMES) or not np.isfinite(values).all():
        raise ValueError('Invalid rich-context row')
    return values


def peer_indices(owner, probability, text_keys):
    """Best other incoming record; lexical tie breaking never depends on IDs."""
    result=np.full(len(owner),-1,dtype=np.int64)
    groups={}
    for i,o in enumerate(owner):
        groups.setdefault(int(o),[]).append(i)
    for indices in groups.values():
        if len(indices)<2:
            continue
        top=sorted(indices,key=lambda i:(-float(probability[i]),text_keys[i]))[:2]
        for i in indices:
            result[i]=top[1] if i==top[0] else top[0]
    return result
