"""Label-free lexical, numeric and dense owner-competition features."""
import math
import re
import unicodedata
from functools import lru_cache

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

LEGAL = set("inc incorporated llc ltd limited pvt private corp corporation co company llp plc gmbh sarl sas sa".split())
ABBR = dict(zip("rd st ave av blvd ln dr hwy ste apt fl bldg ctr pvt ltd inc corp".split(),
                "road street avenue avenue boulevard lane drive highway suite apartment floor building center private limited incorporated corporation".split()))


def normalized(text):
    # Keep every script. Accent folding helps French without deleting Indic text.
    chars, latin = [], False
    for c in unicodedata.normalize("NFKD", text).casefold():
        mark = unicodedata.category(c).startswith("M")
        if mark and latin:
            continue
        if not mark:
            latin = "LATIN" in unicodedata.name(c, "")
        chars.append(c if c.isalnum() or mark else " ")
    text = "".join(chars)
    return " ".join(ABBR.get(t, t) for t in text.split())


def gram(text):
    compact = text.replace(" ", "")
    return frozenset(compact[i:i+3] for i in range(max(0, len(compact)-2)))


def tokens(text):
    return frozenset(text.split())


@lru_cache(maxsize=25000)
def record(name, address):
    n, a = normalized(name), normalized(address)
    core = " ".join(t for t in n.split() if t not in LEGAL)
    nums = tuple(str(int(t)) for t in re.findall(r"\d+", a) if len(t) <= 18)
    long_nums = frozenset(t for t in nums if len(t) >= 5)
    first = nums[0] if nums else ""
    return (n, a, core, tokens(n), tokens(a), tokens(core), gram(n), gram(a),
            frozenset(nums), long_nums, first,
            "".join(t[0] for t in core.split() if t),
            any(ord(c) > 127 for c in name), any(ord(c) > 127 for c in address))


def overlap(a, b):
    shared = len(a & b)
    return [shared/max(1, len(a | b)), shared/max(1, min(len(a), len(b))),
            shared/max(1, len(a)), shared/max(1, len(b)), float(shared)]


def text_features(a, b, wa, wb, ga, gb):
    return [float(bool(a) and a == b), fuzz.ratio(a, b)/100, fuzz.token_sort_ratio(a, b)/100,
            fuzz.token_set_ratio(a, b)/100, JaroWinkler.normalized_similarity(a, b),
            Levenshtein.normalized_similarity(a, b),
            min(len(a), len(b))/max(1, len(a), len(b)), *overlap(wa, wb),
            len(ga & gb)/max(1, len(ga | gb))]


TEXT_NAMES = ["exact", "ratio", "sort", "set", "jaro", "edit", "length_ratio",
              "jaccard", "overlap_min", "overlap_source", "overlap_target", "shared", "char3"]
BASE_NAMES = [f"{view}_{name}" for view in ("name", "address") for name in TEXT_NAMES] + [
    "core_exact", "core_ratio", "core_sort", "core_set", "acronym_exact",
    "numbers_jaccard", "numbers_overlap", "numbers_source", "numbers_target", "numbers_shared",
    "long_numbers_jaccard", "long_numbers_shared", "first_number_exact", "first_number_conflict",
    "first_number_log_delta", "source_name_empty", "target_name_empty", "source_address_empty",
    "target_address_empty", "source_name_length", "target_name_length", "source_address_length",
    "target_address_length", "source_name_non_ascii", "target_name_non_ascii",
    "source_address_non_ascii", "target_address_non_ascii", "source_name_in_target_address",
    "target_name_in_source_address", "name_address_product", "name_address_gap",
    "source_name_frequency", "target_source3", "dense_score", "dense_rank", "dense_to_first",
    "dense_to_second", "dense_to_last", "dense_first_second", "dense_close_count", "dense_above_095",
    "dense_above_090"]
COMP_COLUMNS = ["name_ratio", "address_ratio", "core_ratio", "numbers_jaccard"]
FEATURE_NAMES = BASE_NAMES + [f"{c}_{suffix}" for c in COMP_COLUMNS for suffix in ("to_best_rival", "rank")]


def pair(source, target, score, rank, scores, frequency, source3):
    sn, sa, sc, snw, saw, scw, sng, sag, sdigits, slong, sfirst, sac, snon, sanon = source
    tn, ta, tc, tnw, taw, tcw, tng, tag, tdigits, tlong, tfirst, tac, tnon, tanon = target
    name_sim = fuzz.ratio(sn, tn)/100
    addr_sim = fuzz.ratio(sa, ta)/100
    both_numbers = bool(sfirst and tfirst)
    delta = math.log1p(abs(int(sfirst)-int(tfirst))) if both_numbers else -1
    result = text_features(sn, tn, snw, tnw, sng, tng) + text_features(sa, ta, saw, taw, sag, tag) + [
        float(bool(sc) and sc == tc), fuzz.ratio(sc, tc)/100, fuzz.token_sort_ratio(sc, tc)/100,
        fuzz.token_set_ratio(sc, tc)/100, float(bool(sac) and sac == tac),
        *overlap(sdigits, tdigits), len(slong & tlong)/max(1, len(slong | tlong)), float(len(slong & tlong)),
        float(both_numbers and sfirst == tfirst), float(both_numbers and sfirst != tfirst), delta,
        float(not sn), float(not tn), float(not sa), float(not ta),
        math.log1p(len(sn)), math.log1p(len(tn)), math.log1p(len(sa)), math.log1p(len(ta)),
        float(snon), float(tnon), float(sanon), float(tanon),
        len(snw & taw)/max(1, len(snw)), len(tnw & saw)/max(1, len(tnw)),
        name_sim*addr_sim, abs(name_sim-addr_sim), math.log1p(frequency), float(source3),
        float(score), float(rank), float(score-scores[0]), float(score-scores[1]),
        float(score-scores[-1]), float(scores[0]-scores[1]),
        float(np.sum(scores >= scores[0]-0.01)), float(np.sum(scores >= 0.95)), float(np.sum(scores >= 0.90))]
    if len(result) != len(BASE_NAMES):
        raise ValueError("Feature schema mismatch")
    return result


def shortlist(source_records, target_record, scores, frequencies, source3):
    target = record(*target_record)
    base = np.asarray([pair(record(*s), target, scores[j], j, scores, frequencies[j], source3)
                       for j, s in enumerate(source_records)], dtype=np.float32)
    extra = []
    for name in COMP_COLUMNS:
        values = base[:, BASE_NAMES.index(name)]
        rival = np.asarray([np.max(np.delete(values, j)) for j in range(len(values))])
        ranks = (values[None, :] > values[:, None]).sum(axis=1)
        extra += [values-rival, ranks]
    result = np.column_stack([base, *extra]).astype(np.float32)
    if not np.isfinite(result).all() or result.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Invalid pair features")
    return result
