"""Prepare supplied records and deterministic, name-group-disjoint experiments."""
import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import duckdb

SEED = 20260927
LEGAL = set("inc incorporated llc ltd limited pvt private corp corporation co company llp plc gmbh sarl sas sa".split())


def normalized_group(value):
    text = unicodedata.normalize("NFKC", value).casefold()
    text = "".join(c if unicodedata.category(c)[0] in "LMN" else " " for c in text)
    tokens = [t for t in text.split() if t not in LEGAL]
    return " ".join(sorted(tokens)) or "<empty>"


def bucket(value):
    return int.from_bytes(hashlib.blake2b(f"{SEED}|{value}".encode(), digest_size=8).digest(), "little") % 1000


def sql(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous-ids", type=Path, required=True)
    p.add_argument("--max-encoder-pairs", type=int, default=1000000)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "complete.json").exists():
        raise SystemExit("Preparation already complete; use another output directory")
    started = time.monotonic()
    db = duckdb.connect(str(a.output / "records.duckdb"))
    db.execute("SET threads=8; SET memory_limit='24GB'")
    db.execute(f"SET temp_directory={sql(a.output / 'spill')}")
    db.create_function("group_name", normalized_group, ["VARCHAR"], "VARCHAR")
    db.create_function("split_bucket", bucket, ["VARCHAR"], "INTEGER")

    def read_tsv(path):
        return f"read_csv({sql(path)}, delim='\t', header=true, quote='', escape='', all_varchar=true)"

    db.execute(f"CREATE OR REPLACE TABLE previous_ids AS SELECT column0 AS entity_id FROM read_csv({sql(a.previous_ids)}, header=false, all_varchar=true)")
    for split in ("train", "test"):
        folder = a.dataset / split
        if not folder.is_dir():
            continue
        db.execute(f"""CREATE OR REPLACE TABLE {split}_s1 AS
          SELECT entity_id, coalesce(business_name,'') AS business_name,
            coalesce(business_address,'') AS business_address, country,
            country || '|' || group_name(coalesce(business_name,'')) AS name_group
          FROM {read_tsv(folder / (split + '_source1.tsv'))}""")
        db.execute(f"""CREATE OR REPLACE TABLE {split}_targets AS
          SELECT entity_id, coalesce(business_name,'') AS business_name,
            coalesce(business_address,'') AS business_address, country
          FROM {read_tsv(folder / (split + '_source2.tsv'))}
          UNION ALL SELECT entity_id, coalesce(business_name,''), coalesce(business_address,''), country
          FROM {read_tsv(folder / (split + '_source3.tsv'))}""")
        print(json.dumps({"loaded": split, "seconds": time.monotonic() - started}), flush=True)
    db.execute("CREATE OR REPLACE TABLE previous_groups AS SELECT DISTINCT name_group FROM train_s1 JOIN previous_ids USING(entity_id)")
    # Historical development groups cannot enter any new development/reserve split.
    db.execute("""CREATE OR REPLACE TABLE roles AS SELECT s.entity_id, s.country, s.name_group,
      CASE WHEN p.name_group IS NOT NULL THEN 'encoder'
           WHEN split_bucket(s.name_group)<5 THEN 'reserve'
           WHEN split_bucket(s.name_group)<10 THEN 'tune'
           WHEN split_bucket(s.name_group)<25 THEN 'context'
           WHEN split_bucket(s.name_group)<40 THEN 'pair'
           ELSE 'encoder' END AS role
      FROM train_s1 s LEFT JOIN previous_groups p USING(name_group)""")
    truth_path = a.dataset / "train/train_ground_truth.tsv"
    db.execute(f"CREATE OR REPLACE TABLE truth AS SELECT source1_entity_id AS s1_id, coalesce(matched_entity_ids,'') AS truth_ids FROM {read_tsv(truth_path)}")
    db.execute("CREATE OR REPLACE TABLE owners AS SELECT s1_id, unnest(string_split(truth_ids, ',')) AS target_id FROM truth WHERE truth_ids<>''")
    duplicates = db.execute("SELECT count(*) FROM (SELECT target_id FROM owners GROUP BY 1 HAVING count(*)>1)").fetchone()[0]
    if duplicates:
        raise ValueError(f"Training target IDs have {duplicates} conflicting owners")
    db.execute("CREATE INDEX owner_target_idx ON owners(target_id)")

    def export(query, name):
        target = a.output / name
        db.execute(f"COPY ({query}) TO {sql(target)} (FORMAT PARQUET, COMPRESSION ZSTD)")
        rows = db.execute(f"SELECT count(*) FROM read_parquet({sql(target)})").fetchone()[0]
        print(json.dumps({"prepared": name, "rows": rows}), flush=True)
        return rows

    export("SELECT * FROM roles ORDER BY entity_id", "roles.parquet")
    export("SELECT * FROM truth ORDER BY s1_id", "truth.parquet")
    export("SELECT * FROM owners ORDER BY target_id", "owners.parquet")
    export("""WITH chosen AS (
      SELECT o.*, row_number() OVER(PARTITION BY s1_id ORDER BY md5(target_id || '20260927')) AS rank
      FROM owners o JOIN roles r ON o.s1_id=r.entity_id WHERE r.role='encoder')
      SELECT c.s1_id, c.target_id, s.country, s.name_group,
        s.business_name || ' | ' || s.business_address || ' | ' || s.country AS source_text,
        t.business_name || ' | ' || t.business_address || ' | ' || t.country AS target_text
      FROM chosen c JOIN train_s1 s ON c.s1_id=s.entity_id
      JOIN train_targets t ON c.target_id=t.entity_id WHERE c.rank=1
      ORDER BY md5(c.s1_id || '20260927') LIMIT """ + str(a.max_encoder_pairs), "encoder_pairs.parquet")
    files = {}
    for split in ("train", "test"):
        if not (a.dataset / split).is_dir():
            continue
        for side in ("s1", "targets"):
            countries = db.execute(f"SELECT DISTINCT country FROM {split}_{side} ORDER BY 1").fetchall()
            for (country,) in countries:
                if not re.fullmatch(r"[\w-]+", country):
                    raise ValueError(f"Country needs a safe filename mapping: {country!r}")
                name = f"{split}_{side}_{country}.parquet"
                extra = ", r.role, s.name_group" if split == "train" and side == "s1" else ""
                join = " LEFT JOIN roles r USING(entity_id)" if extra else ""
                query = f"""SELECT s.entity_id, s.business_name, s.business_address, s.country,
                  s.business_name || ' | ' || s.business_address || ' | ' || s.country AS text{extra}
                  FROM {split}_{side} s{join} WHERE s.country={sql(country)} ORDER BY s.entity_id"""
                files[name] = export(query, name)
    groups_overlap = db.execute("SELECT count(*) FROM (SELECT name_group FROM roles GROUP BY 1 HAVING count(DISTINCT role)>1)").fetchone()[0]
    if groups_overlap:
        raise ValueError("Name groups overlap split roles")
    metadata = {"seed": SEED, "split_counts": db.execute("SELECT role,country,count(*) FROM roles GROUP BY 1,2 ORDER BY 1,2").fetchall(),
                "historical_ids_sha256": hashlib.sha256(a.previous_ids.read_bytes()).hexdigest(),
                "historical_groups_excluded_from_evaluation": db.execute("SELECT count(*) FROM previous_groups").fetchone()[0],
                "files": files, "reserve_opened": False, "elapsed_seconds": time.monotonic()-started,
                "duckdb_version": duckdb.__version__}
    (a.output / "complete.json").write_text(json.dumps(metadata, indent=2)+"\n")
    db.close()


if __name__ == "__main__":
    main()
