"""Praproses satu dataset NetFlow v3 menjadi file parquet yang ringkas dan urut waktu.

Langkah:
1. Pass 1: hitung jumlah baris per kategori (hanya membaca kolom Attack).
2. Tentukan laju sampling per kategori agar tiap kategori maksimal --cap baris.
   Sampling acak seragam di sepanjang file, sehingga sebaran waktunya tetap terjaga.
3. Pass 2: baca per potongan, petakan label, buang kolom identitas, ambil sampel.
4. Urutkan berdasarkan FLOW_START_MILLISECONDS dan simpan ke data/processed/<nama>.parquet.

Pemakaian:
    .venv/bin/python scripts/preprocess.py NF-ToN-IoT-v3 --cap 150000
    .venv/bin/python scripts/preprocess.py all
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from common import (ALL_CATEGORIES, DATASETS, EXCLUDE, PROCESSED, RAW,
                    SHARED_CATEGORIES, TIME_COL, clean_features, map_attack)

CHUNK = 1_000_000


def preprocess(name: str, cap: int, categories: list[str], seed: int) -> dict:
    t0 = time.time()
    path = RAW / f"{name}.csv"
    rng = np.random.default_rng(seed)

    # Pass 1: hitung per kategori
    counts: dict[str, int] = {}
    for ch in pd.read_csv(path, usecols=["Attack"], chunksize=CHUNK):
        cat = ch["Attack"].map(map_attack)
        for k, v in cat.value_counts().items():
            counts[k] = counts.get(k, 0) + int(v)
    rates = {c: min(1.0, cap / counts[c]) for c in categories if counts.get(c, 0) > 0}

    # Pass 2: ambil sampel
    header = pd.read_csv(path, nrows=0).columns.str.strip()
    features = [c for c in header if c not in EXCLUDE]
    parts = []
    for ch in pd.read_csv(path, chunksize=CHUNK):
        ch.columns = ch.columns.str.strip()
        cat = ch["Attack"].map(map_attack)
        keep = cat.isin(rates.keys()).to_numpy()
        u = rng.random(len(ch))
        thr = cat.map(rates).fillna(0).to_numpy()
        keep &= u < thr
        sub = ch.loc[keep, features].astype("float32")
        sub["category"] = cat[keep].to_numpy()
        sub["label"] = (sub["category"] != "Benign").astype("int8")
        sub["t"] = ch.loc[keep, TIME_COL].astype("int64").to_numpy()
        parts.append(sub)

    df = pd.concat(parts, ignore_index=True)
    cleaned = clean_features(df, features)
    df = df.sort_values("t", kind="stable").reset_index(drop=True)
    df["category"] = df["category"].astype("category")
    df["src"] = name

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"{name}.parquet"
    df.to_parquet(out, index=False)

    info = {
        "dataset": name,
        "rows_out": len(df),
        "features": features,
        "n_features": len(features),
        "counts_full": counts,
        "sampling_rate": rates,
        "counts_out": df["category"].value_counts().to_dict(),
        "time_span_hours": float((df["t"].max() - df["t"].min()) / 3.6e6),
        "non_finite_replaced_with_0": cleaned,
        "cap": cap,
        "seed": seed,
        "seconds": round(time.time() - t0, 1),
    }
    (PROCESSED / f"{name}.json").write_text(json.dumps(info, indent=2, default=str))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="nama dataset (mis. NF-ToN-IoT-v3) atau 'all'")
    ap.add_argument("--cap", type=int, default=150_000, help="maksimal baris per kategori")
    ap.add_argument("--all-categories", action="store_true",
                    help="pakai 7 kategori, bukan hanya 4 kategori bersama")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cats = ALL_CATEGORIES if args.all_categories else SHARED_CATEGORIES
    names = DATASETS if args.dataset == "all" else [args.dataset]
    for n in names:
        info = preprocess(n, args.cap, cats, args.seed)
        print(f"{n}: {info['rows_out']:,} baris, {info['n_features']} fitur, "
              f"{info['counts_out']}, {info['seconds']} detik")


if __name__ == "__main__":
    main()
