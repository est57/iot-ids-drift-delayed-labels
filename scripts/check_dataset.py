"""Cek cepat file NetFlow v3: kolom, kolom temporal, distribusi label, dan urutan waktu.

Pemakaian:
    python3 scripts/check_dataset.py data/raw/NF-ToN-IoT-v3.csv

Dibaca per potongan (chunk) agar tidak kehabisan memori pada file besar.
Butuh: pip install pandas
"""
import sys
from collections import Counter

import pandas as pd

CHUNK = 1_000_000
TIME_HINTS = ("MILLISECONDS", "IAT", "TIME", "START", "END")


def main(path: str) -> None:
    head = pd.read_csv(path, nrows=5)
    cols = list(head.columns)
    print(f"File   : {path}")
    print(f"Kolom  : {len(cols)}")
    for c in cols:
        print(f"  - {c}")

    time_cols = [c for c in cols if any(h in c.upper() for h in TIME_HINTS)]
    print(f"\nKolom temporal terdeteksi ({len(time_cols)}): {time_cols}")

    label_cols = [c for c in cols if c.lower() in ("attack", "label")]
    start_col = next((c for c in cols if c.upper() == "FLOW_START_MILLISECONDS"), None)

    counts = {c: Counter() for c in label_cols}
    rows = 0
    unsorted_pairs = 0
    prev_last = None
    t_min, t_max = None, None

    usecols = label_cols + ([start_col] if start_col else [])
    for chunk in pd.read_csv(path, usecols=usecols or None, chunksize=CHUNK):
        rows += len(chunk)
        for c in label_cols:
            counts[c].update(chunk[c].astype(str).tolist())
        if start_col:
            s = chunk[start_col]
            unsorted_pairs += int((s.diff() < 0).sum())
            if prev_last is not None and s.iloc[0] < prev_last:
                unsorted_pairs += 1
            prev_last = s.iloc[-1]
            t_min = s.min() if t_min is None else min(t_min, s.min())
            t_max = s.max() if t_max is None else max(t_max, s.max())

    print(f"\nJumlah baris: {rows:,}")
    for c, cnt in counts.items():
        print(f"\nDistribusi '{c}':")
        for k, v in cnt.most_common():
            print(f"  {k:<20} {v:>12,}  ({v / rows:.2%})")

    if start_col:
        span_h = (t_max - t_min) / 3_600_000
        print(f"\nRentang waktu: {span_h:,.1f} jam")
        print(f"Pasangan baris yang tidak urut waktu: {unsorted_pairs:,}")
        print("-> Urutkan berdasarkan FLOW_START_MILLISECONDS sebelum menyusun stream."
              if unsorted_pairs else "-> File sudah urut waktu.")
    else:
        print("\nPERINGATAN: kolom FLOW_START_MILLISECONDS tidak ditemukan; cek daftar kolom di atas.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
