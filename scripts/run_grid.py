"""Jalankan banyak konfigurasi secara paralel, lalu buat tabel ringkasan.

Pemakaian:
    # semua stream di data/streams, semua detektor, empat nilai keterlambatan label
    .venv/bin/python scripts/run_grid.py --delays 0 1000 5000 20000 --jobs 8

    # hanya stream tertentu (pola glob) dan detektor tertentu
    .venv/bin/python scripts/run_grid.py --streams "S2-cross-W5000*" --detectors DDM KS-feat none

    # hanya membuat ulang ringkasan dari hasil yang sudah ada
    .venv/bin/python scripts/run_grid.py --summary-only

Konfigurasi yang hasilnya sudah ada akan dilewati (pakai --force untuk mengulang).
Ringkasan: results/summary_runs.csv (satu baris per run) dan
           results/summary_regions.csv (satu baris per wilayah drift per run).
"""
import os

# Setiap proses paralel cukup satu thread; tanpa ini HistGradientBoosting di 8 proses
# masing-masing memakai semua core dan saling berebut CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from common import RESULTS, STREAMS
from run_experiment import ALL_DETECTORS, Config, out_path, run


def job(args):
    stream, det, delay, budget = args
    warnings.filterwarnings("ignore")
    res = run(stream, det, delay, Config(label_budget=budget))
    p = out_path(stream, det, delay, budget)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    return stream, det, delay, res["f1_attack"], res["seconds_total"]


def summarize() -> None:
    rows, regs = [], []
    drop = {"regions", "alarms", "retrains", "trace_f1", "trace_start", "config", "domains"}
    for p in sorted((RESULTS / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        base = {k: v for k, v in r.items() if k not in drop}
        base["domains"] = "-".join(r["domains"])
        base.setdefault("label_budget", 0)  # hasil lama sebelum fitur anggaran label
        rows.append(base)
        for k, g in enumerate(r["regions"]):
            g = dict(g)
            g["detection_delay"] = g.pop("delay")  # jangan menimpa kolom delay (keterlambatan label)
            regs.append({**{c: base[c] for c in ("stream", "scenario", "width", "domains",
                                                  "seed_stream", "detector", "family", "delay",
                                                  "label_budget")},
                         "region": k, **g})
    if not rows:
        print("Belum ada hasil.")
        return
    pd.DataFrame(rows).to_csv(RESULTS / "summary_runs.csv", index=False)
    pd.DataFrame(regs).to_csv(RESULTS / "summary_regions.csv", index=False)
    df = pd.DataFrame(rows)
    view = (df.groupby(["label_budget", "scenario", "detector", "delay"])
              [["f1_macro", "detection_rate", "mean_detection_delay", "false_alarms_per_10k",
                "n_retrains", "total_loss_area"]]
              .mean().round(3))
    print(view.to_string())
    print(f"\n{len(df)} run -> results/summary_runs.csv, results/summary_regions.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="*", help="pola glob nama stream")
    ap.add_argument("--detectors", nargs="+", default=ALL_DETECTORS)
    ap.add_argument("--delays", type=int, nargs="+", default=[0, 1000, 5000, 20000])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=0,
                    help="anggaran label per stream (0 = tanpa batas; 15000 = maks. 3 latih ulang)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()

    if not a.summary_only:
        streams = sorted(p.stem for p in STREAMS.glob(f"{a.streams}.parquet"))
        todo = [(s, d, dl, a.budget) for s in streams for d in a.detectors for dl in a.delays
                if a.force or not out_path(s, d, dl, a.budget).exists()]
        # baseline 'none' tidak bergantung pada delay maupun anggaran: cukup sekali per stream
        todo = [t for t in todo
                if not (t[1] == "none" and (t[2] != a.delays[0] or a.budget))]
        print(f"{len(streams)} stream, {len(todo)} run akan dijalankan dengan {a.jobs} proses")
        done = 0
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(job, t) for t in todo]
            for f in as_completed(futs):
                s, d, dl, f1, sec = f.result()
                done += 1
                print(f"[{done}/{len(todo)}] {s} {d} d={dl} F1={f1:.3f} ({sec}s)", flush=True)
    summarize()


if __name__ == "__main__":
    main()
