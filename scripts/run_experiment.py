"""Jalankan satu konfigurasi: (stream, detektor, keterlambatan label).

Protokol (sama untuk semua detektor, agar adil):
- Model dasar (HistGradientBoosting, klasifikasi biner serangan vs benign) dilatih pada
  `init` sampel pertama.
- Setiap sampel diprediksi saat tiba. Label sampel j baru tersedia pada langkah j + delay.
- Detektor berbasis error menerima error_j = [pred_j != y_j] saat label j tiba, dan hanya
  untuk prediksi yang dibuat oleh model yang sedang aktif.
- Detektor tanpa label menerima skor model (probabilitas serangan) atau fitur mentah saat itu juga.
- Saat alarm pada langkah t_a: sistem mengumpulkan `collect` sampel baru, lalu melatih ulang
  setelah label-labelnya tiba, yaitu pada t_r = t_a + collect + delay, memakai sampel
  [t_a, t_a + collect). Selama menunggu, alarm lain diabaikan. Setelah latih ulang, detektor direset.
- Baseline: none (tanpa adaptasi), periodic (latih ulang tiap `period` sampel), oracle
  (alarm tepat di awal setiap wilayah drift sebenarnya).

Keluaran: results/runs/<stream>__<detektor>__d<delay>.json
"""
import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from river import drift
from river.drift import binary
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score

from common import RESULTS, STREAMS

META_COLS = {"label", "category", "t", "src", "regime"}

ERROR_BASED = {
    "ADWIN": lambda: drift.ADWIN(),
    "DDM": lambda: binary.DDM(),
    "EDDM": lambda: binary.EDDM(),
    "HDDM_A": lambda: binary.HDDM_A(),
    "HDDM_W": lambda: binary.HDDM_W(),
    "PageHinkley": lambda: drift.PageHinkley(),
}
SCORE_BASED = {  # tanpa label, memantau skor model
    "KSWIN-score": lambda: drift.KSWIN(seed=1),
    "ADWIN-score": lambda: drift.ADWIN(),
    "PH-score": lambda: drift.PageHinkley(),
}
FEATURE_BASED = {"KS-feat"}  # tanpa label, memantau distribusi fitur
BASELINES = {"none", "periodic", "oracle"}
ALL_DETECTORS = list(ERROR_BASED) + list(SCORE_BASED) + sorted(FEATURE_BASED) + sorted(BASELINES)


@dataclass
class Config:
    init: int = 10_000          # sampel awal untuk melatih model pertama
    collect: int = 5_000        # sampel yang dikumpulkan setelah alarm untuk latih ulang
    period: int = 25_000        # interval baseline periodic
    block: int = 5_000          # ukuran blok prediksi
    eval_window: int = 1_000    # jendela evaluasi prequential
    tol: int = 10_000           # toleransi setelah akhir wilayah drift agar alarm dihitung benar
    ks_k: int = 5               # jumlah fitur yang dipantau KS-feat
    ks_test: int = 2_000        # ukuran jendela uji KS-feat
    ks_every: int = 500         # frekuensi uji KS-feat
    ks_alpha: float = 1e-3      # ambang p-value (sebelum koreksi Bonferroni)
    ks_min_d: float = 0.2       # ambang ukuran efek statistik KS
    seed: int = 0
    label_budget: int = 0       # 0 = tanpa batas; mis. 15_000 = maksimal 3 latih ulang


def make_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=100, random_state=seed)


def fit(X, y, seed):
    m = make_model(seed)
    if len(np.unique(y)) < 2:  # jaga-jaga: satu kelas saja
        return None
    return m.fit(X, y)


def run(stream: str, detector: str, delay: int, cfg: Config) -> dict:
    df = pd.read_parquet(STREAMS / f"{stream}.parquet")
    meta = json.loads((STREAMS / f"{stream}.json").read_text())
    feats = [c for c in df.columns if c not in META_COLS]
    X = df[feats].to_numpy(np.float32)
    y = df["label"].to_numpy(np.int8)
    n = len(y)
    regions = meta["drift_regions"]
    starts = [s for s, _ in regions]

    t_start = time.time()
    model = fit(X[: cfg.init], y[: cfg.init], cfg.seed)
    model_start = cfg.init
    pred = np.zeros(n, dtype=np.int8)
    score = np.zeros(n, dtype=np.float32)
    filled = cfg.init

    def ensure(i):
        nonlocal filled
        if i >= filled:
            hi = min(n, filled + cfg.block)
            score[filled:hi] = model.predict_proba(X[filled:hi])[:, 1]
            pred[filled:hi] = (score[filled:hi] >= 0.5)
            filled = hi

    # Fitur yang dipantau KS-feat: k fitur dengan mutual information tertinggi di jendela awal
    top = None
    ref = None
    if detector in FEATURE_BASED:
        mi = mutual_info_classif(X[: cfg.init], y[: cfg.init], random_state=cfg.seed)
        top = np.argsort(mi)[::-1][: cfg.ks_k]
        ref = X[: cfg.init, top]

    def new_det():
        if detector in ERROR_BASED:
            return ERROR_BASED[detector]()
        if detector in SCORE_BASED:
            return SCORE_BASED[detector]()
        return None

    det = new_det()
    alarms: list[int] = []
    retrains: list[int] = []
    pending_retrain = None   # (t_r, t_a)
    retrain_seconds = 0.0
    det_seconds = 0.0
    oracle_left = list(starts)
    planned = 0
    budget_exhausted = False

    for i in range(cfg.init, n):
        # 1) latih ulang jika jadwalnya tiba
        if pending_retrain is not None and i >= pending_retrain[0]:
            t_a = pending_retrain[1]
            lo, hi = t_a, min(t_a + cfg.collect, n)
            r0 = time.time()
            m = fit(X[lo:hi], y[lo:hi], cfg.seed)
            retrain_seconds += time.time() - r0
            if m is not None:
                model = m
                model_start = i
                filled = i
                retrains.append(i)
                det = new_det()
                if ref is not None:
                    ref = X[lo:hi, top]
            pending_retrain = None

        ensure(i)
        alarm = False

        # 2) perbarui detektor (dilewati selama menunggu latih ulang atau jika anggaran label habis)
        if pending_retrain is None and not budget_exhausted:
            d0 = time.time()
            if detector in ERROR_BASED:
                j = i - delay
                if j >= model_start:
                    det.update(int(pred[j] != y[j]))
                    alarm = det.drift_detected
            elif detector in SCORE_BASED:
                det.update(float(score[i]))
                alarm = det.drift_detected
            elif detector in FEATURE_BASED:
                if (i - model_start) >= cfg.ks_test and (i % cfg.ks_every == 0):
                    win = X[i - cfg.ks_test: i, top]
                    for f in range(len(top)):
                        res = ks_2samp(ref[:, f], win[:, f])
                        if res.pvalue < cfg.ks_alpha / len(top) and res.statistic > cfg.ks_min_d:
                            alarm = True
                            break
            elif detector == "periodic":
                alarm = (i - cfg.init) > 0 and (i - cfg.init) % cfg.period == 0
            elif detector == "oracle":
                if oracle_left and i >= oracle_left[0]:
                    oracle_left.pop(0)
                    alarm = True
            det_seconds += time.time() - d0

        if alarm:
            alarms.append(i)
            pending_retrain = (i + cfg.collect + delay, i)
            planned += 1
            # Anggaran label: setiap latih ulang memakai `collect` label. Jika latih ulang
            # berikutnya akan melebihi anggaran, detektor dimatikan sampai akhir stream.
            if cfg.label_budget and (planned + 1) * cfg.collect > cfg.label_budget:
                budget_exhausted = True

    total_seconds = time.time() - t_start
    return evaluate(stream, detector, delay, cfg, meta, y, pred, alarms, retrains,
                    total_seconds, retrain_seconds, det_seconds)


def evaluate(stream, detector, delay, cfg, meta, y, pred, alarms, retrains,
             total_seconds, retrain_seconds, det_seconds) -> dict:
    n = len(y)
    lo = cfg.init
    yt, yp = y[lo:], pred[lo:]
    tp = int(((yt == 1) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())

    # Jejak prequential per jendela
    W = cfg.eval_window
    edges = list(range(lo, n, W))
    trace_f1 = []
    for a in edges:
        b = min(a + W, n)
        trace_f1.append(float(f1_score(y[a:b], pred[a:b], zero_division=0)))
    trace_f1 = np.array(trace_f1)
    win_start = np.array(edges)

    regions = meta["drift_regions"]
    alarms_arr = np.array(alarms, dtype=int)
    used = set()
    per_region = []
    for k, (s, e) in enumerate(regions):
        nxt = regions[k + 1][0] if k + 1 < len(regions) else n
        hit_hi = min(e + cfg.tol, nxt)
        hits = alarms_arr[(alarms_arr >= s) & (alarms_arr < hit_hi)]
        first = int(hits[0]) if len(hits) else None
        if first is not None:
            used.add(first)
        # performa sebelum drift (10 jendela terakhir sebelum s)
        pre = trace_f1[(win_start >= s - 10 * W) & (win_start < s)]
        base = float(pre.mean()) if len(pre) else float("nan")
        seg_mask = (win_start >= s) & (win_start < nxt)
        seg = trace_f1[seg_mask]
        seg_start = win_start[seg_mask]
        loss = float(np.clip(base - seg, 0, None).sum() * W) if len(seg) else float("nan")
        # waktu pulih: jendela pertama setelah s di mana rata-rata 3 jendela >= 0.9 * base,
        # setelah sebelumnya sempat turun di bawahnya
        rec = None
        if len(seg) >= 3 and not np.isnan(base):
            roll = np.convolve(seg, np.ones(3) / 3, mode="valid")
            below = roll < 0.9 * base
            if not below.any():
                rec = 0
            else:
                first_below = int(np.argmax(below))
                after = np.flatnonzero(~below[first_below:])
                rec = int(seg_start[first_below + after[0]] - s) if len(after) else None
        per_region.append(dict(start=s, end=e, detected=first is not None,
                               delay=(first - s) if first is not None else None,
                               pre_f1=base, min_f1=float(seg.min()) if len(seg) else None,
                               loss_area=loss, recovery=rec))
    false_alarms = int(sum(1 for a in alarms if a not in used))
    n_eval = n - lo

    return dict(
        stream=stream, scenario=meta["scenario"], width=meta.get("width"),
        domains=meta["domains"], seed_stream=meta["seed"], detector=detector,
        family=("error" if detector in ERROR_BASED else "score" if detector in SCORE_BASED
                else "feature" if detector in FEATURE_BASED else "baseline"),
        delay=delay, label_budget=cfg.label_budget, n=n,
        f1_attack=float(f1_score(yt, yp, zero_division=0)),
        f1_macro=float(f1_score(yt, yp, average="macro", zero_division=0)),
        recall=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1),
        n_alarms=len(alarms), false_alarms=false_alarms,
        false_alarms_per_10k=false_alarms / n_eval * 1e4,
        n_retrains=len(retrains), labels_used=len(retrains) * cfg.collect,
        regions=per_region,
        mean_detection_delay=float(np.mean([r["delay"] for r in per_region if r["delay"] is not None]))
        if any(r["delay"] is not None for r in per_region) else None,
        detection_rate=float(np.mean([r["detected"] for r in per_region])) if per_region else None,
        total_loss_area=float(np.nansum([r["loss_area"] for r in per_region])),
        alarms=alarms, retrains=retrains,
        trace_f1=trace_f1.round(4).tolist(), trace_start=win_start.tolist(),
        seconds_total=round(total_seconds, 2), seconds_retrain=round(retrain_seconds, 2),
        us_per_sample_detector=round(det_seconds / max(n_eval, 1) * 1e6, 2),
        config=asdict(cfg),
    )


def out_path(stream, detector, delay, budget=0):
    tag = f"__b{budget}" if budget else ""
    return RESULTS / "runs" / f"{stream}__{detector}__d{delay}{tag}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stream")
    ap.add_argument("detector", choices=ALL_DETECTORS)
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--budget", type=int, default=0, help="anggaran label (0 = tanpa batas)")
    a = ap.parse_args()
    res = run(a.stream, a.detector, a.delay, Config(label_budget=a.budget))
    p = out_path(a.stream, a.detector, a.delay, a.budget)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    keys = ["f1_attack", "f1_macro", "n_alarms", "false_alarms", "n_retrains",
            "mean_detection_delay", "total_loss_area", "seconds_total"]
    print({k: res[k] for k in keys})
    for r in res["regions"]:
        print("  region", r)


if __name__ == "__main__":
    main()
