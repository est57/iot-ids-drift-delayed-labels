"""Susun stream berurutan untuk tiga keluarga skenario drift.

S1 synthetic : satu domain (default ToN), drift dibuat dengan mengalikan subset fitur
               (perturbasi kovariat) pada tiap segmen baru.
S2 cross     : beberapa domain berurutan (mis. ToN -> BoT -> UNSW). Pada setiap batas,
               peluang sampel berasal dari domain baru naik linear dari 0 ke 1 sepanjang
               lebar transisi W (W=0 berarti pergantian mendadak).
S3 novel     : satu domain; segmen kedua memunculkan kelas serangan yang sebelumnya tidak ada.

Di dalam setiap domain, urutan waktu asli (FLOW_START_MILLISECONDS) dipertahankan.
Setiap stream disimpan sebagai data/streams/<nama>.parquet + <nama>.json (metadata,
termasuk wilayah drift sebenarnya untuk evaluasi).

Pemakaian:
    .venv/bin/python scripts/build_streams.py preset --seeds 1 2 3
    .venv/bin/python scripts/build_streams.py cross --domains ToN BoT UNSW --width 20000
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import DATASETS, PROCESSED, SHORT, STREAMS

LONG = {v: k for k, v in SHORT.items()}
META_COLS = ["label", "category", "t", "src"]


def load_domain(short: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{LONG[short]}.parquet")


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def draw(df: pd.DataFrame, n: int, attack_ratio: float, rng: np.random.Generator,
         attack_cats: list[str] | None = None, warnings: list | None = None,
         tag: str = "") -> pd.DataFrame:
    """Ambil n baris (benign + serangan sesuai rasio), lalu urutkan kembali menurut waktu."""
    ben = np.flatnonzero((df["category"] == "Benign").to_numpy())
    att_mask = df["label"].to_numpy() == 1
    if attack_cats is not None:
        att_mask &= df["category"].isin(attack_cats).to_numpy()
    att = np.flatnonzero(att_mask)
    n_att = int(round(n * attack_ratio))
    n_ben = n - n_att
    idx = []
    for pool, k, what in ((ben, n_ben, "benign"), (att, n_att, "attack")):
        replace = k > len(pool)
        if replace and warnings is not None:
            warnings.append(f"{tag}: {what} butuh {k:,} tetapi tersedia {len(pool):,}; "
                            "sampling dengan pengembalian")
        idx.append(rng.choice(pool, size=k, replace=replace))
    idx = np.concatenate(idx)
    # ORDER="shuffle" (default): urutan diacak di dalam domain, sehingga satu-satunya drift
    #   adalah yang dirancang di batas segmen. Diperlukan karena ToN-IoT dan BoT-IoT
    #   menjalankan serangan bergiliran menurut waktu (Recon -> DoS -> DDoS).
    # ORDER="time": urutan waktu asli dipertahankan (drift alami di dalam dataset ikut masuk).
    idx = np.sort(idx) if ORDER == "time" else rng.permutation(idx)
    return df.iloc[idx].reset_index(drop=True)


ORDER = "shuffle"


def suffix() -> str:
    return "_time" if ORDER == "time" else ""


def schedule(n_regimes: int, seg_len: int, width: int, rng: np.random.Generator):
    """Tentukan regime (0..n_regimes-1) tiap posisi stream dan wilayah drift sebenarnya."""
    n = n_regimes * seg_len
    regime = np.zeros(n, dtype=np.int16)
    regions = []
    for k in range(1, n_regimes):
        b = k * seg_len
        regime[b:] = k
        if width > 0:
            pos = np.arange(b, min(b + width, n))
            p_new = (pos - b + 1) / (width + 1)
            regime[pos] = np.where(rng.random(len(pos)) < p_new, k, k - 1)
        regions.append([int(b), int(min(b + width, n))])
    return regime, regions


def compose(pools: list[pd.DataFrame], regime: np.ndarray) -> pd.DataFrame:
    """Ambil baris berikutnya dari pool regime yang sesuai pada setiap posisi stream."""
    parts = []
    for r, pool in enumerate(pools):
        pos = np.flatnonzero(regime == r)
        parts.append(pool.iloc[: len(pos)].assign(_pos=pos))
    df = pd.concat(parts).sort_values("_pos").drop(columns="_pos").reset_index(drop=True)
    df["regime"] = regime
    return df


def save(df: pd.DataFrame, name: str, meta: dict) -> None:
    STREAMS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STREAMS / f"{name}.parquet", index=False)
    meta.update(name=name, n=len(df), attack_ratio_actual=float(df["label"].mean()),
                category_counts=df["category"].value_counts().to_dict())
    (STREAMS / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"  {name}: {len(df):,} baris, drift di {meta['drift_regions']}"
          + (f"  [PERINGATAN: {len(meta['warnings'])}]" if meta.get("warnings") else ""))


def build_cross(domains, seg_len, width, attack_ratio, seed):
    rng = np.random.default_rng(seed)
    warns: list[str] = []
    regime, regions = schedule(len(domains), seg_len, width, rng)
    need = [int((regime == k).sum()) for k in range(len(domains))]
    pools = [draw(load_domain(d), need[k], attack_ratio, rng, warnings=warns, tag=d)
             for k, d in enumerate(domains)]
    df = compose(pools, regime)
    name = f"S2-cross-W{width}_{'-'.join(domains)}_s{seed}{suffix()}"
    save(df, name, dict(scenario="S2_cross", order=ORDER, domains=domains, seg_len=seg_len, width=width,
                        attack_ratio=attack_ratio, seed=seed, drift_regions=regions,
                        warnings=warns))


def build_synthetic(domain, n_regimes, seg_len, width, attack_ratio, seed,
                    n_feat=10, factor=(1.5, 3.0)):
    rng = np.random.default_rng(seed)
    warns: list[str] = []
    regime, regions = schedule(n_regimes, seg_len, width, rng)
    base = draw(load_domain(domain), len(regime), attack_ratio, rng, warnings=warns, tag=domain)
    feats = [c for c in feature_cols(base) if base[c].std() > 0]
    perturb = [{}]  # regime 0 = data asli
    for _ in range(1, n_regimes):
        chosen = rng.choice(feats, size=min(n_feat, len(feats)), replace=False)
        perturb.append({c: float(rng.uniform(*factor)) for c in chosen})
    df = base.copy()
    df["regime"] = regime
    for k, p in enumerate(perturb):
        m = regime == k
        for c, f in p.items():
            df.loc[m, c] = df.loc[m, c] * f
    tag = "abrupt" if width == 0 else f"W{width}"
    name = f"S1-synth-{tag}_{domain}_s{seed}{suffix()}"
    save(df, name, dict(scenario="S1_synthetic", order=ORDER, domains=[domain], seg_len=seg_len,
                        width=width, attack_ratio=attack_ratio, seed=seed,
                        drift_regions=regions, perturbations=perturb, warnings=warns))


def build_novel(domain, seg_len, width, attack_ratio, seed, novel="Recon",
                known=("DoS", "DDoS")):
    rng = np.random.default_rng(seed)
    warns: list[str] = []
    regime, regions = schedule(2, seg_len, width, rng)
    src = load_domain(domain)
    need = [int((regime == k).sum()) for k in range(2)]
    p0 = draw(src, need[0], attack_ratio, rng, attack_cats=list(known), warnings=warns, tag=domain)
    p1 = draw(src, need[1], attack_ratio, rng, attack_cats=list(known) + [novel],
              warnings=warns, tag=domain)
    df = compose([p0, p1], regime)
    name = f"S3-novel-{novel}_{domain}_s{seed}{suffix()}"
    save(df, name, dict(scenario="S3_novel", order=ORDER, domains=[domain], seg_len=seg_len, width=width,
                        attack_ratio=attack_ratio, seed=seed, novel_class=novel,
                        known_classes=list(known), drift_regions=regions, warnings=warns))


def preset(seeds, seg_len, attack_ratio, widths):
    orders = [["ToN", "BoT", "UNSW"], ["UNSW", "ToN", "BoT"], ["BoT", "UNSW", "ToN"]]
    for s in seeds:
        print(f"seed {s}")
        for w in widths:
            build_synthetic("ToN", 3, seg_len, w, attack_ratio, s)
            for o in orders:
                build_cross(o, seg_len, w, attack_ratio, s)
        build_novel("ToN", seg_len, 0, attack_ratio, s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["preset", "cross", "synthetic", "novel"])
    ap.add_argument("--domains", nargs="+", default=["ToN", "BoT", "UNSW"])
    ap.add_argument("--seg-len", type=int, default=50_000)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--widths", type=int, nargs="+", default=[0, 5_000, 20_000])
    ap.add_argument("--attack-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--order", choices=["shuffle", "time"], default="shuffle",
                    help="shuffle = acak di dalam domain (default); time = urutan waktu asli")
    a = ap.parse_args()
    global ORDER
    ORDER = a.order

    if a.mode == "preset":
        preset(a.seeds, a.seg_len, a.attack_ratio, a.widths)
    elif a.mode == "cross":
        build_cross(a.domains, a.seg_len, a.width, a.attack_ratio, a.seed)
    elif a.mode == "synthetic":
        build_synthetic(a.domains[0], 3, a.seg_len, a.width, a.attack_ratio, a.seed)
    else:
        build_novel(a.domains[0], a.seg_len, a.width, a.attack_ratio, a.seed)


if __name__ == "__main__":
    main()
