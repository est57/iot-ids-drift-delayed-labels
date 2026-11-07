"""Analisis hasil grid: tabel, uji Friedman + Nemenyi, korelasi peringkat S1 vs S2, grafik.

Dua kondisi dianalisis terpisah:
- tanpa batas label (label_budget = 0)
- dengan anggaran label (mis. label_budget = 15000, maksimal 3 latih ulang per stream)

Teks pada grafik sengaja berbahasa Inggris agar langsung bisa dipakai di naskah.

Pemakaian:
    .venv/bin/python scripts/analyze.py

Keluaran (sufiks _b<anggaran> untuk kondisi dengan anggaran):
    results/tables/main_table*.csv        rata-rata +- sd per skenario x detektor x delay
    results/tables/family_delay*.csv      jeda deteksi & alarm palsu per keluarga detektor x delay
    results/tables/ranks_<skenario>*.csv  peringkat rata-rata (Friedman) per delay
    results/tables/rank_agreement*.csv    korelasi Spearman peringkat S1 vs S2 per delay
    results/tables/label_efficiency.csv   kenaikan macro-F1 di atas 'none' per 10.000 label
    results/tables/budget_comparison.csv  macro-F1 tanpa batas vs dengan anggaran
    results/figures/*.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, spearmanr, studentized_range  # noqa: E402

from common import RESULTS  # noqa: E402

TABLES = RESULTS / "tables"
FIGS = RESULTS / "figures"
DETECTORS = ["ADWIN", "DDM", "EDDM", "HDDM_A", "HDDM_W", "PageHinkley",
             "KSWIN-score", "ADWIN-score", "PH-score", "KS-feat"]
SCENARIOS = ["S1_synthetic", "S2_cross", "S3_novel"]
SCENARIO_EN = {"S1_synthetic": "S1 synthetic drift", "S2_cross": "S2 cross-dataset drift",
               "S3_novel": "S3 novel attack"}
FAMILY_EN = {"error": "error-based", "score": "label-free (score)",
             "feature": "label-free (feature)", "baseline": "baseline"}
FAMILY_COLOR = {"error": "#1f77b4", "score": "#ff7f0e", "feature": "#2ca02c", "baseline": "#555555"}
MARKERS = {"ADWIN": "o", "DDM": "s", "EDDM": "^", "HDDM_A": "v", "HDDM_W": "D",
           "PageHinkley": "P", "KSWIN-score": "o", "ADWIN-score": "s", "PH-score": "^",
           "KS-feat": "D", "oracle": "*", "periodic": "x", "none": "."}
METRICS = ["f1_macro", "detection_rate", "mean_detection_delay", "false_alarms_per_10k",
           "n_retrains", "labels_used", "total_loss_area"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(RESULTS / "summary_runs.csv")
    regs = pd.read_csv(RESULTS / "summary_regions.csv")
    for d in (runs, regs):
        if "label_budget" not in d:
            d["label_budget"] = 0
        d["label_budget"] = d["label_budget"].fillna(0).astype(int)
    # 'none' tidak bergantung pada delay maupun anggaran: gandakan ke semua kombinasi
    delays = sorted(runs["delay"].unique())
    budgets = sorted(runs["label_budget"].unique())

    def expand(df):
        none = df[(df.detector == "none") & (df.delay == 0) & (df.label_budget == 0)]
        rest = df[df.detector != "none"]
        return pd.concat([rest] + [none.assign(delay=dl, label_budget=b)
                                   for dl in delays for b in budgets], ignore_index=True)
    return expand(runs), expand(regs)


def sfx(budget: int) -> str:
    return f"_b{budget}" if budget else ""


def main_table(runs, budget):
    g = runs.groupby(["scenario", "detector", "delay"])[METRICS]
    t = g.mean().round(3).join(g.std().round(3), rsuffix="_sd")
    t.to_csv(TABLES / f"main_table{sfx(budget)}.csv")
    return t


def family_table(runs, regs, budget):
    r = regs[regs.detector.isin(DETECTORS)]
    out = (r.groupby(["scenario", "family", "delay"])
             .agg(detection_rate=("detected", "mean"),
                  median_detection_delay=("detection_delay", "median"))
             .round(2))
    extra = (runs[runs.detector.isin(DETECTORS)]
             .groupby(["scenario", "family", "delay"])[["false_alarms_per_10k", "f1_macro",
                                                        "labels_used"]].mean().round(3))
    out = out.join(extra)
    out.to_csv(TABLES / f"family_delay{sfx(budget)}.csv")
    return out


def friedman_ranks(runs, scenario, budget, metric="f1_macro"):
    """Peringkat rata-rata detektor (1 = terbaik, macro-F1 tertinggi) per delay + CD Nemenyi.

    total_loss_area TIDAK dipakai untuk peringkat karena bias: loss diukur relatif terhadap F1
    sebelum drift, sehingga detektor yang gagal pulih di segmen sebelumnya tampak kecil
    loss-nya di drift berikutnya.
    """
    res = {}
    for d, g in runs[(runs.scenario == scenario) & runs.detector.isin(DETECTORS)].groupby("delay"):
        piv = g.pivot_table(index="stream", columns="detector", values=metric)[DETECTORS].dropna()
        ranks = piv.rank(axis=1, method="average", ascending=False)
        k, n = piv.shape[1], piv.shape[0]
        _, p = friedmanchisquare(*[piv[c] for c in DETECTORS])
        q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
        cd = q * np.sqrt(k * (k + 1) / (6 * n))
        res[d] = dict(ranks=ranks.mean().sort_values(), p=p, cd=cd, n=n)
    tab = pd.DataFrame({d: v["ranks"] for d, v in res.items()})
    tab.loc["friedman_p"] = [res[d]["p"] for d in tab.columns]
    tab.loc["nemenyi_CD"] = [res[d]["cd"] for d in tab.columns]
    tab.loc["n_streams"] = [res[d]["n"] for d in tab.columns]
    tab.round(4).to_csv(TABLES / f"ranks_{scenario}{sfx(budget)}.csv")
    return res


def rank_agreement(r1, r2, budget):
    rows = []
    for d in sorted(set(r1) & set(r2)):
        rho, p = spearmanr(r1[d]["ranks"][DETECTORS], r2[d]["ranks"][DETECTORS])
        rows.append(dict(delay=d, spearman_rho=round(rho, 3), p=round(p, 4),
                         best_S1=r1[d]["ranks"].index[0], best_S2=r2[d]["ranks"].index[0]))
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / f"rank_agreement{sfx(budget)}.csv", index=False)
    return t


# ---------------------------------------------------------------- grafik (bahasa Inggris)

def fig_metric_vs_delay(runs, scenario, metric, ylabel, fname, budget):
    sub = runs[runs.scenario == scenario]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for det in DETECTORS + ["oracle", "periodic"] + (["none"] if metric == "f1_macro" else []):
        dd = sub[sub.detector == det]
        if dd.empty:
            continue
        g = dd.groupby("delay")[metric].mean()
        fam = dd["family"].iloc[0]
        ax.plot(g.index.astype(str), g.values, "--" if fam == "baseline" else "-",
                marker=MARKERS[det], ms=5, lw=1.2, color=FAMILY_COLOR[fam], alpha=0.85,
                label=det)
    ax.set_xlabel("Label delay (samples)")
    ax.set_ylabel(ylabel)
    cond = f", label budget = {budget:,}" if budget else ", unlimited labels"
    ax.set_title(f"{SCENARIO_EN[scenario]}{cond}", fontsize=11)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    fam_handles = [plt.Line2D([], [], color=c, lw=3, label=FAMILY_EN[f])
                   for f, c in FAMILY_COLOR.items()]
    leg1 = ax.legend(handles, labels, fontsize=7, ncol=2, frameon=False,
                     loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=fam_handles, fontsize=7, frameon=False, loc="lower left",
              bbox_to_anchor=(1.01, 0.0), title="Detector family", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / f"{fname}{sfx(budget)}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_trace(stream, delay, dets, fname, budget=0):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    meta = None
    for det in dets:
        b = 0 if det == "none" else budget
        tag = f"__b{b}" if b else ""
        p = RESULTS / "runs" / f"{stream}__{det}__d{0 if det == 'none' else delay}{tag}.json"
        r = json.loads(p.read_text())
        meta = r
        x = np.array(r["trace_start"]) / 1000
        y = pd.Series(r["trace_f1"]).rolling(3, min_periods=1).mean()
        line, = ax.plot(x, y, lw=1.2, label=det)
        for a in r["alarms"]:
            ax.axvline(a / 1000, color=line.get_color(), lw=0.7, alpha=0.6, ymin=0.92)
    for g in meta["regions"]:
        ax.axvspan(g["start"] / 1000, max(g["end"], g["start"] + 300) / 1000,
                   color="red", alpha=0.12)
    ax.set_xlabel("Stream position (×1,000 samples)")
    ax.set_ylabel("Attack-class F1\n(3-window rolling mean)")
    cond = f", label budget = {budget:,}" if budget else ""
    ax.set_title(f"{stream}: label delay = {delay:,}{cond}\n"
                 "shaded = true drift region, ticks at top = alarms", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, ncol=len(dets), frameon=False, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / f"{fname}{sfx(budget)}.png", dpi=200)
    plt.close(fig)


def fig_budget_comparison(runs, scenario):
    """Macro-F1 tanpa batas vs dengan anggaran, per detektor, untuk tiap delay."""
    sub = runs[(runs.scenario == scenario) & runs.detector.isin(DETECTORS + ["oracle", "periodic"])]
    budgets = sorted(sub.label_budget.unique())
    if len(budgets) < 2:
        return
    b = budgets[-1]
    piv = sub.pivot_table(index=["detector", "delay"], columns="label_budget",
                          values="f1_macro", aggfunc="mean")
    delays = sorted(sub.delay.unique())
    order = DETECTORS + ["oracle", "periodic"]
    lo = max(0.5, float(np.nanmin(piv.to_numpy())) - 0.03)
    ncol = 2
    nrow = int(np.ceil(len(delays) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.6, 4.0 * nrow), sharey=True, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    y = np.arange(len(order))
    for ax, dl in zip(axes, delays):
        un = [piv.loc[(d, dl), 0] for d in order]
        bu = [piv.loc[(d, dl), b] for d in order]
        ax.barh(y - 0.2, un, 0.4, color="#9ecae1", label="unlimited labels")
        ax.barh(y + 0.2, bu, 0.4, color="#3182bd", label=f"label budget = {b:,}")
        ax.set_title(f"label delay = {dl:,}", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, order, fontsize=8)
    for ax in axes[len(delays):]:
        ax.set_visible(False)
    axes[0].set_ylim(len(order) - 0.5, -0.5)  # detektor pertama di atas
    axes[0].set_xlim(lo, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=2, loc="upper right")
    fig.supxlabel("Macro-F1")
    fig.suptitle(f"{SCENARIO_EN[scenario]}: unlimited labels vs. label budget",
                 fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / f"{scenario}_budget_comparison.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- analisis per kondisi

def analyze_condition(runs, regs, budget):
    tag = "TANPA BATAS LABEL" if budget == 0 else f"ANGGARAN LABEL {budget:,}"
    print(f"\n################ {tag} ################")
    main_table(runs, budget)
    fam = family_table(runs, regs, budget)
    print("=== Per keluarga detektor ===")
    print(fam.to_string())

    rk = {s: friedman_ranks(runs, s, budget) for s in SCENARIOS}
    for s, res in rk.items():
        print(f"\n=== Peringkat rata-rata {s} (macro-F1, 1 = terbaik) ===")
        print(pd.DataFrame({d: v["ranks"] for d, v in res.items()}).round(2).to_string())
        print("Friedman p:", {d: f"{v['p']:.2e}" for d, v in res.items()},
              "| CD Nemenyi:", {d: round(v["cd"], 2) for d, v in res.items()})
    agree = rank_agreement(rk["S1_synthetic"], rk["S2_cross"], budget)
    print("\n=== Kesesuaian peringkat S1 vs S2 ===")
    print(agree.to_string(index=False))

    for s in SCENARIOS:
        fig_metric_vs_delay(runs, s, "mean_detection_delay", "Mean detection delay (samples)",
                            f"{s}_detection_delay", budget)
        fig_metric_vs_delay(runs, s, "f1_macro", "Macro-F1 (higher is better)",
                            f"{s}_f1_macro", budget)
        fig_metric_vs_delay(runs, s, "false_alarms_per_10k", "False alarms per 10,000 samples",
                            f"{s}_false_alarms", budget)
        fig_metric_vs_delay(runs, s, "labels_used", "Labels used for retraining",
                            f"{s}_labels_used", budget)
    fig_trace("S2-cross-W5000_ToN-BoT-UNSW_s1", 5000, ["none", "oracle", "ADWIN", "KS-feat"],
              "trace_S2_ToN-BoT-UNSW_d5000", budget)


def label_efficiency(runs):
    """Kenaikan macro-F1 di atas 'none' per 10.000 label (kondisi tanpa batas)."""
    r = runs[runs.label_budget == 0]
    base = (r[r.detector == "none"].groupby(["stream", "delay"])["f1_macro"].first()
            .rename("f1_none"))
    x = r[r.detector.isin(DETECTORS + ["oracle", "periodic"])].join(base, on=["stream", "delay"])
    x["gain_per_10k_labels"] = (x.f1_macro - x.f1_none) / (x.labels_used / 1e4).replace(0, np.nan)
    t = (x.groupby(["scenario", "detector", "delay"])
           [["f1_macro", "labels_used", "gain_per_10k_labels"]].mean().round(4))
    t.to_csv(TABLES / "label_efficiency.csv")
    return t


def budget_comparison(runs):
    budgets = sorted(runs.label_budget.unique())
    if len(budgets) < 2:
        return None
    b = budgets[-1]
    t = (runs[runs.detector.isin(DETECTORS + ["oracle", "periodic"])]
         .pivot_table(index=["scenario", "detector", "delay"], columns="label_budget",
                      values=["f1_macro", "labels_used"], aggfunc="mean").round(4))
    t.columns = [f"{m}_{'unlimited' if bb == 0 else f'b{bb}'}" for m, bb in t.columns]
    t[f"f1_change_b{b}"] = (t[f"f1_macro_b{b}"] - t["f1_macro_unlimited"]).round(4)
    t.to_csv(TABLES / "budget_comparison.csv")
    for s in SCENARIOS:
        fig_budget_comparison(runs, s)
    return t


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    runs, regs = load()
    for b in sorted(runs.label_budget.unique()):
        analyze_condition(runs[runs.label_budget == b], regs[regs.label_budget == b], b)
    eff = label_efficiency(runs)
    print("\n=== Efisiensi label S2 (kenaikan macro-F1 di atas 'none' per 10.000 label) ===")
    print(eff.loc["S2_cross"]["gain_per_10k_labels"].unstack().round(3).to_string())
    bc = budget_comparison(runs)
    if bc is not None:
        print("\n=== Perubahan macro-F1 akibat anggaran label (S2) ===")
        col = [c for c in bc.columns if c.startswith("f1_change")][0]
        print(bc.loc["S2_cross"][col].unstack().round(3).to_string())
    print(f"\nTabel -> {TABLES}\nGrafik -> {FIGS}")


if __name__ == "__main__":
    main()
