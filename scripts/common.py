"""Konstanta bersama: daftar dataset, kolom fitur, dan pemetaan label."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
STREAMS = ROOT / "data" / "streams"
RESULTS = ROOT / "results"

DATASETS = ["NF-ToN-IoT-v3", "NF-BoT-IoT-v3", "NF-UNSW-NB15-v3", "NF-CICIDS2018-v3"]
SHORT = {"NF-ToN-IoT-v3": "ToN", "NF-BoT-IoT-v3": "BoT",
         "NF-UNSW-NB15-v3": "UNSW", "NF-CICIDS2018-v3": "CIC"}

TIME_COL = "FLOW_START_MILLISECONDS"

# Kolom yang TIDAK dipakai sebagai fitur model:
# - alamat IP dan port asal: identitas spesifik per dataset -> kebocoran (leakage)
# - L4_DST_PORT: sangat spesifik per lingkungan (Eren dkk. 2026 menemukan drift ekstrem)
# - DNS_QUERY_ID: angka acak transaksi
# - stempel waktu: hanya untuk mengurutkan stream
# - Label/Attack: target
EXCLUDE = {
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "DNS_QUERY_ID",
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Label", "Attack",
}

# Kategori yang muncul di >= 3 dataset; dipakai sebagai default.
SHARED_CATEGORIES = ["Benign", "DoS", "DDoS", "Recon"]
ALL_CATEGORIES = SHARED_CATEGORIES + ["Backdoor", "BruteForce", "WebInjection"]


def clean_features(df, features):
    """Ganti inf/NaN dengan 0.

    SRC_TO_DST_SECOND_BYTES dan DST_TO_SRC_SECOND_BYTES bernilai inf atau NaN pada flow
    berdurasi 0 (pembagian dengan nol) di BoT-IoT dan UNSW-NB15. Laju byte untuk flow
    seperti itu tidak terdefinisi, jadi diisi 0. Mengembalikan jumlah sel yang diganti per kolom.
    """
    import numpy as np
    vals = df[features].to_numpy()
    bad = ~np.isfinite(vals)
    counts = {f: int(c) for f, c in zip(features, bad.sum(axis=0)) if c}
    if counts:
        vals[bad] = 0
        df[features] = vals
    return counts


def map_attack(name: str) -> str | None:
    """Petakan nilai kolom Attack (semua dataset) ke kategori bersama. None = dibuang."""
    a = str(name).strip().lower()
    if a == "benign":
        return "Benign"
    if a.startswith("ddos"):
        return "DDoS"
    if a.startswith("dos"):
        return "DoS"
    if a in ("scanning", "reconnaissance"):
        return "Recon"
    if a == "backdoor":
        return "Backdoor"
    if a in ("password", "ftp-bruteforce", "ssh-bruteforce"):
        return "BruteForce"
    if a in ("xss", "injection", "brute_force_-web", "brute_force_-xss", "sql_injection"):
        return "WebInjection"
    return None  # mitm, ransomware, theft, exploits, fuzzers, generic, shellcode, analysis, worms, bot, infilteration
