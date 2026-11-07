# Temuan awal (diperbarui 7 Nov 2026)

Eksperimen: 39 stream (3 seed × 13 skenario) × 10 detektor + 3 baseline × 4 nilai keterlambatan
label (0, 1.000, 5.000, 20.000 sampel), dalam dua kondisi:

- **Tanpa batas label**: 1.911 run.
- **Anggaran label 15.000 per stream** (maksimal 3 latih ulang × 5.000 label): 1.872 run.
  S1 dan S2 punya 2 drift, S3 punya 1, jadi anggaran ini menyisakan cadangan 1–2 latih ulang.
  Setelah anggaran habis, detektor dimatikan sampai akhir stream.

Tabel ada di `results/tables/` (sufiks `_b15000` untuk kondisi anggaran). Grafik berbahasa Inggris
ada di `results/figures/`. Angka di bawah adalah rata-rata seluruh stream per skenario.

## 1. Keterlambatan label melumpuhkan detektor berbasis error (didukung kuat)

Median jeda deteksi, S2 (drift antar-dataset), tanpa batas label:

| Keluarga | d=0 | d=1.000 | d=5.000 | d=20.000 | Tingkat deteksi d=20.000 |
|---|---|---|---|---|---|
| Berbasis error (butuh label) | 1.371 | 2.407 | 6.260 | 23.047 | 0,3 |
| Tanpa label, skor model | 2.521 | 2.391 | 3.249 | 2.975 | 0,7 |
| Tanpa label, fitur (KS-feat) | 3.000 | 3.000 | 3.000 | 3.000 | 1,0 |

Jeda deteksi detektor berbasis error naik kira-kira sebesar keterlambatan label, sedangkan detektor
tanpa label tidak terpengaruh. Tanpa keterlambatan, detektor berbasis error justru paling cepat.
Grafik: `S2_cross_detection_delay.png`.

## 2. Pada keterlambatan besar, detektor tanpa label mendekati oracle (konsisten di kedua kondisi)

Macro-F1, S2, d=20.000: oracle 0,822 · **ADWIN-score 0,806** · **KS-feat 0,799** ·
detektor berbasis error 0,67–0,70 · tanpa adaptasi 0,637. Angka ini sama dengan atau tanpa
anggaran label, karena pada d=20.000 tidak ada detektor yang memakai lebih dari 3 latih ulang
secara bermakna.

## 3. Menyetarakan biaya label mengubah peringkat secara drastis

Tanpa batas, KSWIN-score menempati peringkat 1 pada d=0, tetapi memakai rata-rata 102.778 label
di S2, sekitar 10 kali lipat oracle. Dengan anggaran 15.000 label:

Perubahan macro-F1 di S2 (anggaran dikurangi tanpa batas), d=0:
KSWIN-score −0,304 · EDDM −0,253 · DDM −0,167 · PH-score −0,127 · HDDM_W −0,121 · periodic −0,110 ·
ADWIN, HDDM_A, ADWIN-score, KS-feat, oracle **0,000**.

Detektor yang sering memicu alarm palsu menghabiskan anggaran sebelum drift sebenarnya datang.
Peringkat teratas S2 dengan anggaran:

| Delay | Peringkat 1 | Peringkat 2 | Peringkat 3 | Terakhir |
|---|---|---|---|---|
| 0 | ADWIN (1,74) | PageHinkley (2,52) | HDDM_A (3,00) | KSWIN-score (9,04) |
| 5.000 | ADWIN-score (2,07) | KS-feat (2,33) | ADWIN (3,11) | KSWIN-score (9,89) |
| 20.000 | ADWIN-score (1,78) | KS-feat (2,07) | PH-score (3,19) | HDDM_W (8,22) |

Friedman p < 1e-24 di semua delay; CD Nemenyi = 2,61. Grafik: `S2_cross_budget_comparison.png`.

Efisiensi label tanpa batas (kenaikan macro-F1 di atas "none" per 10.000 label, S2, d=0):
oracle 0,296 · ADWIN-score 0,279 · HDDM_A 0,274 · ADWIN 0,253 · KS-feat 0,246 · KSWIN-score 0,030.
Hasilnya selaras: detektor yang efisien label adalah detektor yang tidak terpengaruh anggaran.

## 4. Kesesuaian peringkat sintetis vs antar-dataset runtuh pada keterlambatan label besar

Korelasi Spearman peringkat detektor (macro-F1) S1 vs S2:

| Delay | Tanpa batas label | Anggaran 15.000 |
|---|---|---|
| 0 | 0,73 (p=0,016) | 0,94 (p<0,001) |
| 1.000 | 0,85 (p=0,002) | 0,95 (p<0,001) |
| 5.000 | 0,79 (p=0,007) | 0,99 (p<0,001) |
| 20.000 | **0,50 (p=0,14, tidak signifikan)** | **0,49 (p=0,15, tidak signifikan)** |

Drift sintetis cukup memprediksi peringkat pada drift antar-dataset selama label datang cepat,
terlebih jika biaya label disetarakan. **Pada keterlambatan 20.000, kesesuaian itu hilang di kedua
kondisi.** Ini temuan yang konsisten dan menjadi rumusan ulang bagian kedua H1:
*benchmark drift sintetis tidak lagi representatif saat label sangat terlambat.*

## 5. Alarm palsu: hanya sebagian sesuai hipotesis

Alarm palsu per 10.000 sampel (S2, d=0, tanpa batas): skor model 0,61 · berbasis error 0,37 ·
**KS-feat 0,04**. Klaim "tanpa label = lebih banyak alarm palsu" hanya berlaku untuk detektor
berbasis skor. KS-feat justru paling hemat alarm.

## 6. KS-feat buta terhadap serangan baru (S3)

KS-feat tidak pernah mendeteksi munculnya kelas Recon di S3 (tingkat deteksi 0). Lima fitur yang
dipantau dipilih dari data awal, sehingga pergeseran di fitur lain tidak terlihat. Detektor berbasis
skor (ADWIN-score, PH-score) mendeteksinya dan menempati peringkat teratas S3 pada d ≥ 5.000.

## Keterbatasan yang masih terbuka

- S3 hanya 3 stream, sehingga CD Nemenyi besar (7,8) dan tidak ada perbedaan berpasangan yang signifikan.
- Drift sintetis hanya satu jenis (perkalian 10 fitur dengan faktor 1,5–3).
- Parameter detektor memakai nilai bawaan `river`; ambang KS-feat (p < 0,001/5 dan D > 0,2) ditetapkan manual.
- Baru satu model dasar (HistGradientBoosting) dan 3 seed per skenario.
- Anggaran label baru diuji satu nilai (15.000).
- `total_loss_area` bias dan tidak dipakai untuk peringkat (lihat docstring `analyze.py`).
