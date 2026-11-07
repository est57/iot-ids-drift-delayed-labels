# Drift Detectors for IoT Intrusion Detection under Delayed Labels and Cross-Dataset Shift

Repositori ini berisi kode, data olahan, hasil eksperimen, dan naskah untuk penelitian tentang kinerja
detektor *concept drift* pada sistem deteksi intrusi (IDS) jaringan IoT. Penelitian ini menguji sepuluh
detektor drift pada aliran data NetFlow yang mengalami pergeseran antar-dataset, dengan label yang datang
terlambat dan dengan anggaran label yang dibatasi.

**Penulis:** Aditya Maulana, Apri Istiarto, Diah Puspitasari, Sumanto
**Afiliasi:** Program Studi Magister Teknologi Informasi, Fakultas Teknik dan Informatika,
Universitas Bina Sarana Informatika
**Target publikasi:** Journal of Information Systems and Informatics (JISI)

## Ringkasan Hasil

Eksperimen mencakup 39 stream, 10 detektor, 3 baseline, dan 4 tingkat keterlambatan label, dijalankan
dalam dua kondisi (tanpa batas label dan anggaran 15.000 label), dengan total 3.783 run.

| Temuan | Nilai utama |
|---|---|
| Jeda deteksi detektor berbasis error naik mengikuti keterlambatan label | 1.371 → 23.047 flow (median, skenario S2) |
| Detektor tanpa label tidak terpengaruh keterlambatan label | sekitar 3.000 flow di semua tingkat |
| Macro-F1 pada keterlambatan 20.000 flow | ADWIN-score 0,806; KS-feat 0,799; oracle 0,822; berbasis error 0,67–0,70 |
| Dampak anggaran label pada detektor yang sering alarm palsu | penurunan macro-F1 hingga 0,30 (KSWIN-score) |
| Kesesuaian peringkat drift sintetis vs antar-dataset | ρ = 0,73–0,99 pada keterlambatan ≤ 5.000; ρ ≈ 0,50 (tidak signifikan) pada 20.000 |

Uraian lengkap tersedia di [results/TEMUAN_AWAL.md](results/TEMUAN_AWAL.md).

## Kebutuhan Sistem

- Python 3.12
- Ruang disk sekitar 16 GB (14 GB untuk CSV mentah, sisanya untuk arsip zip dan hasil)
- RAM minimal 16 GB untuk praproses dataset terbesar

Seluruh dependensi Python tercantum di [requirements.txt](requirements.txt). Versi pustaka dikunci agar
hasil dapat direproduksi.

## Instalasi

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data

Penelitian ini memakai dataset NetFlow v3 dari University of Queensland. Ketiga dataset memiliki 53 fitur
yang sama, termasuk stempel waktu awal dan akhir flow serta statistik *inter-packet arrival time*.

| Dataset | Peran | Jumlah flow | Proporsi serangan | Rentang waktu |
|---|---|---|---|---|
| NF-ToN-IoT-v3 | IoT | 27.520.260 | 38,98% | 144,6 jam |
| NF-BoT-IoT-v3 | IoT | 16.933.808 | 99,69% | 843,8 jam |
| NF-UNSW-NB15-v3 | Jaringan umum (pembanding) | 2.365.424 | 5,40% | 648,7 jam |

NF-CICIDS2018-v3 ikut diunduh dan diperiksa, tetapi tidak dipakai dalam eksperimen utama.

### Mengunduh data

1. Buka halaman dataset UQ: <https://staff.itee.uq.edu.au/marius/NIDS_datasets/>.
2. Pada bagian *NetFlow V3 Datasets*, unduh file untuk setiap dataset melalui portal UQ RDM.
3. Ekstrak arsip zip dan letakkan file CSV di `data/raw/` dengan nama aslinya, misalnya
   `data/raw/NF-ToN-IoT-v3.csv`.
4. Periksa setiap file:

```bash
.venv/bin/python scripts/check_dataset.py data/raw/NF-ToN-IoT-v3.csv
```

Skrip pemeriksaan menampilkan daftar kolom, distribusi label, rentang waktu, dan jumlah baris yang tidak
berurutan waktu. Hasil pemeriksaan untuk keempat file tersimpan di `results/dataset_check/`. Checksum SHA-1
seluruh file sudah dicocokkan dengan manifest resmi.

### Lisensi dan sitasi data

Dataset NetFlow v3 berstatus *open access* dengan lisensi yang membolehkan pemakaian ulang, tetapi tidak untuk
tujuan komersial. Setiap publikasi yang memakai data ini perlu menyitasi:

> M. Luay et al., "Time matters: Temporal NetFlow features for ML-based network intrusion detection,"
> *IEEE Access*, vol. 14, pp. 66899–66913, 2026, doi: 10.1109/ACCESS.2026.3688204.

> M. Sarhan, S. Layeghy, and M. Portmann, "Towards a standard feature set for network intrusion detection
> system datasets," *Mobile Netw. Appl.*, vol. 27, no. 1, pp. 357–370, 2022, doi: 10.1007/s11036-021-01843-0.

## Menjalankan Eksperimen

Jalankan perintah berikut dari folder `scripts/` secara berurutan.

```bash
cd scripts
../.venv/bin/python preprocess.py all
../.venv/bin/python build_streams.py preset --seeds 1 2 3
../.venv/bin/python run_grid.py --jobs 8
../.venv/bin/python run_grid.py --jobs 8 --budget 15000
../.venv/bin/python analyze.py
```

| Langkah | Skrip | Keluaran |
|---|---|---|
| 1. Praproses | `preprocess.py` | `data/processed/*.parquet` dan ringkasan `*.json` |
| 2. Pembentukan stream | `build_streams.py` | 39 stream di `data/streams/` beserta metadata wilayah drift |
| 3. Eksperimen tanpa batas label | `run_grid.py` | `results/runs/*.json` |
| 4. Eksperimen dengan anggaran label | `run_grid.py --budget 15000` | `results/runs/*__b15000.json` |
| 5. Analisis | `analyze.py` | `results/tables/`, `results/figures/` |

`run_grid.py` melewati kombinasi yang hasilnya sudah ada. Gunakan `--force` untuk mengulang, atau
`--summary-only` untuk membuat ulang ringkasan tanpa menjalankan eksperimen. Untuk satu kombinasi saja,
gunakan `run_experiment.py`:

```bash
../.venv/bin/python run_experiment.py S2-cross-W5000_ToN-BoT-UNSW_s1 KS-feat --delay 5000
```

## Struktur Repositori

```
data/
  raw/            CSV asli NetFlow v3 (tidak diubah)
  processed/      Data hasil praproses (parquet)
  streams/        Stream skenario S1, S2, S3 beserta metadata
scripts/
  common.py           Konstanta, kolom yang dibuang, pemetaan label
  check_dataset.py    Pemeriksaan file CSV mentah
  preprocess.py       Praproses dan sampling
  build_streams.py    Pembentukan stream skenario drift
  run_experiment.py   Satu run: stream × detektor × keterlambatan label
  run_grid.py         Eksekusi paralel dan ringkasan hasil
  analyze.py          Tabel, uji Friedman–Nemenyi, korelasi Spearman, grafik
results/
  runs/           Hasil mentah per run (JSON)
  tables/         Tabel ringkasan (CSV)
  figures/        Grafik hasil (PNG, teks berbahasa Inggris)
  summary_runs.csv, summary_regions.csv
  TEMUAN_AWAL.md
```

## Desain Eksperimen

**Praproses.** Kolom alamat IP, port, DNS query ID, dan stempel waktu tidak dipakai sebagai fitur karena
mengidentifikasi lingkungan perekaman dan dapat menyebabkan kebocoran informasi dataset. Stempel waktu hanya
dipakai untuk mengurutkan data. Label disatukan ke empat kategori yang muncul di minimal dua dataset (Benign,
DoS, DDoS, Recon), lalu setiap kategori dibatasi 150.000 flow melalui sampling acak seragam. Nilai *inf* dan
*NaN* pada kolom laju byte (flow berdurasi nol) diganti dengan nol.

**Urutan data.** ToN-IoT dan BoT-IoT menjalankan serangan secara bergiliran (Recon, kemudian DoS, kemudian
DDoS). Karena itu, urutan flow diacak di dalam setiap dataset agar drift yang terjadi hanya drift yang
dirancang. Urutan waktu asli tetap dapat dipakai melalui opsi `build_streams.py --order time`.

**Skenario drift.**

| Skenario | Deskripsi | Jumlah stream |
|---|---|---|
| S1 | Drift sintetis: 10 fitur dikalikan faktor acak 1,5–3,0 pada setiap regime baru | 9 |
| S2 | Drift antar-dataset dengan 3 urutan dataset dan lebar transisi W = 0, 5.000, 20.000 flow | 27 |
| S3 | Kelas serangan baru (Recon) muncul pada segmen kedua | 3 |

**Protokol adaptasi.** Model dasar adalah `HistGradientBoostingClassifier` yang dilatih pada 10.000 flow
pertama. Label setiap flow baru tersedia setelah *d* flow (d = 0; 1.000; 5.000; 20.000). Ketika detektor
memberi alarm pada posisi *t*, sistem mengumpulkan 5.000 flow berikutnya dan melatih ulang model pada posisi
*t* + 5.000 + *d*. Protokol ini sama untuk semua detektor. Pada kondisi anggaran, setiap stream dibatasi
15.000 label (maksimal tiga kali latih ulang).

**Detektor yang diuji.**

| Kelompok | Detektor | Sinyal yang dipantau |
|---|---|---|
| Berbasis error | ADWIN, DDM, EDDM, HDDM_A, HDDM_W, Page-Hinkley | Error prediksi (memerlukan label) |
| Tanpa label, skor | KSWIN-score, ADWIN-score, PH-score | Probabilitas serangan dari model |
| Tanpa label, fitur | KS-feat | Distribusi lima fitur dengan *mutual information* tertinggi |
| Baseline | none, periodic, oracle | Tanpa adaptasi, latih ulang tiap 25.000 flow, alarm tepat di awal drift |

**Metrik dan uji statistik.** Tingkat deteksi, jeda deteksi, alarm palsu per 10.000 flow, macro-F1, dan
jumlah label yang dipakai. Peringkat detektor diuji dengan uji Friedman dan *critical difference* Nemenyi
(α = 0,05). Kesesuaian peringkat antara S1 dan S2 diukur dengan korelasi Spearman. Metrik *loss area* tetap
dihitung, tetapi tidak dipakai untuk peringkat karena bias terhadap detektor yang gagal pulih pada segmen
sebelumnya.

## Keterbatasan

Hasil penelitian ini perlu dibaca dengan memperhatikan beberapa keterbatasan berikut:
- hanya satu model dasar dan satu jenis fitur (NetFlow) yang diuji;
- setiap skenario memakai tiga *seed*;
- parameter detektor memakai nilai bawaan pustaka River;
- skenario S3 hanya terdiri atas tiga stream;
- anggaran label baru diuji pada satu nilai.
