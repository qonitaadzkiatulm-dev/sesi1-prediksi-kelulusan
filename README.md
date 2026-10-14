# Prediksi Kelulusan Mahasiswa

Studi kasus berbasis kuliah praktisi **"Make Sense of Data with Analysis and AI"**
(Andria Arisal, BRIN, *Informatics Practitioner Lecture Series*, Universitas Bengkulu).
Proyek ini menerapkan alur analitik dari materi tersebut pada kasus perguruan tinggi
yang ingin mendeteksi mahasiswa berisiko **lulus terlambat** sejak dini.

> **Catatan:** seluruh data adalah **sintetis** (dibangkitkan oleh `src/generate_data.py`),
> bukan data mahasiswa sebenarnya. Angka performa hanya menggambarkan alur kerja,
> bukan klaim tentang mahasiswa nyata.

## 1. Pertanyaan bisnis -> pertanyaan analitik

| Aspek | Isi |
|---|---|
| Pertanyaan bisnis | Mahasiswa mana yang perlu didampingi dosen wali agar tidak terlambat lulus? |
| Pertanyaan analitik | Berdasarkan data sampai akhir **semester 4**, berapa peluang mahasiswa lulus > 8 semester? |
| Unit analisis (granularitas) | 1 baris = 1 mahasiswa |
| Target | `lulus_tepat_waktu` (1 = ≤ 8 semester). Kelas positif model = **terlambat** |
| Level analitik | Deskriptif -> Diagnostik -> **Prediktif** -> (arah) Preskriptif berupa daftar pendampingan |
| Metrik utama | **Recall kelas terlambat** (jangan sampai mahasiswa berisiko terlewat), dilengkapi precision, F2, ROC-AUC, PR-AUC |

## 2. Alur CRISP-DM yang diterapkan

1. **Business understanding** - bagian 1 di atas.
2. **Data understanding** - `data_prep.quality_report()` memeriksa duplikat, nilai kosong, IPS di luar 0-4, dan huruf mutu tidak baku.
3. **Data preparation** (porsi kerja terbesar, sesuai pesan pemateri *garbage in, garbage out*):
   - hapus duplikat, standarkan huruf mutu (`" a "` -> `A`), IPS di luar 0-4 dijadikan kosong lalu diimputasi median (aturan bisnis);
   - **pencegahan data leakage**: `ipk_akhir`, `lama_studi_semester`, `status_yudisium`, serta IPS/SKS semester 5-8 dibuang karena baru diketahui setelah prediksi seharusnya dibuat;
   - fitur *point-in-time*: IPS S1-S4, rata-rata, minimum, tren IPS, total SKS ambil/lulus, rasio SKS lulus, nilai MK kunci (Kalkulus, Algoritma, Struktur Data, Basis Data), jumlah MK mengulang, jumlah nilai D/E, cuti, kerja paruh waktu, jalur masuk.
4. **Modeling** - Logistic Regression, Decision Tree, Random Forest, dibandingkan dengan *baseline* mayoritas.
   Data tidak seimbang (~38% terlambat), jadi memakai `class_weight` agar model tidak "malas" menebak semua tepat waktu.
5. **Evaluation**
   - **Split berbasis waktu**: latih angkatan 2015-2018, uji angkatan 2019-2020 (meniru pemakaian nyata ke angkatan baru).
   - *Threshold* dipilih dari prediksi *out-of-fold* data latih (memaksimalkan F2), tanpa menyentuh data uji.
6. **Deployment/report** - `src/predict.py` dan `reports/daftar_pendampingan.csv` (urut skor risiko).

## 3. Hasil (data uji angkatan 2019-2020)

Lihat `reports/perbandingan_model.csv`, `reports/hasil.json`, dan gambar di `reports/`.

| Model | Recall (terlambat) | Precision | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Baseline mayoritas | 0.00 | 0.00 | 0.500 | 0.386 |
| **Logistic Regression** | **0.92** | 0.49 | 0.787 | **0.721** |
| Decision Tree | 0.95 | 0.46 | 0.775 | 0.652 |
| Random Forest | 0.97 | 0.44 | 0.785 | 0.704 |

**Interpretasi.** Ketiga model jauh lebih baik daripada baseline yang tidak menemukan satu pun mahasiswa berisiko.
Logistic Regression dipilih karena PR-AUC tertinggi dan koefisiennya mudah dijelaskan ke dosen wali.
Konsekuensinya, dengan ambang rendah demi recall tinggi, sekitar separuh mahasiswa yang ditandai sebenarnya
akan lulus tepat waktu (precision ~0.49). Dalam konteks pendampingan, biaya salah tandai (satu sesi bimbingan)
jauh lebih kecil daripada biaya terlewat, sehingga pertukaran ini wajar.

## 4. Keterbatasan dan etika

- Model **membantu** dosen wali, bukan memutuskan. Keputusan pendampingan tetap oleh manusia.
- Sebab keterlambatan (alasan finansial, kesehatan, motivasi) tidak terlihat di data; perlu percakapan langsung.
- Data nyata bersifat pribadi: gunakan anonimisasi dan izin institusi sebelum dipakai.
- Peringatan dini bisa menstigma; skor sebaiknya tidak dibagikan di luar dosen wali dan unit terkait.
- Hasil model perlu dipantau berkala (kurikulum dan kebijakan berubah) dan dilatih ulang.

## 5. Menjalankan

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/generate_data.py      # membuat data/mahasiswa_raw.csv
python src/train.py              # melatih, mengevaluasi, menyimpan laporan ke reports/
python src/predict.py            # contoh prediksi memakai model tersimpan
```

## 6. Struktur proyek

```
prediksi-kelulusan/
├── data/mahasiswa_raw.csv        # data sintetis (berantakan, sengaja)
├── src/generate_data.py          # pembangkit data
├── src/data_prep.py              # kualitas data, pembersihan, fitur
├── src/train.py                  # modeling dan evaluasi
├── src/predict.py                # inferensi
├── reports/                      # metrik, grafik, model, daftar pendampingan
├── requirements.txt
└── README.md
```
