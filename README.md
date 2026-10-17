# Identitas Pembuat
* **Nama:** Anisa Julianti
* **NPM:** G1A023052
* **Program Studi:** Informatika, Universitas Bengkulu

---

# Studi Kasus Skripsi: Segmentasi Pelanggan UMKM Berbasis Unsupervised Machine Learning (K-Means & RFM)

Repositori ini berisi implementasi kode program Python untuk analisis data transaksi *Point of Sale* (POS) UMKM guna melakukan segmentasi pelanggan secara otomatis menggunakan pendekatan *Unsupervised Learning* (Algoritma *K-Means Clustering* dan Model RFM). Proyek ini dikembangkan sebagai bentuk penerapan praktis ilmu data untuk membantu pelaku UMKM beralih dari pemasaran massal (*mass marketing*) menuju personalisasi pemasaran yang tertarget.

---

## 1. Latar Belakang & Permasalahan UMKM
* **Isu Industri:** Banyak pelaku usaha kecil dan menengah (UMKM) belum memiliki sistem manajemen hubungan pelanggan (*CRM*) yang memadai, sehingga strategi promosi sering kali bersifat pukul rata (*mass marketing*) dan kurang efektif.
* **Solusi Berbasis IT:** Dengan memanfaatkan data transaksi penjualan harian, sarjana Informatika dapat merancang sistem otomatisasi segmentasi pelanggan menggunakan teknik klasterisasi tanpa memerlukan label data awal (*unlabeled data*).
* **Tujuan Proyek (`implementasi_materi4.py`):** 
  1. Melakukan ekstraksi fitur perilaku belanja pelanggan menggunakan model **RFM (*Recency, Frequency, Monetary*)**.
  2. Menerapkan normalisasi data dengan *StandardScaler* agar skala metrik seimbang.
  3. Mengimplementasikan algoritma **K-Means Clustering** didukung evaluasi *Silhouette Score* untuk membagi pelanggan ke dalam beberapa kluster strategis.
  4. Menghasilkan laporan eksekutif untuk acuan personalisasi promosi pemasaran UMKM.

---

## 2. Alur Kerja Analisis (*Data Science Pipeline*)
Skrip ini mengikuti alur standar industri pengolahan data dan *Machine Learning*:
1. **Pembangkitan Data Transaksi POS:** Mensimulasikan data transaksi penjualan harian produk (seperti lini produk kuliner UMKM).
2. **Feature Engineering (Model RFM):** Mengagregasi data transaksi mentah menjadi profil unik per pelanggan:
   * **Recency:** Seberapa baru/lama pelanggan melakukan transaksi terakhir.
   * **Frequency:** Seberapa sering pelanggan berbelanja.
   * **Monetary:** Total akumulasi nominal uang yang dibelanjakan.
3. **Data Preprocessing:** Melakukan standarisasi fitur menggunakan `StandardScaler` agar algoritma K-Means tidak bias terhadap skala nilai yang berbeda.
4. **Machine Learning & Evaluasi:** Menjalankan algoritma *K-Means Clustering* (K=4) dan mengevaluasi kualitas pemisahan kluster menggunakan *Silhouette Score*.
5. **Business Interpretation & Reporting:** Memberikan label deskriptif pada setiap kluster dan mengekspor laporan eksekutif ke dalam format `.csv`.

---

## 3. Implementasi Kode (`implementasi_materi4.py`)

Skrip Python ini dibangun menggunakan pustaka standar industri (*Pandas, NumPy, Scikit-Learn*):

```python
# 1. Simulasi Data Transaksi POS UMKM Skala Menengah
# 2. Advanced Feature Engineering (Model RFM: Recency, Frequency, Monetary)
# 3. Data Normalization menggunakan StandardScaler
# 4. Unsupervised K-Means Clustering & Evaluasi Silhouette Score
# 5. Interpretasi Bisnis & Ekspor Laporan CSV Siap Pakai
