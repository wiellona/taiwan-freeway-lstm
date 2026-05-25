# Tugas Akhir Kecerdasan Buatan (ENCE616029)
## Prediksi Waktu Tempuh Taiwan Freeway Menggunakan Arsitektur Long Short-Term Memory (LSTM)

Repositori ini memuat proyek akhir untuk mata kuliah Kecerdasan Buatan (ENCE616029), Program Studi Teknik Komputer, Universitas Indonesia. Proyek ini berfokus pada analisis data deret waktu (time-series) untuk memprediksi kondisi lalu lintas dan waktu tempuh jangka panjang di ruas jalan tol Taiwan.

### Anggota Tim (Kelompok K08)
Proyek ini dikembangkan oleh:
* Grace Yunike M. Sitorus (2306267031)
* Isyana Trevia Pohaci (2306250592)
* Stevie Nathania Siregar (2306242382)
* Wiellona Darlene O. S. (2306264396)

### Deskripsi Proyek
Mengurai kemacetan dan memprediksi waktu tempuh merupakan tantangan analitik yang kompleks karena melibatkan berbagai faktor eksternal. Oleh karena itu, penelitian dalam proyek ini memanfaatkan arsitektur *Long Short-Term Memory* (LSTM) yang dirancang secara spesifik dan andal untuk menangani data sekuensial. 

Terdapat dua fokus analisis utama dalam proyek ini:
1. **Evaluasi Komparatif:** Membandingkan performa model LSTM dengan beberapa model *baseline* non-sekuensial untuk mengukur signifikansi peningkatan akurasi.
2. **Analisis Variabilitas Performa:** Menguji secara komprehensif performa model pada berbagai kondisi temporal, secara khusus membandingkan tingkat akurasi saat jam sibuk (*rush hour*) dengan kondisi jalanan lengang (*non-rush hour*).

### Lingkungan Pengembangan
* **Lingkungan Eksekusi:** Google Colab

### Struktur Repositori
* `dataset/` : Direktori yang memuat sekumpulan data historis lalu lintas dari Taiwan Freeway dalam format `.csv` (contoh: `from01F0147Sto01F0155S.csv`).
* `Laporan_Akhir_AI_K08.docx.pdf` : Dokumen komprehensif yang memaparkan landasan arsitektur, metodologi, hingga hasil analisis akhir proyek.
* `metadata_K08.json` : Berkas informasi metadata operasional dan deklarasi kontribusi masing-masing anggota tim.
* `File Notebook (.ipynb)` : Kode sumber utama yang memuat keseluruhan alur program, dari tahap pra-pemrosesan data hingga inferensi prediksi waktu tempuh.

### Panduan Eksekusi (melalui Google Colab)
Proyek ini dirancang agar dapat dieksekusi secara praktis tanpa memerlukan instalasi perangkat lunak lokal, yakni dengan memanfaatkan platform Google Colab. Berikut adalah tahapan untuk menjalankan program:

1. **Akses Notebook:** Buka file `.ipynb` utama yang tersedia di dalam repositori ini menggunakan platform Google Colab.
2. **Persiapan Dataset:** Pastikan seluruh file `.csv` dari direktori `dataset/` telah diunggah ke dalam direktori file sesi Google Colab (atau hubungkan sesi dengan Google Drive yang memuat dataset tersebut) agar model dapat mengakses data.
3. **Konfigurasi Akselerator (Opsional namun Disarankan):** Untuk mempercepat proses pelatihan (*training*) model, disarankan untuk menggunakan instans GPU. Ubah runtime melalui menu `Runtime` > `Change runtime type` > pilih `GPU (T4)` pada bagian *Hardware accelerator*.
4. **Eksekusi Kode:** Jalankan sel kode secara berurutan dari atas ke bawah. Alur eksekusi akan mencakup tahapan impor pustaka (*library*), pra-pemrosesan data, pelatihan model (*training*), dan diakhiri dengan visualisasi grafik hasil prediksi.
