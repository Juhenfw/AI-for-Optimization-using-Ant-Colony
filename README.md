# Rute Antar Kantor Kecamatan Surabaya

Repository ini berisi skrip Python yang bertujuan untuk menyelesaikan masalah Traveling Salesman Problem (TSP) di Kota Surabaya. Dengan memanfaatkan algoritma Ant Colony Optimization, skrip ini menghitung rute optimal yang menghubungkan kantor-kantor kecamatan di seluruh Surabaya. Hasilnya adalah visualisasi jalur yang dapat diikuti untuk mengunjungi semua kantor kecamatan dengan jarak tempuh minimum.

## Fitur Utama

- **Visualisasi Rute**: Skrip ini menampilkan peta Kota Surabaya dengan jalur rute optimal yang dihasilkan oleh algoritma. Jalur tersebut menunjukkan urutan kunjungan ke kantor kecamatan.
- **Algoritma Optimasi**: Menggunakan algoritma Ant Colony Optimization untuk menemukan solusi terbaik dari TSP, yang mengoptimalkan rute berdasarkan jarak antar kecamatan.
- **Analisis dan Statistik**: Menyediakan perhitungan statistik untuk membandingkan kinerja hasil konvergen dari algoritma dengan variasi acak yang terjadwal.
- **Ekstensi dan Eksperimen**: Mudah diperluas untuk mencakup wilayah lain seperti Sidoarjo atau Depok, serta untuk menguji berbagai parameter algoritma.

## Teknologi yang Digunakan

- **Python**: Bahasa pemrograman utama untuk pengembangan skrip ini.
- **Matplotlib**: Digunakan untuk visualisasi peta dan jalur optimal.
- **NumPy**: Mendukung operasi matematis yang cepat dan efisien.
- **Pandas**: Digunakan untuk analisis statistik hasil optimasi.

## Cara Penggunaan

1. **Instalasi**: Pastikan Anda telah menginstal semua dependensi yang dibutuhkan, seperti Matplotlib, NumPy, dan Pandas.
2. **Eksekusi**: Jalankan skrip `Rute_Antar_Kantor_Kecamatan_Surabaya.py` untuk memulai proses optimasi dan menghasilkan visualisasi rute.
3. **Hasil**: Lihat gambar hasil yang disimpan di direktori kerja Anda, atau langsung dari output Jupyter Notebook jika Anda menggunakan lingkungan tersebut.

## Contoh Penggunaan

Anda dapat melihat contoh hasil jalur optimal yang dihasilkan dari beberapa skenario berbeda, seperti:
- Jalur optimal untuk seluruh kecamatan di Surabaya.
- Variasi rute dengan penjadwalan waktu berbeda untuk menunjukkan dampak terhadap performa algoritma.

![Peta Kantor Kecamatan Surabaya](path.png)
![Jalur Optimal](Kecamatan.png)

## Lisensi

Proyek ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detailnya.
