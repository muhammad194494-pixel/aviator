# Aviator Data Logger & Pattern Analysis

Project eksperimen buat ngumpulin data hasil round game crash "Aviator" secara otomatis dari sisi client, lalu nganalisa data tersebut secara statistik.

## Latar Belakang

Crash game model "Aviator" nampilin riwayat hasil multiplier tiap round di UI, tapi ga ada cara resmi buat ekspor data tersebut. Project ini dibuat buat ngotomatisin pencatatan history yang udah keliatan di layar, supaya bisa dianalisa secara terstruktur tanpa harus nyatet manual satu-satu.

## Cara Kerja

1. **Hook logging** — sebuah baris log debug (tag `KUPING_CRASH`) ditambahkan pada titik di mana game menampilkan hasil akhir tiap round (event "pesawat jatuh"), supaya nilai multiplier final round itu ikut tercatat di Android logcat.
2. **Logger Python (`aviator_logger.py`)** — jalan di background lewat Termux, terus-terusan baca stream `logcat -s KUPING_CRASH`, extract nilai multiplier pakai regex, lalu append ke file JSON.
3. **Penyimpanan data** — tiap multiplier yang berhasil ditangkap disimpan sebagai array angka di file JSON (`aviator_data*.json`), terurut secara kronologis (data terlama di atas, terbaru di bawah).

## Struktur File

| File | Keterangan |
|---|---|
| `aviator_logger.py` | Script utama, baca logcat & tulis ke JSON |
| `aviator_data*.json` | Hasil capture multiplier per sesi |
| `decision_support*.py` | Analisis pola statistik dari data historis |
| `pola_lstm*.py` / `*.keras` | Eksperimen model neural network (LSTM) untuk uji pola sekuens |
| `prediction_log.txt` | Log hasil eksperimen prediksi |
| `server_upload.py` | Upload data ke server/storage eksternal |

## ⚠️ Catatan Penting Soal Validitas Statistik

Platform crash game seperti ini umumnya menggunakan sistem **provably fair** — hasil tiap round digenerate dari seed kriptografis yang independen secara statistik dari round-round sebelumnya. Artinya:

- Pola apapun yang "ketemu" dari analisis historis (termasuk hasil model LSTM di repo ini) **sangat mungkin cuma noise statistik**, bukan sinyal yang valid, apalagi dengan sample size yang masih kecil.
- Project ini dibuat untuk tujuan **eksperimen & riset teknis** (apakah outputnya benar-benar random atau ada pola yang bisa diverifikasi secara matematis), **bukan** sebagai dasar pengambilan keputusan taruhan.
- Cara paling valid buat verifikasi fairness sebuah platform provably fair adalah lewat mekanisme reveal seed/hash resminya (kalau disediakan platform), bukan dari analisis pola output semata.

## Requirements

- Termux (Android, rooted)
- Python 3 (`pip install` — lihat masing-masing script untuk dependency)
- Akses `su` untuk membaca logcat sistem

## Disclaimer

Project ini dibuat untuk tujuan riset/eksperimen pribadi. Penggunaan data atau hasil analisis di repo ini untuk keputusan finansial/taruhan dilakukan atas risiko sendiri.
