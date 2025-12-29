# 🚀 Panduan Setup Web Interface Deteksi Iklan

## 📋 Requirements
Pastikan Anda sudah menginstall:
- Python 3.8 atau lebih tinggi
- pip (Python package manager)

## 🔧 Langkah-Langkah Instalasi

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Melatih Model (Retraining)
Jika Anda menambahkan data baru ke folder `dataset/` atau `ad_detection_model.keras` belum ada, jalankan perintah ini untuk melatih ulang model:

```bash
python train_model.py
```

Proses ini akan:
- Memuat dataset dari folder `dataset/` (iklan & noniklan)
- Melatih model MobileNetV2 selama 10 epoch
- Menyimpan model baru ke `ad_detection_model.keras`

**⏱️ Estimasi waktu**: 2-5 menit tergantung spesifikasi komputer.

### 3. Jalankan Aplikasi Web (Flask)
Setelah model siap, jalankan aplikasi web dengan perintah:

```bash
python app.py
```

Aplikasi akan berjalan di: http://127.0.0.1:5000

## 🎯 Cara Menggunakan Web Interface

1. Buka browser dan pergi ke `http://127.0.0.1:5000`
2. **Upload Gambar**: Klik tombol upload atau drag & drop gambar.
3. **Lihat Hasil**: Aplikasi akan menampilkan:
   - Gambar yang diupload
   - Prediksi (IKLAN atau NON-IKLAN)
   - Confidence score (%)

## 📁 Struktur File
```
.
├── app.py                       # Aplikasi web (Flask)
├── train_model.py               # Script untuk melatih model
├── requirements.txt             # Daftar library yang dibutuhkan
├── dataset/
│   ├── iklan/                   # Folder gambar iklan
│   └── noniklan/                # Folder gambar bukan iklan
└── ad_detection_model.keras     # File model hasil training
```

## 🛠️ Troubleshooting

### Error: "Model not loaded"
✅ **Solusi**: Pastikan Anda sudah menjalankan `python train_model.py` setidaknya sekali untuk membuat file model.

### Error: "ModuleNotFoundError"
✅ **Solusi**: Install library yang kurang dengan `pip install -r requirements.txt`.
