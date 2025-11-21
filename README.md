# 🍜 Indonesian Food Classification using Vision Transformers

Comparative study of Vision Transformer architectures (ViT, Swin, DeiT) for Indonesian food image classification.

---

## 📌 Cara Menjalankan

Berikut langkah-langkah untuk menjalankan eksperimen Vision Transformer (ViT, Swin, dan DeiT) menggunakan notebook yang ada di repository ini. Dapat dijalankan di Jupyter Notebook (VS Code, JupyterLab, Google Colab).

---

## 🔹 1. Buka Notebook yang Anda Gunakan

Pilih salah satu:
- **VS Code** dengan Jupyter extension
- **JupyterLab** atau Jupyter Notebook
- **Google Colab**: [https://colab.research.google.com/](https://colab.research.google.com/)

---

## 🔹 2. Clone/Download Repository

**Jika menggunakan notebook lokal (VS Code/Jupyter):**
```bash
git clone https://github.com/username/Indonesian-Food-Classification-ViT.git
cd Indonesian-Food-Classification-ViT
```

**Jika menggunakan Google Colab:**
```bash
!git clone https://github.com/username/Indonesian-Food-Classification-ViT.git
%cd Indonesian-Food-Classification-ViT
```

> Ganti `username` dengan username GitHub Anda.

---

## 🔹 3. Install Dependencies

Jalankan perintah berikut untuk menginstall semua library yang dibutuhkan:

```bash
!pip install -r requirements.txt
```

Library yang akan ter-install mencakup:
* PyTorch & Torchvision
* timm (pre-trained Vision Transformers)
* scikit-learn
* pandas & numpy
* matplotlib & seaborn
* Pillow

---

## 🔹 4. Aktifkan GPU (Opsional tapi Disarankan)

**Untuk Google Colab:**
1. Masuk ke menu **Runtime → Change runtime type**
2. Pada bagian **Hardware accelerator**, pilih **GPU**
3. Tekan **Save**

**Untuk VS Code/Jupyter lokal:**
- Pastikan CUDA dan NVIDIA drivers sudah terinstall
- PyTorch akan otomatis mendeteksi GPU jika tersedia

Verifikasi apakah GPU aktif dengan menjalankan:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
```

Jika berhasil, output akan menampilkan informasi GPU seperti **Tesla T4**, **L4**, atau model GPU lainnya.

---

## 🔹 5. Siapkan Dataset

Dataset makanan Indonesia harus memiliki struktur berikut:

```
project/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── test/
│   ├── image1.jpg
│   └── ...
├── train.csv
└── test.csv
```

**Format CSV:**
```csv
filename,label
image1.jpg,bakso
image2.jpg,rendang
...
```

**Untuk lokal (VS Code/Jupyter):**
- Pastikan folder `train/`, `test/`, dan file CSV sudah ada di direktori project

**Untuk Google Colab:**
```python
# Upload file
from google.colab import files
uploaded = files.upload()

# Atau mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🔹 6. Jalankan Notebook

Buka file notebook utama:

```
food_classification.ipynb
```

**Untuk semua platform (VS Code/Jupyter/Colab):**
- Jalankan seluruh sel secara berurutan
- Atau gunakan **Run All** untuk menjalankan semua sel sekaligus

Notebook ini akan menjalankan seluruh pipeline eksperimen meliputi:
* Load & preprocessing dataset makanan Indonesia
* Training tiga model Vision Transformer: **ViT**, **Swin Transformer**, dan **DeiT**
* Evaluasi performa model: akurasi, precision, recall, F1-score
* Pembuatan confusion matrix untuk 5 kategori makanan
* Visualisasi kurva training dan validation (loss & accuracy)
* Perhitungan jumlah parameter & ukuran model
* Perhitungan waktu inferensi & throughput

---

## 🔹 7. (Opsional) Menyimpan Model

Model yang sudah dilatih akan disimpan secara otomatis dengan nama:
```
best_vit_model.pth
best_swin_model.pth
best_deit_model.pth
```

**Untuk Google Colab**, download model ke lokal:
```python
from google.colab import files
files.download('best_vit_model.pth')
files.download('best_swin_model.pth')
files.download('best_deit_model.pth')
```

**Untuk lokal**, file sudah tersimpan di direktori project.

---

## 📊 Output yang Dihasilkan

Setelah eksperimen selesai, akan dihasilkan:
* `training_curves.png` - Grafik loss dan akurasi
* `confusion_matrices.png` - Confusion matrix untuk setiap model
* `model_comparison_summary.csv` - Tabel perbandingan metrik
* `best_*.pth` - Model weights terbaik

---

## 🍽️ Kategori Makanan

Model ini mengklasifikasikan 5 jenis makanan Indonesia:
1. 🥘 **Bakso**
2. 🥗 **Gado-gado**
3. 🍛 **Nasi Goreng**
4. 🍖 **Rendang**
5. 🍲 **Soto Ayam**

---

## ⚙️ Konfigurasi Default

```python
CONFIG = {
    'img_size': 224,
    'batch_size': 8,
    'num_epochs': 10,
    'learning_rate': 1e-4,
    'validation_split': 0.2,
}
```

Modifikasi sesuai kebutuhan dan spesifikasi hardware Anda.

```

Notebook telah siap dijalankan dan seluruh hasil eksperimen akan tersimpan secara otomatis. Selamat mencoba! 🚀