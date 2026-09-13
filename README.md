# ♻️ Klasifikasi Sampah dengan Transfer Learning (CNN)

Final Project Machine Learning — model *image classification* untuk mengenali jenis sampah dari foto, menggunakan transfer learning berbasis CNN.

## Deskripsi

Proyek ini membangun model deep learning untuk mengklasifikasikan gambar sampah ke dalam **6 kelas**:

| Kelas | Deskripsi |
|---|---|
| 🥤 Botol Plastik | Botol berbahan plastik |
| 🍾 Kaca | Botol/wadah kaca |
| 🧃 Karton Minuman | Kemasan karton untuk minuman (misal jus, susu) |
| 🥫 Kemasan Logam | Kaleng/kemasan berbahan logam |
| 📦 Kertas & Karton | Kertas, kardus, dan sejenisnya |
| 🗑️ Sampah Rumah Tangga | Kategori sampah campuran/umum |

Model final mencapai **98% accuracy** pada test set, menggunakan **MobileNetV2** yang di-*fine-tune* dari base model pretrained ImageNet.

## Dataset

- Sumber: [`manonstr/tipe-webscraping`](https://www.kaggle.com/datasets/manonstr/tipe-webscraping) (Kaggle)
- Total: 7.018 gambar setelah dibersihkan, tersebar di 6 kelas (rasio imbalance ringan, 1,4x)
- Diunduh otomatis lewat `kagglehub`, tidak perlu upload manual

## Metodologi

1. **Pembersihan data** — mapping nama folder (asli berbahasa Prancis) ke label Indonesia, menangani anomali penamaan (folder rusak akibat encoding, file salah label)
2. **EDA** — cek distribusi kelas & variasi resolusi gambar
3. **Preprocessing** — resize ke 224×224, stratified split 70/15/15 (train/val/test), augmentasi pada data training
4. **Baseline** — CNN sederhana dilatih dari nol sebagai pembanding (val accuracy 75,02%)
5. **Transfer Learning** — bandingkan 3 arsitektur pretrained (base layer freeze): MobileNetV2, ResNet50, EfficientNetB0
6. **Fine-Tuning** — unfreeze 20 layer terakhir model terbaik (MobileNetV2), uji learning rate 1e-4 vs 1e-5
7. **Evaluasi** — classification report, confusion matrix, analisis overfitting, sanity-check di folder test bawaan dataset, uji coba gambar dunia nyata

## Hasil

| Tahap | Val Accuracy |
|---|---|
| Baseline CNN | 75,02% |
| Transfer Learning (MobileNetV2, base freeze) | 97,63% |
| Fine-Tuning (MobileNetV2, lr=1e-4) | **98,39%** |

**Test set (1.053 gambar):** accuracy 98%, weighted f1-score 0,98
**Gap train–val:** 0,31% (tidak overfitting)

Detail lengkap analisis dan interpretasi tiap tahap ada di notebook / laporan proyek.

## Struktur Repo

```
.
├── finpro_emel.ipynb          # Notebook utama: EDA, training, evaluasi
├── best_garbage_model.keras   # Model final hasil fine-tuning
└── README.md
```

## Cara Menjalankan

1. Clone repo ini dan buka `finpro_emel.ipynb` di Google Colab atau Jupyter Notebook (disarankan pakai GPU)
2. Jalankan seluruh cell secara berurutan — dataset akan otomatis terunduh lewat `kagglehub`
3. Model final akan tersimpan sebagai `best_garbage_model.keras`

### Memuat Model untuk Prediksi

```python
from tensorflow.keras.models import load_model

model = load_model("best_garbage_model.keras")
# lakukan preprocessing gambar (resize 224x224, rescale) sebelum predict
prediction = model.predict(image_array)
```

## Tools & Library

- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn (evaluasi)
- KaggleHub (download dataset)

## Catatan & Keterbatasan

- Model dievaluasi tinggi (98%) pada test set utama, tapi performanya turun (70,70%) pada folder test bawaan dataset yang jauh lebih kecil dan berbeda distribusinya — indikasi model perlu diuji lebih lanjut pada data dunia nyata yang lebih beragam sebelum deployment penuh.
- Pemilihan model terbaik antara MobileNetV2 dan EfficientNetB0 sempat seri di val accuracy; ke depannya bisa dipertimbangkan tie-breaker eksplisit atau membandingkan hasil fine-tuning keduanya.

## 👤 Author

Final Project — Mata Kuliah Machine Learning  
