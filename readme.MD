# 🚗 Vehicle Detection App
> Capstone Project Module 4 — Purwadhika Data Science & Machine Learning

Aplikasi deteksi kendaraan berbasis **YOLOv12** yang mampu mengenali dan menghitung jumlah **bus**, **car**, dan **van** dari sebuah gambar secara real-time.

---

## 🎯 Use Case

Diberikan sebuah gambar jalan raya, sistem akan:
1. Mendeteksi semua kendaraan yang ada di gambar
2. Mengklasifikasikan tiap kendaraan ke dalam kelas `bus`, `car`, atau `van`
3. Menampilkan jumlah total per kelas

**Contoh output:** `car: 4 · van: 2 · bus: 1`

---

## 🛠️ Tech Stack

| Komponen | Tools |
|---|---|
| Model | YOLOv12n (fine-tuned) |
| Framework | Ultralytics |
| Evaluasi | Supervision (mAP) |
| Aplikasi | Streamlit |
| Training | Lokal |

---

## 📁 Struktur Project

```
project/
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── Train_YOLOv12_Vehicle.ipynb     # Training notebook
├── best_vehicle.pt                 # Trained model weights
├── .gitignore
└── README.md
```

---

## 🚀 Cara Menjalankan Lokal

**1. Clone repository**
```bash
git clone https://github.com/username/vehicle-detection.git
cd vehicle-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Jalankan aplikasi**
```bash
streamlit run app.py
```

**4. Buka browser** di `http://localhost:8501`

> ⚠️ Pastikan file `best_vehicle.pt` ada di folder yang sama dengan `app.py`

---

## 🧠 Model Training

Model dilatih menggunakan **YOLOv12n** dengan fine-tuning pada dataset kendaraan:

| Parameter | Value |
|---|---|
| Epochs | 150 (early stopping patience=10) |
| Batch size | 16 |
| Image size | 640×640 |
| Optimizer | Auto (AdamW) |
| Augmentasi | HSV (hue, saturation, brightness), Random Erasing |

### Augmentasi yang digunakan (non-geometris)
- `hsv_h=0.015` — variasi hue
- `hsv_s=0.7` — variasi saturasi
- `hsv_v=0.4` — variasi brightness
- `erasing=0.4` — random erasing

### Evaluasi (Test Set)

| Metric | Score |
|---|---|
| mAP@0.5 | — |
| mAP@0.5:0.95 | — |
| Precision | — |
| Recall | — |

> ⚠️ *Update tabel di atas setelah training selesai*

---

## 📊 Dataset

- **Format:** YOLOv12 (images + labels + data.yaml)
- **Classes:** `bus`, `car`, `van`
- **Split:** Train / Validation / Test

---

## 🌐 Live Demo

👉 [Buka di Streamlit Cloud](https://share.streamlit.io) *(update link setelah deploy)*

---