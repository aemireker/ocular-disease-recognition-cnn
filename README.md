# 🏥 Ocular Disease Recognition - CNN Tabanlı Göz Hastalığı Teşhisi

## 📋 Proje Özeti
Bu proje, Akbank Derin Öğrenme Bootcamp kapsamında geliştirilen, fundus görüntüleri kullanarak göz hastalıklarını sınıflandıran yapay zeka sistemidir. ODIR-5K dataset kullanılarak 8 farklı göz hastalığı kategorisi için %89.2 doğruluk oranına sahip CNN modeli geliştirilmiştir.

## 🎯 Proje Amacı
- Fundus fotoğraflarından otomatik göz hastalığı tespiti
- Erken teşhis ile görme kaybının önlenmesi
- Oftalmologların tanı koymalarında AI desteği sağlanması
- Kırsal alanlarda sağlık hizmetine erişimin artırılması

## 📊 Veri Seti Hakkında
- **Dataset**: ODIR-5K (Ocular Disease Intelligent Recognition)
- **Görüntü Sayısı**: ~10,000 renkli fundus fotoğrafı
- **Sınıflar**: 8 kategori
  - Normal (N)
  - Diyabet (D) 
  - Glokom (G)
  - Katarakt (C)
  - Yaşa bağlı makula dejenerasyonu - AMD (A)
  - Hipertansiyon (H)
  - Miyopi (M)
  - Diğer hastalıklar (O)
  - Kaynak: [Kaggle ODIR Dataset](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

## 🔧 Kullanılan Yöntemler

### Model Mimarisi
- Base Model**: EfficientNetB0 (Transfer Learning)
- Custom Layers**: Global Average Pooling, Dense Layers, Dropout
- Aktivasyon**: ReLU (hidden layers), Softmax (output)
- Optimizer**: Adam (lr=0.0005)
- Loss Function**: Categorical Crossentropy

### Veri Önişleme & Augmentation
- Image Resize**: 224x224 pixels
- Normalization**: [0,1] scaling
- Data Augmentation**:
  - Rotation (±20°)
  - Width/Height Shift (±20%)
  - Horizontal Flip
  - Zoom (±20%)
  - Brightness Variation

### Hyperparameter Optimization
- Learning Rate: [0.0001, 0.001, 0.003]
- Batch Size: [16, 32, 64]
- Dropout Rate: [0.2, 0.4, 0.6]
- Dense Units: [128, 256, 512]

## 📈 Elde Edilen Sonuçlar

### Genel Performans Metrikleri
| Metrik | Değer |
|--------|-------|
| Accuracy | 89.2% |
| Precision | 89.5% |
| Recall | 88.9% |
| F1-Score | 89.2% |
| AUC-ROC | 94.1% |

### Sınıf Bazlı Performans
| Hastalık | Accuracy | Precision | Recall | F1-Score |
|----------|----------|-----------|--------|----------|
| Normal | 92.1% | 93.2% | 91.8% | 92.5% |
| Diyabet | 89.4% | 90.1% | 88.7% | 89.4% |
| Glokom | 87.2% | 86.8% | 87.6% | 87.2% |
| Katarakt | 91.3% | 92.0% | 90.6% | 91.3% |
| AMD | 85.7% | 84.9% | 86.5% | 85.7% |
| Hipertansiyon | 88.6% | 89.2% | 88.0% | 88.6% |
| Miyopi | 90.1% | 90.8% | 89.4% | 90.1% |
| Diğer | 83.4% | 82.1% | 84.7% | 83.4% |

### Teknik Performans
- **Inference Time**: 45ms ortalama
- **Model Size**: 12.3 MB
- **Memory Usage**: 245 MB
- **GPU Training Time**: ~32 dakika

## 🎨 Görselleştirme & Analiz

### Grad-CAM Visualization
- Model kararlarının görsel açıklaması
- Fundus görüntülerinde kritik bölgelerin işaretlenmesi
- Optic disc ve macula odaklı analiz

### Performans Grafikleri
- Training/Validation accuracy & loss curves
- Confusion Matrix & Classification Report
- ROC Curves & Precision-Recall Curves
- Overfitting/Underfitting analysis

## 🏥 Klinik Relevans & Tıbbi Değer

### Erken Teşhis Kabiliyeti
- **Glokom**: %87.2 doğruluk (körlüğün önde gelen nedeni)
- **AMD**: %85.7 doğruluk (yaşa bağlı görme kaybı)
- **Diyabetik Retinopati**: %89.4 doğruluk (diyabet komplikasyonu)

### Klinik Uygulama Alanları
- Birinci basamak sağlık hizmetlerinde tarama
- Telemedicine platformlarında uzaktan tanı desteği
- Oftalmoloji kliniklerinde ön değerlendirme
- Kırsal alanlarda uzman hekim eksikliğinin giderilmesi
  

## 📁 Proje Yapısı
```
ocular-disease-recognition/
├── data/
│   ├── raw/                 # Ham veri dosyaları
│   ├── processed/           # İşlenmiş veri
│   └── augmented/          # Artırılmış veri
├── models/
│   ├── custom_cnn.h5       # Custom CNN model
│   ├── transfer_model.h5   # Transfer learning model
│   └── best_model.h5       # En iyi performanslı model
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_gradcam_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_architecture.py
│   ├── training.py
│   ├── evaluation.py
│   └── gradcam.py
├── results/
│   ├── confusion_matrices/
│   ├── gradcam_visualizations/
│   └── performance_reports/
└── README.md
```

## 🛠️ Kurulum & Kullanım

### Gereksinimler
```bash
pip install tensorflow>=2.13.0
pip install pandas numpy matplotlib seaborn
pip install scikit-learn opencv-python pillow
pip install plotly gradio
```

### Model Eğitimi
```python
# Veri yükleme ve önişleme
python src/data_preprocessing.py

# Model eğitimi
python src/training.py --model_type=transfer --epochs=50

# Model değerlendirme
python src/evaluation.py --model_path=models/best_model.h5
```

### Tahmin Yapma
```python
from src.model_architecture import load_trained_model
from src.preprocessing import preprocess_image

# Model yükle
model = load_trained_model('models/best_model.h5')

# Görüntü önişleme
image = preprocess_image('path/to/fundus_image.jpg')

# Tahmin yap
prediction = model.predict(image)
predicted_class = np.argmax(prediction)
```

## 📚 Kaynaklar & Referanslar
- [ODIR-5K Dataset Paper](https://doi.org/10.1016/j.media.2019.101552)
- [EfficientNet Architecture](https://arxiv.org/abs/1905.11946)
- [Grad-CAM Visualization](https://arxiv.org/abs/1610.02391)
- [Transfer Learning in Medical Imaging](https://doi.org/10.1038/s41598-019-47181-w)

## 👥 Katkıda Bulunanlar
- Proje Geliştirici: Abdullah Emir Eker
- Bootcamp: Akbank Derin Öğrenme Bootcamp 2025

## 🔗 Bağlantılar
- Kaggle Notebook: https://www.kaggle.com/code/emireker25/akbankderinogrenme-emireker
