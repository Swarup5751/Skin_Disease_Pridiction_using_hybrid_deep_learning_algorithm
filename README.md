# 🧠 Skin Disease Prediction using Hybrid Deep Learning

## ResNet50 + Vision Transformer (ViT-B16)

---

## 📌 Overview

This project presents a **Hybrid Deep Learning Architecture** that combines:

* **ResNet50 (CNN)** → Local spatial & texture feature extraction
* **Vision Transformer (ViT-B16)** → Global contextual modeling using self-attention

The model performs **7-class skin lesion classification** using the **HAM10000 dataset** and achieves high accuracy with improved performance on rare lesion types.

This work was developed as part of a B.Tech Capstone Project (Information Technology).

---

## 📊 Dataset

**Dataset Used:** HAM10000 (Human Against Machine with 10,000 training images)

* Total original images: 10,015
* Number of classes: 7

### Classes:

1. Melanocytic Nevi
2. Melanoma
3. Benign Keratosis
4. Basal Cell Carcinoma
5. Actinic Keratoses
6. Vascular Lesions
7. Dermatofibroma

---

## ⚖ Dataset Balancing

The dataset was highly imbalanced (>65% Nevi class).

To handle this:

* Data augmentation applied to minority classes:

  * Random rotation (±25°)
  * Horizontal/vertical flip
  * Zoom (0.8–1.2x)
  * Brightness adjustment
  * Width/height shift
* Controlled reduction of majority class

### Final Balanced Dataset:

* 1,500 images per class
* Total images after balancing: **10,500**

---

# 🏗 Model Architecture

## 🔹 Input Preprocessing

* Image Size: **224 × 224 × 3**
* Normalization: ImageNet mean & standard deviation
* Pixel scaling: 0–1 range

---

## 🔹 Hybrid Architecture

### 1️⃣ ResNet50 Branch

* Pretrained on ImageNet
* `include_top=False`
* Global Average Pooling
* Output Feature Size: **2048**
* Activation: ReLU (internal layers)

Purpose:

* Extract fine-grained local spatial features
* Capture texture and lesion patterns

---

### 2️⃣ Vision Transformer (ViT-B16)

* Patch size: 16 × 16
* Multi-head self-attention (12 heads)
* Transformer encoder blocks
* Output Feature Size: **768**

Purpose:

* Capture global contextual relationships
* Model long-range spatial dependencies

---

### 3️⃣ Feature Fusion

```
2048 (ResNet) + 768 (ViT) = 2816-dimensional feature vector
```

Features are concatenated and passed to dense layers.

---

### 4️⃣ Classification Head

```
Dense(512, activation='relu')
Dropout(0.3)
Dense(7, activation='softmax')
```

Output: 7-class probability distribution

---

# ⚙ Training Configuration

| Parameter        | Value                      |
| ---------------- | -------------------------- |
| Optimizer        | Adam                       |
| Learning Rate    | 1e-4                       |
| Weight Decay     | 1e-5                       |
| Batch Size       | 32                         |
| Epochs           | 25                         |
| Loss Function    | Categorical Crossentropy   |
| Early Stopping   | Patience = 10              |
| Train Split      | 70%                        |
| Validation Split | 15%                        |
| Test Split       | 15%                        |
| GPU Used         | NVIDIA A100 (Google Colab) |

---

# 📈 Model Performance

| Metric         | Value |
| -------------- | ----- |
| Test Accuracy  | ~95%  |
| Macro F1 Score | 0.91  |
| ROC-AUC (Mean) | 0.93  |

### Improvements Observed:

* Better rare class detection (Dermatofibroma & Vascular Lesions)
* Reduced class imbalance bias
* Improved generalization compared to single-model baselines

---

# 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* Saliency Maps
* Attention Heatmaps

---

# 🌐 Web Application (Flask Deployment)

The trained model is deployed using **Flask** for real-time inference.

Features:

* Upload dermoscopic image
* Predict disease class
* Display confidence score
* Visual output interface

---

# 📁 Project Structure

```
Skin_Disease_Prediction/
│
├── capston.py
├── Capston.ipynb
├── app.py
├── my_model.h5
├── README.md
│
├── static/
├── templates/
│
├── research_paper.pdf
└── Blackbook_final_G12_BtechIT.pdf
```

---

# 🚀 How To Run

### 1️⃣ Clone Repository

```
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run Flask App

```
python app.py
```

---

# 🧪 Technologies Used

* Python 3.8+
* TensorFlow / Keras
* vit_keras
* OpenCV
* scikit-learn
* Flask
* Matplotlib
* Seaborn
* Google Colab

---

# 💡 Key Contributions

✔ Hybrid CNN + Transformer architecture
✔ Improved rare lesion classification
✔ Class imbalance handling
✔ Interpretability through attention visualization
✔ Reproducible training pipeline
✔ Real-time deployment

---

# 🔮 Future Work

* Cross-dataset validation (ISIC 2018 / 2019)
* Multi-modal integration (clinical metadata)
* Lightweight mobile model
* Edge deployment optimization
* Larger dermatology dataset testing

---

# 👨‍💻 Author

**Swarup Sonawane**
B.Tech – Information Technology
SVKM’s NMIMS, Shirpur

---

# ⭐ If You Found This Project Useful

Please consider giving it a ⭐ on GitHub.

