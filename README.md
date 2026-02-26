🧠 Skin Disease Prediction using Hybrid Deep Learning (ResNet50 + ViT)
📌 Overview

This project presents a Hybrid Deep Learning Architecture combining ResNet50 (CNN) and Vision Transformer (ViT-B16) for multi-class skin lesion classification using the HAM10000 dataset.

The model leverages:

🧩 Local feature extraction (ResNet50)

🌍 Global contextual modeling (Vision Transformer)

🔗 Feature fusion for improved classification

📄 Project Report: 

Blackbook_final_G12_BtechIT


📄 Research Paper: 

research_paper

📊 Dataset

HAM10000 Dataset

Total images: 10,015

Classes: 7

Melanocytic Nevi

Melanoma

Benign Keratosis

Basal Cell Carcinoma

Actinic Keratoses

Vascular Lesions

Dermatofibroma

⚖ Class Balancing

Augmentation applied to minority classes

Final balanced dataset:

1,500 images per class

Total: 10,500 images

🏗 Model Architecture
🔹 Input

Image size: 224 × 224 × 3

Normalization: ImageNet mean & std

Augmentation:

Rotation (±25°)

Flip

Zoom (0.8–1.2)

Brightness adjustment

Width/height shift

🔹 Hybrid Architecture
1️⃣ ResNet50 Branch

Pretrained on ImageNet

include_top=False

Global Average Pooling

Output: 2048-dimensional feature vector

Activation used internally: ReLU

2️⃣ Vision Transformer (ViT-B16)

Patch size: 16 × 16

Multi-head self-attention: 12 heads

Transformer encoder blocks

Output: 768-dimensional feature vector

3️⃣ Feature Fusion

Concatenation Layer

Final feature size:

2048 (ResNet) + 768 (ViT) = 2816 features
4️⃣ Classification Head
Dense(512, activation='relu')
Dropout(0.3)
Dense(7, activation='softmax')
⚙ Training Configuration
Parameter	Value
Optimizer	Adam
Learning Rate	1e-4
Weight Decay	1e-5
Batch Size	32
Epochs	25
Loss Function	Categorical Crossentropy
Early Stopping	Patience = 10
Train Split	70%
Validation Split	15%
Test Split	15%
GPU	NVIDIA A100 (Google Colab Pro)
📈 Model Performance
✅ Final Metrics
Metric	Value
Test Accuracy	95.6%
Macro F1 Score	0.91
ROC-AUC	0.93
📌 Improvements Over Single Models

Better rare class detection

Higher Dermatofibroma & Vascular Lesion recall

Reduced class imbalance bias

🖥 Web Application

Built using Flask:

Upload dermoscopy image

Real-time prediction

Confidence score display

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC Curve

Saliency Maps

Attention Heatmaps
