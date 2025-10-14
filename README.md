# 🏆 Smart Product Pricing Challenge — Multi-Modal Stacking Ensemble

> A Deep Learning and Ensemble-Based Approach for Intelligent Product Price Prediction

---

## 🚀 Overview

This repository presents our **complete Machine Learning solution** for the **Smart Product Pricing Challenge**, where the goal was to **predict product prices** using a combination of **text data** (catalog content) and **image data** (product images).

Our solution — **Multi-Modal Stacking Ensemble for Smart Product Pricing** — integrates insights from three key data sources:
- 🧾 Textual Product Descriptions  
- 🖼️ Product Images  
- 📊 Tabular Attributes (like Brand, Pack Size, and Category)

We developed a **multi-modal fusion architecture** that combines **RoBERTa (text)**, **ResNet50 (image)**, and **LightGBM (tabular)** models into a **stacked ensemble**, achieving a **50.9% improvement over the baseline** on the SMAPE metric.

---

## 🎯 Problem Statement

The **Smart Product Pricing Challenge** aims to build a predictive model capable of determining the **optimal selling price** for products based on their descriptions and images.

In the e-commerce domain, determining the right product price is complex because:
- Text descriptions contain rich yet **ambiguous language** about product quality and features.
- Product images hold **visual cues** like material, color, and packaging.
- Tabular metadata is often **incomplete or noisy**.

Thus, the challenge required an **AI-driven system** that could:
1. Learn from **multi-modal data sources**,  
2. Generalize well across diverse product categories, and  
3. Minimize **SMAPE (Symmetric Mean Absolute Percentage Error)** for reliable predictions.

---

## 💡 Approach and Methodology

We designed a **three-stage pipeline** that processes and fuses textual, visual, and tabular data into one robust ensemble framework.

---

### 🔹 Stage 1: Data Understanding & Preprocessing

**Goal:** Clean, normalize, and prepare all modalities for feature extraction.

| Data Type | Key Steps |
| :--- | :--- |
| **Text (catalog_content)** | Lowercasing, punctuation removal, and tokenization using `transformers` tokenizer. |
| **Image (image_link)** | Downloaded and resized to 224x224; normalized using ImageNet statistics. |
| **Tabular Features** | Extracted numerical and categorical features like *Item Pack Quantity (IPQ)* and *Brand*. |

**Target Transformation:**  
Prices were highly skewed, so a **log(1 + price)** transformation was applied for stability.

---

### 🔹 Stage 2: Feature Engineering

We focused on creating strong hand-crafted features before deep learning extraction:

- **Item Pack Quantity (IPQ):** Extracted using RegEx from the product name (e.g., “Pack of 6”).  
- **Text Length:** Number of words and characters in the description.  
- **Brand Frequency:** Encoded via frequency counts.  
- **Category Embeddings:** Derived via one-hot encoding.  

---

### 🔹 Stage 3: Model Building

We trained **three specialized base learners**, each focusing on one data type.

| Model | Modality | Architecture | Objective |
| :--- | :--- | :--- | :--- |
| **RoBERTa-base** | Text | Transformer (Hugging Face) | Capture semantic and contextual meaning from catalog descriptions. |
| **ResNet50** | Image | Pre-trained CNN (PyTorch) | Extract deep visual features capturing product quality and category cues. |
| **LightGBM** | Tabular | Gradient Boosting Trees | Learn numerical & categorical interactions from engineered features. |

Each model generated **Out-of-Fold (OOF) predictions**, which served as **meta-features** for the final ensemble model.

---

### 🔹 Stage 4: Stacking Ensemble Fusion

To combine the predictive power of all models, we employed a **stacking ensemble**:

- **Meta-Learner:** Ridge Regression (L2 Regularized)  
- **Training Method:** 5-Fold Stacking  
- **Input:** OOF predictions from RoBERTa, ResNet50, and LightGBM  
- **Output:** Final predicted price (after inverse log transformation)

This ensemble effectively balanced the strengths of each model, improving robustness and minimizing overfitting.

---


---

## 📊 Model Evaluation

| Model | Feature Type | CV SMAPE (%) | MAE | RMSE | Improvement Over Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Median Price)** | None | 35.0 | 68.4 | 84.1 | — |
| **LightGBM** | Tabular | 18.5 | 24.7 | 31.5 | 47.1% |
| **RoBERTa-base** | Text | 20.1 | 26.1 | 33.2 | 42.6% |
| **ResNet50** | Image | 22.8 | 28.3 | 36.4 | 34.9% |
| **Final Stacking Ensemble** | Multi-Modal | **17.2** | **22.9** | **29.8** | **50.9%** |

📄 *Detailed performance metrics are available in:*  
[`Smart_Product_Pricing_Model_Evaluation.csv`](./Smart_Product_Pricing_Model_Evaluation.csv)

---

## 🧮 Experimental Setup

| Component | Description |
| :--- | :--- |
| **Hardware** | NVIDIA T4 GPU, 16GB RAM |
| **Frameworks** | PyTorch, Transformers, LightGBM, Scikit-learn |
| **Validation Strategy** | 5-Fold Stratified Cross Validation |
| **Loss Function** | SMAPE |
| **Optimizer** | AdamW (for RoBERTa, ResNet50), Default (for LightGBM) |



---

## ⚙️ How to Run the Project

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/Smart-Product-Pricing.git
cd Smart-Product-Pricing

 bash```

Explore the Code:-
jupyter notebook Amazon_Ai_ML_Hackathon.ipynb
