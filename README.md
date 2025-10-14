# 🏆 Smart Product Pricing Challenge — Multi-Modal Stacking Ensemble

## 🚀 Project Overview

This repository presents our end-to-end Machine Learning solution for the **Smart Product Pricing Challenge**, where the goal is to **predict optimal product prices** using a combination of **textual product descriptions** and **visual product images**.

We designed a **Multi-Modal Stacking Ensemble** that integrates:
- **RoBERTa-based NLP model** for semantic understanding of product descriptions,  
- **ResNet50 CNN** for visual feature extraction, and  
- **LightGBM** for engineered tabular features.

These models were stacked using a **Ridge Regression meta-learner**, achieving a significant **50.9% improvement over the baseline** in terms of SMAPE.

---

## ✨ Highlights

- 🔍 **Multi-modal Learning** — Combines text, image, and structured data.
- 🧠 **Stacking Ensemble Framework** — Integrates three complementary models.
- ⚙️ **Optimized for Generalization** — Achieved state-of-the-art cross-validation performance.
- 📈 **Performance Improvement** — +50.9% better SMAPE over baseline.

---

## 🧩 Problem Statement

Predicting accurate product prices is challenging due to variability in:
- Product descriptions (semantic context),
- Product images (visual appearance),
- Missing or inconsistent tabular features (like brand or packaging).

This solution builds a unified learning pipeline that **captures semantic, visual, and numeric relationships** to generate more realistic price predictions.

---

## 🧠 Architecture Overview

Our architecture follows a **three-tier modular structure**, designed for efficiency and interpretability.

### 🔹 1. Feature Engineering
- **Log transformation** applied on price: `log(1 + price)` for target stabilization.
- **Item Pack Quantity (IPQ)** extracted using RegEx from product names.
- Derived **Brand**, **Description Length**, and **Category** features.
- Normalized all numerical features for robust training.

### 🔹 2. Base Models
Each base model learns from a specific data modality and generates out-of-fold (OOF) predictions:

| Model | Data Type | Algorithm | Description |
| :--- | :--- | :--- | :--- |
| **LightGBM** | Tabular | Gradient Boosting | Learns interactions between IPQ, Brand, and Category features. |
| **RoBERTa-base** | Text | Transformer (Hugging Face) | Captures semantic meaning of product descriptions. |
| **ResNet50** | Image | CNN (Transfer Learning) | Extracts visual patterns like color, material, and packaging. |

### 🔹 3. Stacking Ensemble
- Combined the OOF predictions from all base models.
- Trained a **Ridge Regressor** as the meta-learner.
- Used **K-Fold stacking** to ensure stable blending and reduce overfitting.

---

## 📊 Model Evaluation

We evaluated each model using **SMAPE, MAE, and RMSE** across 5-fold cross-validation.  
The stacked ensemble achieved the best overall performance.

| Model | Data Type | CV SMAPE (%) | MAE | RMSE | Improvement Over Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Median Price)** | None | 35.0 | 68.4 | 84.1 | — |
| **LightGBM** | Tabular | 18.5 | 24.7 | 31.5 | 47.1% |
| **RoBERTa-base** | Text | 20.1 | 26.1 | 33.2 | 42.6% |
| **ResNet50** | Image | 22.8 | 28.3 | 36.4 | 34.9% |
| **Stacking Ensemble (Final)** | Multi-Modal | **17.2** | **22.9** | **29.8** | **50.9%** |

📂 *All evaluation data can be found in:*  
[`Smart_Product_Pricing_Model_Evaluation.csv`](./Smart_Product_Pricing_Model_Evaluation.csv)

---

## 🧮 Methodology Summary

| Stage | Task | Technique / Model | Output |
| :--- | :--- | :--- | :--- |
| **Data Preparation** | Cleaning, transformation | Pandas, NumPy | Structured dataset |
| **Feature Extraction (Text)** | Tokenization & embedding | RoBERTa-base | 768-d text embeddings |
| **Feature Extraction (Image)** | CNN feature extraction | ResNet50 (pretrained) | 2048-d visual embeddings |
| **Feature Engineering (Tabular)** | IPQ, Brand, Category | Custom feature scripts | Normalized feature set |
| **Model Training** | LightGBM, RoBERTa, ResNet50 | 5-Fold CV | OOF predictions |
| **Model Fusion** | Meta-learning via Ridge Regression | Scikit-learn | Final stacked predictions |

---

## 📁 Repository Structure

