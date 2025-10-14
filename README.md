# 🏆 Multi-Modal Stacking Ensemble for Smart Product Pricing

## ✨ Project Summary

This repository contains the high-performance solution for the **Smart Product Pricing Challenge**. The objective was to build a robust machine learning system capable of accurately predicting product prices by integrating **multi-modal data**: textual product descriptions ($\texttt{catalog\_content}$) and visual product images ($\texttt{image\_link}$).

Our solution, a **Multi-Modal Stacking Ensemble**, strategically fused specialized deep learning and boosting models to achieve exceptional accuracy and generalization.

### Key Results

The solution demonstrated a **50.9% improvement** over the baseline model, achieving a highly competitive SMAPE score by minimizing prediction error across the entire test set.

| Metric | Final Score (SMAPE) | Improvement Over Baseline |
| :--- | :--- | :--- |
| **Cross-Validation SMAPE** | **17.2%** | **50.9%** |

---

## 🧠 Architecture Overview: Fusing Semantic and Visual Signals

The system employs a sophisticated three-tier architecture to maximize predictive signal from each data type:

### 1. Feature Engineering (The Foundation)
* **Target Stabilization:** The highly skewed $\texttt{price}$ was successfully stabilized using a **Log Transformation** ($\log(1+price)$).
* **Critical Feature Extraction:** The **Item Pack Quantity (IPQ)** was extracted via RegEx and confirmed as the most critical numerical feature. Brand and text length features were also derived.

### 2. Base Models (Specialized Predictors)

Specialized models were trained to generate Out-Of-Fold (OOF) predictions, providing a rich set of "meta-features" for the final ensemble.

| Model | Modality | Key Algorithm | Role in the Ensemble |
| :--- | :--- | :--- | :--- |
| **LightGBM** | Engineered Tabular Features | Gradient Boosting Machine | Exploits interaction between **IPQ** and **Brand** features. |
| **RoBERTa-base** | Text ($\texttt{catalog\_content}$) | Fine-tuned Transformer | Captures deep **semantic meaning** and context from descriptions. |
| **ResNet50** | Image ($\texttt{image\_link}$) | Transfer Learning (CNN) | Extracts **visual cues** related to quality, material, and category. |

### 3. Final Ensemble (The Fusion Layer)
* **Method:** **K-Fold Stacking**.
* **Meta-Learner:** A simple, stable **Ridge Regressor** was trained on the OOF predictions of the three base models. This stage determined the optimal weighting for the final prediction, resulting in the minimum SMAPE score.

---

## 📊 Detailed Performance Metrics

The following metrics were derived from the cross-validation performance, confirming the synergistic effect of the stacking approach (data source: $\texttt{Smart\_Product\_Pricing\_Model\_Evaluation.csv}$).

| Model | Feature Set | CV SMAPE (%) | MAE | RMSE | Improvement Over Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Median Price)** | None | 35.0 | 68.4 | 84.1 | N/A |
| **LightGBM (Tabular)** | Numerical + Categorical | 18.5 | 24.7 | 31.5 | 47.1% |
| **RoBERTa-base (Text)** | $\texttt{catalog\_content}$ | 20.1 | 26.1 | 33.2 | 42.6% |
| **ResNet50 (Image)** | $\texttt{image\_link}$ | 22.8 | 28.3 | 36.4 | 34.9% |
| **Stacking Ensemble (Final Model)** | **Combined Multi-Modal** | **17.2** | **22.9** | **29.8** | **50.9%** |

---

## 💻 Code Structure and Execution

**All code for this project is consolidated into the single Jupyter Notebook:** $\texttt{Amazon\_Ai\_ML\_Hackathon.ipynb}$.

### Files Included

* $\texttt{Amazon\_Ai\_ML\_Hackathon.ipynb}$: **The complete, end-to-end executable code pipeline.**
* $\texttt{Smart\_Product\_Pricing\_Challenge\_Final\_Report.pdf}$: The official 2-page technical report detailing the methodology.
* $\texttt{Smart\_Product\_Pricing\_Model\_Evaluation.csv}$: Raw performance metrics table.

### How to Run the Project

1.  **Preparation:** Ensure your $\texttt{train.csv}$ and $\texttt{test.csv}$ files are placed in the same directory as the notebook.
2.  **Environment:** Open $\texttt{Amazon\_Ai\_ML\_Hackathon.ipynb}$ in a Jupyter environment (e.g., JupyterLab or VS Code). A **GPU is strongly recommended** to speed up the RoBERTa and ResNet feature extraction steps.
3.  **Dependencies:** Run the initial setup cells to install all required libraries ($\text{torch, transformers, lightgbm, etc.}$).
4.  **Execution:** **Run all notebook cells sequentially.** The notebook's comprehensive structure executes the following:
    * Data loading and preprocessing.
    * IPQ extraction and tabular feature creation.
    * Image downloading and ResNet feature extraction.
    * RoBERTa tokenization and embedding generation.
    * K-Fold training for all three base models.
    * Stacking ensemble training.
    * Final prediction generation and saving to $\texttt{test\_out.csv}$.
