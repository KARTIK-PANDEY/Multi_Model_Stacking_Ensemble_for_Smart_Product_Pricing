# 🏆 Multi-Modal Stacking Ensemble for Smart Product Pricing

## ✨ Project Summary

This project presents the high-performance, winning solution for the **Smart Product Pricing Challenge**. The core task was to build a robust machine learning system capable of accurately predicting product prices by integrating complex **multi-modal data**: textual product details and visual product images.

Our solution, the **Multi-Modal Stacking Ensemble**, achieved superior performance by strategically combining specialized models for each data modality. This approach successfully minimized the Symmetric Mean Absolute Percentage Error (SMAPE) across the test set.

| Metric | Final Score (SMAPE) | Improvement Over Baseline |
| :--- | :--- | :--- |
| **Cross-Validation SMAPE** | **17.2%** | **50.9%** |

---

## 🧠 Solution Architecture: The Stacking Advantage

The system operates on a three-tier architecture designed to extract and fuse complementary signals:

### 1. Feature Engineering (The Foundation)
* **Target Stabilization:** The target price was **log-transformed** ($\log(1+price)$) to address skewness and improve model convergence.
* **Crucial IPQ Extraction:** The **Item Pack Quantity (IPQ)** was extracted from the text, confirmed as the most critical numerical feature influencing final cost.
* **Brand Extraction:** The brand name was parsed from the $\texttt{catalog\_content}$ and used as a high-value categorical feature.

### 2. Base Models (Feature Extraction & Prediction)
Specialized deep learning and boosting models were used to generate **Out-Of-Fold (OOF)** predictions, which serve as the "meta-features" for the final stage.

| Model | Modality | Key Algorithm | Role |
| :--- | :--- | :--- | :--- |
| **Tabular** | Engineered Features | **LightGBM** | Excellent at exploiting $\text{IPQ}$ and $\text{Brand}$ feature interactions. |
| **Text** | $\texttt{catalog\_content}$ | **RoBERTa-base** (Fine-tuned) | Used to capture deep semantic meaning and sentiment from product descriptions. |
| **Image** | $\texttt{image\_link}$ | **ResNet50** (Transfer Learning) | Used to extract visual features reflecting product quality, style, and category. |

### 3. Final Ensemble (The Fusion)
* **Method:** **K-Fold Stacking**.
* **Meta-Learner:** A simple, stable **Ridge Regressor** was trained on the OOF predictions of the three base models. This stage learned the optimal non-linear weighting of the predictions, providing the final boost in accuracy.

---

## 📊 Detailed Performance Analysis

The ensemble's success lies in its ability to compensate for the weaknesses of individual models. Performance metrics (from $\texttt{Smart\_Product\_Pricing\_Model\_Evaluation.csv}$) are provided below:

| Model | Feature Set | CV SMAPE (%) | MAE | Improvement Over Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Median Price)** | None | 35.0 | 68.4 | N/A |
| **LightGBM (Tabular)** | Numerical + Categorical Features | 18.5 | 24.7 | 47.1% |
| **RoBERTa-base (Text)** | $\texttt{catalog\_content}$ (Text) | 20.1 | 26.1 | 42.6% |
| **ResNet50 (Image)** | $\texttt{image\_link}$ (Visual Features) | 22.8 | 28.3 | 34.9% |
| **Stacking Ensemble (Final Model)** | **Combined (Text + Image + Tabular)** | **17.2** | **22.9** | **50.9%** |

### Key Takeaways from Error Analysis:
* Feature importance analysis confirmed **'IPQ' and Brand features** contributed most to prediction stability.
* The final $\text{SMAPE}$ of $17.2\%$ is highly competitive, proving the necessity of multi-modal fusion in complex e-commerce pricing tasks.

---

## 🚀 Execution and Repository Details

**All code for this project is consolidated into a single Jupyter Notebook.**

### Files Included in the Upload

* $\texttt{Amazon\_Ai\_ML\_Hackathon.ipynb}$ **(The Complete Codebase)**: Contains all Python logic from data preparation, feature engineering, model training (LGBM, RoBERTa, ResNet), stacking, and submission file generation.
* $\texttt{Smart\_Product\_Pricing\_Challenge\_Final\_Report.pdf}$: Detailed 2-page technical report covering methodology, experiments, and conclusions.
* $\texttt{Smart\_Product\_Pricing\_Model\_Evaluation.csv}$: Raw performance metrics used to build the result tables.

### How to Run the Project

1.  **Preparation:** Ensure your $\texttt{train.csv}$ and $\texttt{test.csv}$ files are placed in the working directory alongside the notebook.
2.  **Environment:** Open $\texttt{Amazon\_Ai\_ML\_Hackathon.ipynb}$ in a Jupyter environment. Ensure required libraries ($\text{torch, transformers, lightgbm, etc.}$) are installed.
3.  **Execution:** Run all cells sequentially. The notebook's execution will:
    * Set up helper functions (SMAPE, IPQ extraction).
    * Download and process images.
    * Extract features and embeddings.
    * Train the base models using K-Fold Cross-Validation.
    * Train the final stacking meta-learner.
    * Generate the final submission file: $\texttt{test\_out.csv}$.

### Future Directions (From Final Report)
* Explore direct **SMAPE-optimized neural loss functions**.
* Experiment with **multimodal attention fusion networks** (e.g., CLIP-based models).
* Extend feature extraction to include product review text sentiment as an auxiliary input.
