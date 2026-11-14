# Macro-Driven Bank Telemarketing & Customer Re-engagement Analysis

## 1. Project Overview
This project investigates how macroeconomic cycles moderate the effectiveness of direct telemarketing campaigns for bank term deposits using the **Bank Marketing Dataset** (41,188 call records during the 2008–2010 financial crisis). Following the **KDD (Knowledge Discovery in Databases)** framework, we combine unsupervised clustering, non-parametric statistical testing, and supervised machine learning to optimize customer targeting strategies.


## 2. Methodology & Pipeline
1. **Data Preprocessing (`src/01_data_preparation.py`):** Filtered invalid calls (`duration = 0`), encoded ordinal/binary variables, and engineered behavioral bins (`duration_group`, `campaign_group`).
2. **Macroeconomic Regime Clustering (`src/02_macro_clustering.py`):** Standardized 4 core macroeconomic indicators (`euribor3m`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`) and applied **K-Means & Hierarchical Clustering** (validated via Elbow & Silhouette analysis) to identify 3 economic regimes: **Growth**, **Recession**, and **Trough**.
3. **Campaign Driver Analysis (`src/03_campaign_analysis_rq2.py`):** Applied **Chi-square, Kruskal-Wallis, Mann-Whitney U tests, Cohen's d**, and regime-specific **Decision Trees** (5-Fold Stratified CV) to evaluate channel, timing, call frequency, and consultation duration effects.
4. **Re-engagement & Interaction Modeling (`src/04_reapproach_simpson_rq3.py`):** Built a **Logistic Regression** model with interaction terms (`poutcome × economic_phase`) to quantify re-engagement effectiveness across business cycles.

## 3. Key Findings & Business Insights
* **Macroeconomic Regimes Matter:** Baseline term-deposit conversion shifts drastically across regimes—from **4.83%** in *Growth* to **20.07%** in *Recession* and **42.56%** in *Trough* ($\chi^2 = 4564.57, p < 0.001$).
* **Simpson's Paradox in Customer Re-engagement:** Aggregate data falsely suggests previously rejected leads convert better than new prospects (14.23% vs. 8.83%). Controlling for economic phases reveals that **first-time prospects consistently outperform previously rejected leads across all 3 regimes** (e.g., 18.02% vs. 12.42% in *Recession*).
* **Strong Predictive Performance:** The Logistic Regression interaction model achieved a 5-Fold Cross-Validation **AUC-ROC of 0.9144** (0.9150 on full dataset).

## 4. Visual Highlights
| Macroeconomic Cluster Profile | Simpson's Paradox Distribution | Logistic Regression ROC Curve |
| :---: | :---: | :---: |
| ![Cluster Profile](./images/cluster_profile_heatmap.png) | ![Simpson Paradox](./images/distribution_simpson.png) | ![ROC Curve](./images/RQ3_ROC_Curve_Result.png) |

## 5. Tech Stack
* **Language:** Python 3
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`
