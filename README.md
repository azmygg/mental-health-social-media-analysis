# 🧠 Mental Health & Social Media — Data Mining Analysis

> Identifying behavioral criteria of mental health disorders linked to social media usage and physical lifestyle indicators across 5,000 individuals, using data mining and ML techniques.

---

## 📌 Overview

This project presents a complete end-to-end data mining pipeline that investigates the relationship between **social media usage patterns**, **physical lifestyle habits**, and **mental health outcomes**. By applying a combination of unsupervised learning, fuzzy reasoning, and evolutionary optimization, the system is capable of profiling users into risk groups and generating personalized mental health risk scores.

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

**Libraries used:**
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `sklearn-extra` · `skfuzzy` · `scipy`

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Records** | 5,000 individuals |
| **Features** | 15 (mix of numeric and categorical) |
| **Domain** | Mental Health & Digital Behavior |
| **Source** | Public dataset (Kaggle) |
| **Target Variable** | `mental_state` — Healthy / Stressed / At_Risk |

**Key features include:** age, daily screen time, social media time, sleep hours, physical activity, anxiety level, stress level, mood level, gender, and platform.

---

## 🔬 Project Pipeline

```
Raw Data
   │
   ▼
📊 Exploratory Data Analysis & Visualization
   │
   ▼
🔧 Data Preprocessing
   (Outlier treatment · Feature engineering · Encoding · Scaling)
   │
   ▼
🧬 Genetic Algorithm — Feature Selection
   (Selects optimal feature subset using silhouette score as fitness)
   │
   ├──────────────────────────┐
   ▼                          ▼
🔵 K-Medoids Clustering    🟢 Hierarchical Clustering
   (k=3, PAM method)          (Ward · Complete · Average linkage)
   │                          │
   └──────────┬───────────────┘
              ▼
        🟡 Cluster Profiles
        (Healthy · Moderate · High Risk)
              │
              ▼
        🔮 Fuzzy Logic Inference System
        (Screen Time + Sleep Hours → Risk Score 0–100)
              │
              ▼
        📋 Mental Health Assessment Report
        (Cluster label · Risk score · Recommendations)
```

---

## 🛠️ Tasks Breakdown

### 📊 Task 1 — Exploratory Data Analysis
- 7 visualizations covering distributions, correlations, and categorical breakdowns
- Correlation heatmap revealing strong links between screen time, anxiety, stress, and sleep
- Platform-level and gender-level mental state breakdowns

### 🔧 Task 2 — Data Preprocessing
- IQR-based outlier detection with domain-justified treatment decisions
- 4 engineered features: `social_media_ratio`, `net_interaction_score`, `sleep_deficit`, `activity_per_screen_hour`
- One-Hot Encoding for gender, platform, and mental state
- StandardScaler applied to all numeric features

### 🧬 Task 6 — Genetic Algorithm (Feature Selection)
- **Chromosome:** Binary array of length 24 (one bit per candidate feature)
- **Gene:** Single bit — 1 = feature selected, 0 = excluded
- **Selection:** Roulette Wheel Selection
- **Crossover:** Single-point crossover
- **Mutation:** Bit-flip with 10% probability
- **Fitness Function:** Silhouette score from AgglomerativeClustering (k=2)
- **Result:** Optimal feature subset selected from 24 candidates across 30 generations

### 🔵 Task 3 — K-Medoids Clustering
- Implemented using `sklearn_extra.KMedoids` (PAM method)
- Optimal k=3 selected via Elbow Method + domain knowledge
- t-SNE and multi-panel cluster visualizations

| Cluster | Label | Profile |
|---|---|---|
| 0 | Moderate Stress | Mid screen time, adequate sleep, moderate anxiety |
| 1 | Healthy / Resilient | Low screen time, best sleep, lowest anxiety, oldest age group |
| 2 | High Risk | Highest screen time, poorest sleep, highest anxiety, youngest |

### 🟢 Task 4 — Hierarchical Clustering
- Three linkage methods compared: Ward, Complete, Average
- Dendrogram visualization with truncation
- ARI score computed against K-Medoids to validate consistency

### 🔮 Task 5 — Fuzzy Logic Inference System
- **Inputs:** Daily Screen Time · Sleep Hours
- **Output:** Mental Health Risk Score (0–100)
- **9 IF-THEN rules** grounded in clinical domain knowledge
- Centroid defuzzification method
- Validated against all 5,000 dataset records
- Cluster assignment used as contextual validation layer

### 📋 Task 7 — System Implementation Pipeline
- `MentalHealthPipeline` class integrating all components
- Takes a new patient record and returns cluster label, risk score, risk category, and recommendations
- Visual flowchart + ASCII pipeline diagram included

---

## 📈 Key Findings

- **93%** of participants are classified as Stressed, with heavy concentration in younger age groups
- **Screen time and sleep** are the strongest behavioral predictors of mental health risk
- **Cluster 2 (High Risk)** users average 450 min/day screen time and only 6.75 hours of sleep
- **Cluster 1 (Healthy)** users are significantly older (avg 47 years) with the lowest digital engagement
- **Fuzzy system validation:** risk scores align correctly with cluster profiles across all 5,000 records
- **GA optimization** improved clustering silhouette score progressively across 30 generations

---

## 📁 Repository Structure

```
📦 mental-health-social-media-analysis
 ┣ 📓 main_notebook.ipynb          ← Full executed Jupyter Notebook
 ┣ 📄 mental_health_social_media_dataset.csv  ← Dataset
 ┣ 🖼️ viz1_mental_state_distribution.png
 ┣ 🖼️ kmedoids_optimal_k.png
 ┣ 🖼️ kmedoids_cluster_analysis.png
 ┣ 🖼️ hierarchical_dendrograms.png
 ┣ 🖼️ fuzzy_membership_functions.png
 ┣ 🖼️ fuzzy_defuzzification.png
 ┣ 🖼️ pipeline_flowchart.png
 ┗ 📄 README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/mental-health-social-media-analysis.git
cd mental-health-social-media-analysis

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scikit-learn-extra skfuzzy scipy jupyter

# 3. Launch Jupyter
jupyter notebook main_notebook.ipynb
```

---

## 👥 Authors

> Developed as a Data Mining Course Project — Spring 2026

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute it.
