# Titanic ML Pipeline: Production-Ready Survival Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian%20Tuning-blueviolet.svg)](https://optuna.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-red.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline developed for the RMS Titanic dataset. Moving beyond standard exploratory scripts, this repository implements a **leakage-free, modular architecture** that standardizes preprocessing, dual-matrix routing (linear vs. tree models), out-of-fold cross-validation benchmarking, Optuna hyperparameter optimization, and TreeSHAP explainability.

The project packages trained estimators and frozen data pipelines into serializable `.joblib` production bundles capable of real-time single-record scoring and batch inference.

---

## Architecture Overview

```
Raw Data (df_train / df_test)
           │
           ▼
┌────────────────────────────────────────────────────────┐
│               TitanicDataPipeline                      │
│  - Regex title extraction & honorific mapping          │
│  - Hierarchical Age/Fare fallback imputation           │
│  - Ticket & family group size reconstruction           │
│  - One-hot encoding & schema consistency enforcement   │
└───────────────────────┬────────────────────────────────┘
                        │
            Bifurcated Feature Routing
           ┌────────────┴────────────┐
           ▼                         ▼
   X_linear (20 feats)       X_trees (25 feats)
   - Log-transformed fare    - Native FarePerPerson
   - Reference dummies dropped- Complete dummy representation
   - Collinearity-free       - Optimized split paths
           │                         │
           ▼                         ▼
   Logistic Regression         Random Forest
   (StandardScaler scaled)     (Unscaled)
           │                         │
           └────────────┬────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│               TitanicPredictor                         │
│  - Frozen pipeline state & schema reindexing           │
│  - Dual Serving: predict_single (API) / predict_batch  │
│  - Configurable decision thresholds                    │
│  - Atomic disk serialization (.joblib)                 │
└────────────────────────────────────────────────────────┘
```

---

## Key Engineering Decisions

* **Zero Data Leakage (Strict Fold Isolation)**: Imputation statistics (hierarchical age medians based on `Title` × `Pclass`, class-level fare medians, ticket frequencies, and boarding modes) are learned strictly on training partitions. When evaluated in cross-validation, a fresh pipeline is fitted per fold.
* **Feature Redundancy & Collinearity Elimination**:
  * Binary `Sex` was eliminated from target modeling matrices. Extracted honorifics (`Title`) fully preserve biological sex while simultaneously stratifying age and life stages (e.g., isolating young boys under `Master` with ~58% survival vs. adult `Mr` at ~16%).
  * For linear models, reference categories were dropped across all dummy variables (`Pclass_1`, `Title_Mr`, `GroupType_Alone`, `Embarked_S`, `Deck_U`) to prevent the dummy variable trap.
* **Ticket & Family Cohort Resolution**: Continuous `GroupSize` obscured survival dynamics. Passengers traveling in small groups (`SmallGroup`: 2–4 members) exhibited a **+0.31** positive correlation with survival, whereas solo travelers (`Alone`) had **-0.24**. Frequencies on unseen test batches merge frozen training frequencies with local batch occurrences to avoid shape mismatches.
* **Variance Stabilization**: Transformed skewed ticket prices into `LogFarePerPerson`, boosting feature-target linear correlation from **0.25** to **0.32**.

---

## Benchmark & Performance

Model performance was evaluated across a 5-fold stratified cross-validation setup, comparing naive unmanipulated baselines against the engineered modular pipeline:

| Model Architecture | Feature Matrix | Baseline 5-Fold CV Acc | Engineered 5-Fold CV Acc | Mean CV ROC-AUC | Optimal Hyperparameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | `X_trees` (25 feats) | 81.33% ($\pm$ 0.0233) | **84.06%** | **0.8692** | `n_est`: 200, `depth`: 10, `split`: 5, `leaf`: 2, `criterion`: 'entropy' |
| **Logistic Regression** | `X_linear` (20 feats) | 79.78% ($\pm$ 0.0190) | **83.39%** | **0.8717** | `C`: 0.0205, `solver`: 'lbfgs' |

> **Variance Diagnostics**: After 50 Bayesian optimization trials with Optuna, the Random Forest model achieved **90.57%** training accuracy versus **84.06%** cross-validation accuracy. Logistic Regression achieved **83.95%** train accuracy versus **83.39%** CV accuracy, demonstrating an ultra-stable, low-variance linear benchmark.

---

## Model Explainability (TreeSHAP)

Individual predictions were deconstructed using SHAP waterfall attributions against a background reference population ($E[f(X)] = 0.422$):

* **VIP Passenger Profile (`Madeleine Astor`)**: Final model probability: **83.8%**.
  * Primary positive drivers: `Title_Mr = 0` ($+0.13$), `Title_Mrs = 1` ($+0.10$), `Pclass_3 = 0` ($+0.09$), and high `FarePerPerson` ($+0.08$).
  * Primary negative drag: Ticket sharing within a larger traveling group (`GroupSize = 5`, $-0.04$).
* **Budget Passenger Profile (`Johan Svensson`)**: Final model probability: **11.5%**.
  * Primary negative penalties: Adult male placement (`Title_Mr = 1`, $-0.17$), lack of female demographic markers (`Title_Mrs = 0`, $-0.05$), 3rd class assignment (`Pclass_3 = 1`, $-0.02$), and lack of assigned cabin (`HasCabin = 0`, $-0.02$).

---

## Repository Structure

```text
├── notebooks/
│   ├── 01_Data_analysis.ipynb    # Exploratory analysis, skewness audits & initial baseline
│   └── 02_Pipeline.ipynb    # Pipeline design, OOF benchmarking, Optuna tuning & SHAP
├── models/
│   ├── titanic_model_prod.joblib # Serialized Random Forest predictor bundle
│   └── titanic_lr_prod.joblib    # Serialized Logistic Regression predictor bundle
├── data/
│   ├── Titanic-Dataset.csv       # Raw training data
│   └── test.csv                  # Raw evaluation data
├── submission.csv                # Reproducible Kaggle submission artifact
├── requirements.txt              # Environment dependencies
└── README.md
```

---

## Quickstart & Operational Inference

The bundled `.joblib` artifacts encapsulate the trained estimator, the fitted scaler, and the frozen feature engineering pipeline. Predictions can be served directly from raw dictionaries without manual data preprocessing:

```python
import pandas as pd
from notebooks.titanic_train import TitanicPredictor

# 1. Load production bundle
prod_api = TitanicPredictor.load('models/titanic_model_prod.joblib')

# 2. Score a single unparsed payload (e.g., incoming REST request)
raw_payload = {
    'PassengerId': 1001,
    'Pclass': 1,
    'Name': 'Astor, Mrs. John Jacob (Madeleine Talmage Force)',
    'Sex': 'female',
    'Age': 18.0,
    'SibSp': 1,
    'Parch': 0,
    'Ticket': 'PC 17757',
    'Fare': 227.525,
    'Cabin': 'C62',
    'Embarked': 'C',
}

response = prod_api.predict_single(raw_payload)
print(response)
# Output:
# {
#   'survived': 1,
#   'survival_probability': 0.8376,
#   'decision_threshold': 0.5,
#   'status': 'High Survival Chance'
# }

# 3. Vectorized batch scoring on raw DataFrames
df_test = pd.read_csv('data/test.csv')
scored_df = prod_api.predict_batch(df_test)
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/titanic-ml-pipeline.git
   cd titanic-ml-pipeline
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE.txt) file for details.
