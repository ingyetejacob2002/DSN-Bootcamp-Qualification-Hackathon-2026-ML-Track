# DSN Bootcamp Qualification Hackathon 2026 – ML Track

**Predict total product-store sales (Regression – RMSE)**

---

## Competition Overview

DSN Mart operates a chain of stores across Nigeria. The task is to build a machine learning model that predicts **total_sales** for a given product at a given store, based on product attributes and store characteristics.

- **Metric**: Root Mean Squared Error (RMSE) — lower is better  
- **Train**: 6,818 rows  
- **Test**: 1,705 rows  

---

## Best Result

| Metric                    | Value          |
|---------------------------|----------------|
| Best Model                | RidgeStack     |
| 5-Fold CV RMSE            | **1075.74**    |
| Baseline (predict mean)   | ~1698          |

> **Note**: Achieving RMSE < 900 is extremely ambitious given the target standard deviation (~1698) and the limited number of stores (only 10). The solution prioritizes leakage-free, well-regularized modeling.

---

## Solution Pipeline

1. **Exploratory Data Analysis (EDA)**
   - Target distribution (raw + log)
   - Missing value analysis
   - Store-level and category-level sales patterns
   - Price vs Sales relationship
   - Correlation analysis

2. **Data Cleaning & Preprocessing**
   - Standardized product category casing
   - Imputed `product_weight_kg` (product-level median → global median)
   - Imputed `store_size` (store-level mode → "Unknown")

3. **Feature Engineering**
   - Numeric transforms (`log_price`, `price_per_kg`, etc.)
   - Rich interactions (`price × age`, `visibility × price`, …)
   - Binary flags for store format, tier, size, fat content
   - Frequency encodings
   - Binned features
   - **Safe Out-of-Fold Target Encoding** for:
     - `product_code`
     - `store_code`
     - `product_category`
     - `category × store_format`
     - `category × location_tier`
   - Product-level aggregates (mean price & visibility)

4. **Modeling**
   - Target transformed with `log1p` (inverted with `expm1` after prediction)
   - 5-Fold Cross-Validation
   - Models trained:
     - CatBoost
     - LightGBM
     - XGBoost
     - HistGradientBoosting
     - RandomForest
     - ExtraTrees
   - Strong regularization + early stopping to control overfitting

5. **Ensembling**
   - Weighted blend of top models
   - **Ridge Stacking** (meta-learner) → best performance

---

## Model Comparison (CV RMSE)

| Rank | Model              | CV RMSE   |
|------|--------------------|-----------|
| 1    | **RidgeStack**     | **1075.74** |
| 2    | WeightedBlend      | 1108.96   |
| 3    | ExtraTrees         | 1112.74   |
| 4    | CatBoost           | 1113.44   |
| 5    | RandomForest       | 1114.95   |
| 6    | LightGBM           | 1118.89   |
| 7    | HistGradientBoosting | 1120.19 |
| 8    | XGBoost            | 1124.97   |

---

## Project Structure

```
artifacts/
├── DSN_Bootcamp_ML_Solution.ipynb   # Full reproducible notebook
├── submission.csv                   # Final predictions (best model)
├── model_comparison.csv             # All model CV scores
├── README.md                        # This file
└── plots/                           # EDA and residual plots
    ├── 01_target_distribution.png
    ├── 02_missing_values.png
    ├── 03_store_avg_sales.png
    ├── 04_category_sales.png
    ├── 05_price_format.png
    └── 07_residuals.png
```

---

## How to Run

### Requirements
```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost seaborn matplotlib
```

### Execute the Notebook
Open `DSN_Bootcamp_ML_Solution.ipynb` in Jupyter / VS Code / Colab / Kaggle and run all cells.

The notebook is self-contained and will:
- Perform full EDA with visualizations
- Engineer features
- Train all models
- Generate the final `submission.csv`

---

## Key Design Decisions (Anti-Overfitting)

- **Strict Out-of-Fold target encoding** — no leakage
- Log-transform of the target to handle skewness
- Early stopping on validation folds
- Regularization (L2, depth limits, min_child_samples, subsample/colsample)
- Diverse model pool for robust stacking

---

## Files for Submission

- **Primary submission file**: `submission.csv`  
  Format: `id,total_sales`

---

## Author Notes

This solution follows the full competition requirements:
- Dataset exploration & understanding
- Factor analysis
- Cleaning & preprocessing
- EDA with visualizations
- Feature engineering
- Multiple model training & evaluation
- Final prediction generation
- Clear communication of results

Good luck with the DSN AI Bootcamp selection!
