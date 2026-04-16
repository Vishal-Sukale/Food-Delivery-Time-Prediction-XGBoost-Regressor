# Food Delivery Time Prediction Using XGBoost Regressor with Hyperparameter Tuning

## Problem Statement
Food delivery platforms face the challenge of accurately predicting
delivery time to improve customer satisfaction and operational efficiency.
Delivery time is influenced by multiple real-world factors such as
distance, traffic level, weather conditions, preparation time, vehicle
type and partner ratings.

This project uses XGBoost Regressor (Extreme Gradient Boosting) — the
most powerful and widely-used ML algorithm in industry and data science
competitions — to predict food delivery time (in minutes). XGBoost's
gradient boosting optimization, built-in regularization and speed make
it superior to traditional regression models for this real-world task.

## Objective
- Predict food delivery time (in minutes) using XGBoost Regressor
- Perform Feature Engineering to extract 4 new meaningful features
- Apply One-Hot Encoding on categorical features
- Follow correct ML pipeline — Encoding → Split → Model
- Improve model using GridSearchCV Hyperparameter Tuning
- Evaluate using MAE, MSE, RMSE and R² Score
- Compare Before and After tuning performance

## Dataset
| Detail | Info |
|---|---|
| Name | Food Delivery Dataset |
| Records | 1,000 |
| Features | 9 |
| Target | Delivery_Time_min |
| Missing Values | None  |

### Features
| Feature | Type | Description |
|---|---|---|
| Distance_km | Numerical | Delivery distance in km |
| Order_Time | Numerical | Hour of order placement |
| Traffic_Level | Categorical | Low / Medium / High |
| Weather | Categorical | Clear / Cloudy / Rainy |
| Restaurant_Rating | Numerical | Rating of restaurant |
| Prep_Time_min | Numerical | Food preparation time |
| Delivery_Partner_Rating | Numerical | Rating of delivery partner |
| Vehicle_Type | Categorical | Bike / Scooter / Bicycle |

## Tech Stack
| Tool | Usage |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Operations |
| Scikit-learn | ML Evaluation & GridSearch |
| XGBoost | ML Model |
| Jupyter Notebook | Development Environment |

## ML Pipeline
```
1. Import Libraries
2. Load Dataset
3. Data Understanding
4. Data Preprocessing
   - Drop Order_ID column
   - Feature Engineering (4 new features)
   - One-Hot Encoding (get_dummies, drop_first=True, dtype=int)
5. Model Building
   - Split Features & Target
   - Train-Test Split (80/20)
   - Model Before Tuning (default XGBoost)
   - GridSearchCV Hyperparameter Tuning
   - Model After Tuning (best params)
6. Evaluation & Prediction Report
```

## Feature Engineering
| New Feature | Formula | Purpose |
|---|---|---|
| Distance_x_Traffic | Distance × Traffic(1/2/3) | Combined impact of distance & traffic |
| Rating_Diff | Restaurant - Partner Rating | Quality gap between restaurant & partner |
| Total_Time_Est | Distance×5 + Prep Time | Rough estimated delivery time |
| Rush_Hour | 1 if 6PM-10PM else 0 | Peak hour flag |

## Results

| Metric | Before Tuning | After Tuning |
|---|---|---|
| **R² Score** | 93.89% | **95.39%**  |
| **MAE** | 3.770 | **3.482** |
| **MSE** | 22.569 | **17.052**  |
| **RMSE** | 4.751 | **4.129** |

### Best Parameters Found
| Parameter | Value |
|---|---|
| colsample_bytree | 0.8 |
| learning_rate | 0.1 |
| max_depth | 3 |
| n_estimators | 200 |
| subsample | 0.8 |

## Key Insights
- R² improved from **93.89% → 95.39%** after tuning 
- All 4 metrics improved significantly after tuning 
- MAE reduced from **3.770 → 3.482** 
- MSE reduced from **22.569 → 17.052** (significant!) 
- RMSE reduced from **4.751 → 4.129** 
- R² improvement of **+1.50%** 
- colsample_bytree=0.8 → reduces overfitting 
- subsample=0.8 → better generalization 
- Average prediction error = only **3.48 minutes** — production ready! 
- XGBoost best performing regression model in portfolio:
  - Random Forest R²: 94.08%
  - AdaBoost R²: 94.64%
  - **XGBoost R²: 95.39%** ← Best! 

## Regression Model Comparison

| Model | R² Score |
|---|---|
| Random Forest | 94.08% |
| AdaBoost | 94.64% |
| **XGBoost** | **95.39%** |

## Project Structure
```
Food-Delivery-Time-Prediction-XGBoost-Regressor/
│
├── Xgboost_Regressor_Project.ipynb     # Main Jupyter Notebook
├── food_delivery_dataset.csv           # Dataset
└── README.md                           # Project Documentation
```

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/yourusername/Food-Delivery-Time-Prediction-XGBoost-Regressor.git
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost
```

3. Open Jupyter Notebook
```bash
jupyter notebook Xgboost_Regressor_Project.ipynb
```

## Conclusion
The XGBoost Regressor successfully predicted food delivery time with
an outstanding R² Score of **95.39%** after hyperparameter tuning —
a significant improvement from the **93.89%** baseline (+1.50%).

Feature Engineering added powerful interaction features like
Distance_x_Traffic and Total_Time_Est that helped XGBoost capture
real-world delivery patterns. GridSearchCV found the optimal combination
of n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.8
and colsample_bytree=0.8.

With an average prediction error of only **3.48 minutes**, this model
is production-ready for real-world food delivery platforms. XGBoost
outperformed all previous regression models — confirming its position
as the go-to algorithm for tabular data prediction tasks.

## Author
**Vishal Sukale**


