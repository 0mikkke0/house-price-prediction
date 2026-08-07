# House Price Prediction System

An end-to-end regression machine learning project that predicts residential property sale prices.

## What it demonstrates
- Regression fundamentals
- Exploratory data analysis
- Missing-value handling
- Numerical and categorical preprocessing
- One-hot encoding
- Feature engineering
- Model comparison
- MAE, RMSE and R² evaluation
- Reproducible scikit-learn pipelines
- Streamlit prediction interface

## Dataset
This project uses the Ames Housing dataset available through OpenML (dataset ID 42165).

The project downloads the data automatically, so the large raw dataset does not need to be committed to GitHub.

## Structure

```text
house-price-prediction/
├── data/
├── models/
├── notebooks/
│   └── house_price_analysis.ipynb
├── src/
│   ├── download_data.py
│   ├── train.py
│   └── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Recommended: Python 3.11 or 3.12.

```bash
python -m venv .venv
```

Windows CMD:

```bat
.venv\Scripts\activate.bat
pip install -r requirements.txt
python src/download_data.py
python src/train.py
streamlit run app.py
```

## Modeling approach

The training pipeline:
1. loads the Ames Housing dataset,
2. separates the sale-price target,
3. creates a train/test split,
4. handles numerical missing values with median imputation,
5. handles categorical missing values with most-frequent imputation,
6. one-hot encodes categorical features,
7. compares Linear Regression and Random Forest Regression,
8. evaluates MAE, RMSE and R²,
9. saves the best model pipeline.

## Why this project matters

Projects 1 and 2 focused on classification. This project adds **supervised regression**, giving the portfolio a different ML problem type.

## Interview topics

Be ready to explain:
- classification vs regression,
- MAE vs RMSE,
- R²,
- why RMSE penalizes large errors more heavily,
- one-hot encoding,
- missing-value strategies,
- feature engineering,
- overfitting,
- train/test splitting,
- cross-validation,
- data leakage,
- why a production house-price model would require current local-market data.

## Disclaimer

This is a portfolio/learning project using a public housing benchmark. It should not be treated as a professional property valuation system.
