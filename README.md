# House Price Prediction System

An end-to-end regression machine learning project that predicts residential property sale prices.

## What It Demonstrates

* Regression fundamentals
* Exploratory data analysis
* Missing-value handling
* Numerical and categorical preprocessing
* One-hot encoding
* Feature engineering
* Model comparison
* MAE, RMSE and R² evaluation
* Reproducible scikit-learn pipelines
* Streamlit prediction interface

## Dataset

This project uses the Ames Housing dataset available through OpenML (dataset ID 42165).

The dataset is downloaded automatically, so the large raw dataset does not need to be committed to GitHub.

## Project Structure

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

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate.bat
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset:

```bash
python src/download_data.py
```

Train the models:

```bash
python src/train.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

## Modeling Approach

The training pipeline:

1. Loads the Ames Housing dataset.
2. Separates the sale-price target from the input features.
3. Creates a train/test split.
4. Handles numerical missing values using median imputation.
5. Handles categorical missing values using most-frequent imputation.
6. One-hot encodes categorical features.
7. Compares Linear Regression and Random Forest Regression.
8. Evaluates models using MAE, RMSE and R².
9. Saves the best-performing model pipeline.

The preprocessing and model are combined into a reproducible scikit-learn pipeline to ensure that the same transformations are applied during both training and prediction.

## Evaluation Metrics

The project uses three standard regression metrics:

* **MAE (Mean Absolute Error):** Measures the average absolute difference between predicted and actual prices.
* **RMSE (Root Mean Squared Error):** Measures prediction error while placing greater emphasis on larger errors.
* **R² (Coefficient of Determination):** Measures how much of the variation in the target variable is explained by the model.

## Results

Results will be added after training and evaluation.

Example:

```text
Model: Best-performing model
MAE: XX.XX
RMSE: XX.XX
R²: XX.XX
```

The final values will be based on the actual test-set performance of the trained models.

## Key ML Concepts

This project demonstrates practical understanding of:

* Supervised regression
* Train/test splitting
* Numerical and categorical preprocessing
* Missing-value imputation
* One-hot encoding
* Feature engineering
* Model comparison
* MAE, RMSE and R²
* Overfitting and generalization
* Cross-validation
* Data leakage
* Reproducible machine learning pipelines

## Why This Project Matters

The first two projects in this portfolio focus on classification problems:

* Customer Churn Prediction — classification
* Credit Card Fraud Detection — imbalanced classification

This project introduces **supervised regression**, demonstrating the ability to work with a different type of machine learning problem and different evaluation metrics.

## Future Improvements

Possible improvements include:

* Hyperparameter tuning
* Cross-validation-based model selection
* Gradient boosting models such as XGBoost or LightGBM
* Feature importance and model explainability
* Log transformation of the target variable
* Improved feature engineering based on domain knowledge
* Integration with current local housing-market data
* Production monitoring and model retraining

## Disclaimer

This is a portfolio/learning project using a public housing benchmark dataset. It is not intended to represent a professional property valuation system or provide real-world property pricing advice.
