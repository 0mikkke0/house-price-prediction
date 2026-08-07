from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ames_housing.csv"
MODEL = ROOT / "models" / "house_price_model.joblib"
METRICS = ROOT / "models" / "metrics.json"

def load_data():
    if not DATA.exists():
        raise FileNotFoundError("Run `python src/download_data.py` first.")

    df = pd.read_csv(DATA)

    # OpenML versions can expose the target as SalePrice.
    if "SalePrice" not in df.columns:
        raise ValueError("Expected SalePrice target was not found.")

    df["SalePrice"] = pd.to_numeric(df["SalePrice"], errors="coerce")
    df = df.dropna(subset=["SalePrice"])
    return df

def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "mae": round(mean_absolute_error(y_test, pred), 2),
        "rmse": round(mean_squared_error(y_test, pred) ** 0.5, 2),
        "r2": round(r2_score(y_test, pred), 4),
    }

df = load_data()

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    ),
}

results = {}
fitted = {}

for name, estimator in models.items():
    print(f"Training {name}...")

    pipe = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", estimator),
    ])

    pipe.fit(X_train, y_train)
    results[name] = evaluate(pipe, X_test, y_test)
    fitted[name] = pipe

    print(name, results[name])

# Higher R² is better; use it as the simple model-selection criterion.
best_name = max(results, key=lambda name: results[name]["r2"])

MODEL.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(fitted[best_name], MODEL)

METRICS.write_text(json.dumps({
    "best_model": best_name,
    "models": results
}, indent=2))

print(f"\nBest model: {best_name}")
print(f"Saved model: {MODEL}")
print(f"Saved metrics: {METRICS}")
