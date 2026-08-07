from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "house_price_model.joblib"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Prediction")
st.caption("Regression demo using the Ames Housing dataset.")

if not MODEL.exists():
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()

model = joblib.load(MODEL)

st.write("Enter a small set of common property attributes.")

overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 6)
gr_liv_area = st.number_input("Above-ground living area (sq ft)", 300, 5000, 1500)
garage_cars = st.number_input("Garage capacity (cars)", 0, 5, 2)
total_bsmt_sf = st.number_input("Basement area (sq ft)", 0, 4000, 800)
year_built = st.number_input("Year Built", 1870, 2025, 2000)
full_bath = st.number_input("Full Bathrooms", 0, 5, 2)
bedroom_abv_gr = st.number_input("Bedrooms above ground", 0, 10, 3)

st.info(
    "The trained model uses the full dataset feature schema. "
    "This demo maps the visible inputs to that schema and fills other fields with defaults."
)

if st.button("Predict Price", type="primary"):
    # Build the complete Ames schema from the training data.
    # Defaults make this demo usable without asking for dozens of fields.
    row = {
        "MSSubClass": 60,
        "MSZoning": "RL",
        "LotFrontage": 70.0,
        "LotArea": 10000,
        "Street": "Pave",
        "Alley": "NA",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": "NAmes",
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "OverallQual": overall_qual,
        "OverallCond": 5,
        "YearBuilt": year_built,
        "YearRemodAdd": year_built,
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": "None",
        "MasVnrArea": 0,
        "ExterQual": "TA",
        "ExterCond": "TA",
        "Foundation": "PConc",
        "BsmtQual": "TA",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "Unf",
        "BsmtFinSF1": 0,
        "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0,
        "BsmtUnfSF": total_bsmt_sf,
        "TotalBsmtSF": total_bsmt_sf,
        "Heating": "GasA",
        "HeatingQC": "TA",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "1stFlrSF": gr_liv_area,
        "2ndFlrSF": 0,
        "LowQualFinSF": 0,
        "GrLivArea": gr_liv_area,
        "BsmtFullBath": 0,
        "BsmtHalfBath": 0,
        "FullBath": full_bath,
        "HalfBath": 0,
        "BedroomAbvGr": bedroom_abv_gr,
        "KitchenAbvGr": 1,
        "KitchenQual": "TA",
        "TotRmsAbvGrd": 6,
        "Functional": "Typ",
        "Fireplaces": 0,
        "FireplaceQu": "NA",
        "GarageType": "Attchd",
        "GarageYrBlt": year_built,
        "GarageFinish": "Unf",
        "GarageCars": garage_cars,
        "GarageArea": garage_cars * 250,
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PavedDrive": "Y",
        "WoodDeckSF": 0,
        "OpenPorchSF": 50,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "PoolQC": "NA",
        "Fence": "NA",
        "MiscFeature": "NA",
        "MiscVal": 0,
        "MoSold": 6,
        "YrSold": 2010,
        "SaleType": "WD",
    }

    prediction = float(model.predict(pd.DataFrame([row]))[0])
    st.success(f"Estimated sale price: ${prediction:,.0f}")
