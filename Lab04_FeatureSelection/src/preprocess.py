"""Reusable cleaning and feature engineering for the Ames Housing dataset."""

import pandas as pd


NONE_COLUMNS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "MasVnrType",
]

ZERO_COLUMNS = [
    "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
    "MasVnrArea",
]

QUALITY_COLUMNS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]

QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
OUTLIER_GR_LIV_AREA = 4000
OUTLIER_PRICE = 300000


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Ames data and add the features used by the model."""
    result = df.copy()

    result = result.loc[
        ~(
            (result["GrLivArea"] > OUTLIER_GR_LIV_AREA)
            & (result["SalePrice"] < OUTLIER_PRICE)
        )
    ].copy()
    result = result.drop(columns=["Id", "GarageYrBlt"], errors="ignore")

    result[NONE_COLUMNS] = result[NONE_COLUMNS].fillna("None")
    result[ZERO_COLUMNS] = result[ZERO_COLUMNS].fillna(0)

    result["TotalSF"] = (
        result["TotalBsmtSF"] + result["1stFlrSF"] + result["2ndFlrSF"]
    )
    result["HouseAge"] = (result["YrSold"] - result["YearBuilt"]).clip(lower=0)
    result["RemodAge"] = (result["YrSold"] - result["YearRemodAdd"]).clip(lower=0)
    result["TotalBath"] = (
        result["FullBath"]
        + 0.5 * result["HalfBath"]
        + result["BsmtFullBath"]
        + 0.5 * result["BsmtHalfBath"]
    )
    result["MSSubClass"] = result["MSSubClass"].astype(str)

    for column in QUALITY_COLUMNS:
        result[column] = result[column].map(QUALITY_MAP)

    return result