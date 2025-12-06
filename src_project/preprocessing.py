import pandas as pd
import numpy as np
from .config import GROWTH_RATE, YEARS

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric with median and categorical with mode"""
    df = df.copy()

    numeric_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # numeric to median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # categorical to mode
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def add_future_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add Future_Price_5Y column"""
    df = df.copy()
    growth_factor = (1 + GROWTH_RATE) ** YEARS

    if "Future_Price_5Y" not in df.columns:
        df['Future_Price_5Y'] = df['Price_in_Lakhs'] * growth_factor

    return df

def add_good_investment(df: pd.DataFrame) -> pd.DataFrame:
    """Adding Good_Investment label using city-wise median Price_per_SqFt"""
    df = df.copy()

    if "Good_Investment" not in df.columns:
        if "City" in df.columns:
            city_median = df.groupby("City")["Price_per_SqFt"].transform("median")
            df["Good_Investment"] = (df["Price_per_SqFt"] <= city_median).astype(int)
        else:
            overall_median = df["Price_per_SqFt"].median()
            df["Good_Investment"] = (
                df["Price_per_SqFt"] <= overall_median
            ).astype(int)

    return df

def make_features_targets(df: pd.DataFrame):
    """
    Splitting into:
    X -> Features
    y_class -> Good_Investment
    y_reg -> Future_Price_5Y
    """
    df = df.copy()

    y_class = df["Good_Investment"]
    y_reg = df["Future_Price_5Y"]

    drop_cols = ["Good_Investment", "Future_Price_5Y"]
    if "ID" in df.columns:
        drop_cols.append("ID")

    X = df.drop(columns=drop_cols)
    return X, y_class, y_reg