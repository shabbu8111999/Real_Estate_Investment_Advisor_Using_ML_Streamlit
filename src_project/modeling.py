from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor

from .config import TEST_SIZE, RANDOM_STATE


def split_data(
        X: pd.DataFrame, y_class: pd.Series, y_reg: pd.Series
) -> Tuple:
    
    """Train and Test Split for both Classification and Regression targets."""
    return train_test_split(
        X,
        y_class,
        y_reg,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Creating Numeric and Categorical Preprocessing Transformer"""

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.to_list()
    categorical_features = X.select_dtypes(include=["object"]).columns.to_list()

    # numeric: median and scale
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # categorical: most frequent and one-hot
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ("encoder", OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    # 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor

def build_models(preprocessor: ColumnTransformer):
    """Build classifier and regressor pipelines."""

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
    )

    reg = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        random_state=RANDOM_STATE,
    )

    clf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )

    reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", reg),
        ]
    )

    return clf_pipeline, reg_pipeline

