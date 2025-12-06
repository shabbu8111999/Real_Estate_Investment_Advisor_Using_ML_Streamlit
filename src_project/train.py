import joblib
from .config import ARTIFACTS_DIR
from .data_loader import load_raw_data
from .preprocessing import (
    fill_missing_values,
    add_future_price,
    add_good_investment,
    make_features_targets,
)
from .modeling import split_data, build_preprocessor, build_models
from .evaluate import eval_classifier, eval_regressor


def run_training():
    """Full training flow: load → preprocess → train → evaluate → save."""

    # 1. Load data
    df = load_raw_data()

    # 2. Preprocess
    df = fill_missing_values(df)
    df = add_future_price(df)
    df = add_good_investment(df)

    # 3. Features and targets
    X, y_class, y_reg = make_features_targets(df)

    # 4. Split
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = split_data(
        X, y_class, y_reg
    )

    # 5. Preprocessor and models
    preprocessor = build_preprocessor(X_train)
    clf_pipeline, reg_pipeline = build_models(preprocessor)

    # 6. Train models
    clf_pipeline.fit(X_train, y_class_train)
    reg_pipeline.fit(X_train, y_reg_train)

    # 7. Evaluate
    y_class_pred = clf_pipeline.predict(X_test)
    y_reg_pred = reg_pipeline.predict(X_test)

    clf_metrics = eval_classifier(y_class_test, y_class_pred)
    reg_metrics = eval_regressor(y_reg_test, y_reg_pred)

    print("Classification metrics:", clf_metrics)
    print("Regression metrics:", reg_metrics)

    # 8. Save models
    joblib.dump(clf_pipeline, ARTIFACTS_DIR / "class_model.pkl")
    joblib.dump(reg_pipeline, ARTIFACTS_DIR / "reg_model.pkl")

    # 9. Save splits (for MLflow or further analysis)
    joblib.dump(X_train, ARTIFACTS_DIR / "X_train.pkl")
    joblib.dump(X_test, ARTIFACTS_DIR / "X_test.pkl")
    joblib.dump(y_class_train, ARTIFACTS_DIR / "y_class_train.pkl")
    joblib.dump(y_class_test, ARTIFACTS_DIR / "y_class_test.pkl")
    joblib.dump(y_reg_train, ARTIFACTS_DIR / "y_reg_train.pkl")
    joblib.dump(y_reg_test, ARTIFACTS_DIR / "y_reg_test.pkl")

    return clf_metrics, reg_metrics
