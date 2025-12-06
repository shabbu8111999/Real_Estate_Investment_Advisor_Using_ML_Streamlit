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

import mlflow
import mlflow.sklearn
from .mlflow_utils import setup_mlflow, log_common_params


def run_training():
    """Full training flow with MLflow Logging."""

    setup_mlflow()

    with mlflow.start_run(run_name="rf_lgbm_baseline"):

        # Logging project-levels params
        log_common_params()
        mlflow.log_param("classifier -", "RandomForestClassifier")
        mlflow.log_param("regressor -", "LGBMRegressor")

        # Loading data
        df = load_raw_data()

        # Preprocess
        df = fill_missing_values(df)
        df = add_future_price(df)
        df = add_good_investment(df)

        # Features and targets
        X, y_class, y_reg = make_features_targets(df)

        # Split
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = split_data(
            X, y_class, y_reg
        )

        # Preprocessor and models
        preprocessor = build_preprocessor(X_train)
        clf_pipeline, reg_pipeline = build_models(preprocessor)

        # Train models
        clf_pipeline.fit(X_train, y_class_train)
        reg_pipeline.fit(X_train, y_reg_train)

        # Evaluate
        y_class_pred = clf_pipeline.predict(X_test)
        y_reg_pred = reg_pipeline.predict(X_test)

        clf_metrics = eval_classifier(y_class_test, y_class_pred)
        reg_metrics = eval_regressor(y_reg_test, y_reg_pred)

        print("Classification metrics:", clf_metrics)
        print("Regression metrics:", reg_metrics)

        # Logging metrics to mlflow
        for k, v in clf_metrics.items():
            # example names: class_accuracy, class_f1, class_precision, etc.
            mlflow.log_metric(f"class_{k}", float(v))

        for k, v in reg_metrics.items():
            # example names: reg_mse, reg_rmse, reg_r2, etc.
            mlflow.log_metric(f"reg_{k}", float(v))

        # Save models
        joblib.dump(clf_pipeline, ARTIFACTS_DIR / "class_model.pkl")
        joblib.dump(reg_pipeline, ARTIFACTS_DIR / "reg_model.pkl")

        # Save splits (for MLflow or further analysis)
        joblib.dump(X_train, ARTIFACTS_DIR / "X_train.pkl")
        joblib.dump(X_test, ARTIFACTS_DIR / "X_test.pkl")
        joblib.dump(y_class_train, ARTIFACTS_DIR / "y_class_train.pkl")
        joblib.dump(y_class_test, ARTIFACTS_DIR / "y_class_test.pkl")
        joblib.dump(y_reg_train, ARTIFACTS_DIR / "y_reg_train.pkl")
        joblib.dump(y_reg_test, ARTIFACTS_DIR / "y_reg_test.pkl")

        # Logging models to mlflow
        mlflow.sklearn.log_model(clf_pipeline, "classifier_model")
        mlflow.sklearn.log_model(reg_pipeline, "regressor_model")

        # Logging the artifacts
        mlflow.log_artifact(ARTIFACTS_DIR / "class_model.pkl", artifact_path="artifacts")
        mlflow.log_artifact(ARTIFACTS_DIR / "reg_model.pkl", artifact_path="artifacts")

        return clf_metrics, reg_metrics
