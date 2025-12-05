# For classification Model (Good Investment)

import mlflow
import mlflow.sklearn
import joblib
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


# Loading data splits and models from artifacts

X_train = joblib.load("project_jupyter_file/artifacts/X_train.pkl")
X_test = joblib.load("project_jupyter_file/artifacts/X_test.pkl")

y_class_train = joblib.load("project_jupyter_file/artifacts/y_class_train.pkl")
y_class_test = joblib.load("project_jupyter_file/artifacts/y_class_test.pkl")

y_reg_train = joblib.load("project_jupyter_file/artifacts/y_reg_train.pkl")
y_reg_test = joblib.load("project_jupyter_file/artifacts/y_reg_test.pkl")

# Loading Trained Pipelines
clf_pipeline = joblib.load("project_jupyter_file/artifacts/class_model.pkl")
reg_pipeline = joblib.load("project_jupyter_file/artifacts/reg_model.pkl")


# Starting the MLflow run for Classification Model (Good_Invetment)
with mlflow.start_run(run_name="Good_Invetsment_Classification"):

    y_class_pred = clf_pipeline.predict(X_test)

    acc = accuracy_score(y_class_test, y_class_pred)
    f1 = f1_score(y_class_test, y_class_pred)

    # Logging the Parameters
    mlflow.log_param("task", "classification")
    mlflow.log_param("model_type", "Classification_Pipeline")

    # Logging Metrics
    mlflow.log_metric("Accuracy -", acc)
    mlflow.log_metric("F1 Score -", f1)

    # Logging the Model to ML Flow
    mlflow.sklearn.log_model(clf_pipeline, name="Classification_Model")

    print("Classification - Accuracy:", acc)
    print("Classification - F1 Score:", f1)


# Starting the MLflow run for Regression Model (Future_Price_5Y)
with mlflow.start_run(run_name="Future_Price_Regression"):

    y_reg_pred = reg_pipeline.predict(X_test)

    mse = mean_squared_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_reg_test, y_reg_pred)

    # Logging the Parameters
    mlflow.log_param("task", "regression")
    mlflow.log_param("model_type", "Regression_Pipeline")

    # Logging Metrics
    mlflow.log_metric("Mean Squared Error -", mse)
    mlflow.log_metric("Root Mean Squared Error -", rmse)
    mlflow.log_metric("R2 Score -", r2)

    # Logging the Model to ML Flow
    mlflow.sklearn.log_model(reg_pipeline, name="Regression_Model")

    print("Regression - MSE:", mse)
    print("Regression - RMSE:", rmse)
    print("Regression - R2 Score:", r2)