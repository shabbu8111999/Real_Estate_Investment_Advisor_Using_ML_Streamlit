import mlflow
import mlflow.sklearn


from .config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    TEST_SIZE,
    RANDOM_STATE,
    GROWTH_RATE,
    YEARS
)


def setup_mlflow():
    """Setting tracking URI and experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_common_params():
    """Log Basic params for this project."""
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("growth_rate", GROWTH_RATE)
    mlflow.log_param("years", YEARS)