from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Data and Artifacts Path
DATA_PATH = BASE_DIR/"data"/"india_housing_prices.csv"
ARTIFACTS_DIR = BASE_DIR/"artifacts"

# Create artifacts folder if not present
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Train and Test Split setting
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Target creation settings
GROWTH_RATE = 0.08  # 8% yearly
YEARS = 5