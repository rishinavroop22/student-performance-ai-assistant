"""Shared paths and schema for the project."""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(ROOT_DIR, "artifacts"))
DATA_PATH = os.path.join(ROOT_DIR, "src", "notebook", "data", "stud.csv")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
SCHEMA_PATH = os.path.join(ARTIFACTS_DIR, "schema.json")

TARGET_COLUMN = "math_score"
NUMERICAL_COLUMNS = ["reading_score", "writing_score"]
CATEGORICAL_COLUMNS = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]
ENGINEERED_COLUMNS = ["avg_literacy_score", "reading_writing_gap"]
RANDOM_STATE = 42
