import os
import sys
from functools import lru_cache

import pandas as pd

from src.config import CATEGORICAL_COLUMNS, MODEL_PATH, NUMERICAL_COLUMNS, PREPROCESSOR_PATH, SCHEMA_PATH
from src.exception import CustomException
from src.utils import load_json, load_object

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS


class ModelNotTrainedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_artifacts():
    """Load model + preprocessor once per process instead of on every request."""
    for path in (MODEL_PATH, PREPROCESSOR_PATH, SCHEMA_PATH):
        if not os.path.exists(path):
            raise ModelNotTrainedError(
                "Model artifacts not found. Run `python -m src.pipeline.train_pipeline` first."
            )
    return load_object(MODEL_PATH), load_object(PREPROCESSOR_PATH), load_json(SCHEMA_PATH)


def clear_cache():
    _load_artifacts.cache_clear()


def get_schema():
    return _load_artifacts()[2]


class PredictPipeline:
    def __init__(self):
        self.model, self.preprocessor, self.schema = _load_artifacts()

    def predict(self, features: pd.DataFrame):
        try:
            transformed = self.preprocessor.transform(features[FEATURE_COLUMNS])
            preds = self.model.predict(transformed)
            return [float(min(100.0, max(0.0, p))) for p in preds]
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def to_dict(self):
        return {
            "gender": self.gender,
            "race_ethnicity": self.race_ethnicity,
            "parental_level_of_education": self.parental_level_of_education,
            "lunch": self.lunch,
            "test_preparation_course": self.test_preparation_course,
            "reading_score": float(self.reading_score),
            "writing_score": float(self.writing_score),
        }

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame([self.to_dict()])
        except Exception as e:
            raise CustomException(e, sys)
