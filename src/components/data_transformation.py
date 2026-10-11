import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLUMNS,
    ENGINEERED_COLUMNS,
    NUMERICAL_COLUMNS,
    PREPROCESSOR_PATH,
    SCHEMA_PATH,
    TARGET_COLUMN,
)
from src.exception import CustomException
from src.logger import logging
from src.utils import save_json, save_object


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: combined literacy score and reading/writing gap."""
    df = df.copy()
    df["avg_literacy_score"] = (df["reading_score"] + df["writing_score"]) / 2
    df["reading_writing_gap"] = df["reading_score"] - df["writing_score"]
    return df


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = PREPROCESSOR_PATH
    schema_file_path: str = SCHEMA_PATH


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> Pipeline:
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            columns = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, NUMERICAL_COLUMNS + ENGINEERED_COLUMNS),
                    ("cat_pipeline", cat_pipeline, CATEGORICAL_COLUMNS),
                ]
            )
            return Pipeline(
                steps=[
                    ("feature_engineering", FunctionTransformer(add_engineered_features)),
                    ("columns", columns),
                ]
            )
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data")

            preprocessing_obj = self.get_data_transformer_object()
            feature_columns = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS

            X_train = train_df[feature_columns]
            y_train = train_df[TARGET_COLUMN].to_numpy()
            X_test = test_df[feature_columns]
            y_test = test_df[TARGET_COLUMN].to_numpy()

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, y_train]
            test_arr = np.c_[X_test_arr, y_test]

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)

            # Schema used by the API for input validation and by the assistant for context.
            full = pd.concat([train_df, test_df])
            schema = {
                "categories": {c: sorted(full[c].dropna().unique().tolist()) for c in CATEGORICAL_COLUMNS},
                "numeric_range": {c: [0, 100] for c in NUMERICAL_COLUMNS},
                "target_mean": float(train_df[TARGET_COLUMN].mean()),
                "target_std": float(train_df[TARGET_COLUMN].std()),
            }
            save_json(self.data_transformation_config.schema_file_path, schema)
            logging.info("Saved preprocessing object and schema")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)
