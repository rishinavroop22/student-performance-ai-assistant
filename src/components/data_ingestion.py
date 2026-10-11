import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ARTIFACTS_DIR, DATA_PATH, RANDOM_STATE
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(ARTIFACTS_DIR, "train.csv")
    test_data_path: str = os.path.join(ARTIFACTS_DIR, "test.csv")
    raw_data_path: str = os.path.join(ARTIFACTS_DIR, "data.csv")


class DataIngestion:
    def __init__(self, source_path: str = DATA_PATH):
        self.ingestion_config = DataIngestionConfig()
        self.source_path = source_path

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            df = pd.read_csv(self.source_path)
            before = len(df)
            df = df.drop_duplicates()
            logging.info(f"Read {before} rows, {before - len(df)} duplicates dropped")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)
