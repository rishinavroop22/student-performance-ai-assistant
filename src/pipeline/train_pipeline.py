"""End-to-end training: ingestion -> feature engineering/preprocessing -> model selection.

Run from the project root:
    python -m src.pipeline.train_pipeline
"""
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException


class TrainPipeline:
    def run(self):
        try:
            train_path, test_path = DataIngestion().initiate_data_ingestion()
            train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(train_path, test_path)
            return ModelTrainer().initiate_model_trainer(train_arr, test_arr)
        except Exception as e:
            raise CustomException(e, sys)


def main():
    metrics = TrainPipeline().run()
    print(f"\nBest model: {metrics['best_model']} (selected by {metrics['selection']})")
    print(f"Test set:   R2={metrics['test_metrics']['r2']}  MAE={metrics['test_metrics']['mae']}  "
          f"RMSE={metrics['test_metrics']['rmse']}\n")
    print(f"{'Model':32} {'CV R2':>8} {'± std':>8}")
    for row in metrics["leaderboard"]:
        print(f"{row['model']:32} {row['cv_r2_mean']:>8.4f} {row['cv_r2_std']:>8.4f}")


if __name__ == "__main__":
    main()
