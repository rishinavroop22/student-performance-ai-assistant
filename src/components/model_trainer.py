import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config import METRICS_PATH, MODEL_PATH, RANDOM_STATE
from src.exception import CustomException
from src.logger import logging
from src.utils import save_json, save_object

CV_FOLDS = 5


def get_candidates():
    """Candidate models and their hyperparameter grids."""
    return {
        "Linear Regression": (LinearRegression(), {}),
        "Ridge Regression": (Ridge(), {"alpha": [0.1, 1.0, 10.0]}),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {"max_depth": [4, 6, 8, None], "min_samples_leaf": [1, 5, 10]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [100, 300], "max_depth": [None, 10], "min_samples_leaf": [1, 3]},
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"learning_rate": [0.05, 0.1], "n_estimators": [100, 300], "subsample": [0.8, 1.0]},
        ),
        "XGBoost": (
            XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"learning_rate": [0.05, 0.1], "n_estimators": [100, 300], "max_depth": [3, 5]},
        ),
        "CatBoost": (
            CatBoostRegressor(verbose=False, random_state=RANDOM_STATE, allow_writing_files=False),
            {"depth": [4, 6], "learning_rate": [0.05, 0.1], "iterations": [200, 400]},
        ),
        "AdaBoost": (
            AdaBoostRegressor(random_state=RANDOM_STATE),
            {"learning_rate": [0.05, 0.1, 0.5], "n_estimators": [50, 100]},
        ),
        "K-Nearest Neighbors": (KNeighborsRegressor(), {"n_neighbors": [5, 9, 15], "weights": ["uniform", "distance"]}),
        "Support Vector Machine (SVR)": (
            SVR(),
            {"kernel": ["rbf", "linear"], "C": [1, 10, 50], "epsilon": [0.1, 1.0]},
        ),
    }


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = MODEL_PATH
    metrics_file_path: str = METRICS_PATH


class ModelTrainer:
    def __init__(self, candidates=None):
        self.model_trainer_config = ModelTrainerConfig()
        self.candidates = candidates or get_candidates()

    def initiate_model_trainer(self, train_array, test_array):
        """Tune every candidate with 5-fold CV on the training set, select the model with the
        best mean CV R^2, then report its metrics once on the untouched test set."""
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

            leaderboard = []
            fitted = {}
            for name, (model, grid) in self.candidates.items():
                start = time.perf_counter()
                search = GridSearchCV(model, grid, cv=cv, scoring="r2", n_jobs=-1, refit=True)
                search.fit(X_train, y_train)
                idx = search.best_index_
                leaderboard.append(
                    {
                        "model": name,
                        "cv_r2_mean": round(float(search.cv_results_["mean_test_score"][idx]), 4),
                        "cv_r2_std": round(float(search.cv_results_["std_test_score"][idx]), 4),
                        "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                                        for k, v in search.best_params_.items()},
                        "tuning_seconds": round(time.perf_counter() - start, 2),
                    }
                )
                fitted[name] = search.best_estimator_
                logging.info(f"{name}: CV R2 {leaderboard[-1]['cv_r2_mean']}")

            leaderboard.sort(key=lambda r: r["cv_r2_mean"], reverse=True)
            best_name = leaderboard[0]["model"]
            best_model = fitted[best_name]

            if leaderboard[0]["cv_r2_mean"] < 0.6:
                raise CustomException("No model reached CV R2 >= 0.6", sys)

            preds = best_model.predict(X_test)
            test_metrics = {
                "r2": round(float(r2_score(y_test, preds)), 4),
                "mae": round(float(mean_absolute_error(y_test, preds)), 3),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
            }

            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            metrics = {
                "best_model": best_name,
                "selection": f"{CV_FOLDS}-fold cross-validated R2 on the training set",
                "test_metrics": test_metrics,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "leaderboard": leaderboard,
            }
            save_json(self.model_trainer_config.metrics_file_path, metrics)
            logging.info(f"Best model {best_name}; test metrics {test_metrics}")
            return metrics
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(e, sys)
