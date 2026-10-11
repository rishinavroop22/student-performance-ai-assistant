import pandas as pd

from src.components.data_transformation import add_engineered_features
from src.config import METRICS_PATH
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_json


def test_engineered_features():
    df = add_engineered_features(pd.DataFrame({"reading_score": [80], "writing_score": [70]}))
    assert df["avg_literacy_score"][0] == 75
    assert df["reading_writing_gap"][0] == 10


def test_metrics_report_cv_selection():
    metrics = load_json(METRICS_PATH)
    models = {row["model"] for row in metrics["leaderboard"]}
    for required in ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost",
                     "Support Vector Machine (SVR)"]:
        assert required in models
    scores = [row["cv_r2_mean"] for row in metrics["leaderboard"]]
    assert scores == sorted(scores, reverse=True)
    assert metrics["best_model"] == metrics["leaderboard"][0]["model"]
    assert metrics["test_metrics"]["r2"] > 0.8


def test_prediction_in_range(student):
    pred = PredictPipeline().predict(CustomData(**student).get_data_as_data_frame())[0]
    assert 0 <= pred <= 100


def test_reading_and_writing_are_not_swapped(student):
    """Regression test for the old bug where the form swapped reading/writing scores."""
    data = CustomData(**student).to_dict()
    assert data["reading_score"] == 62 and data["writing_score"] == 58
