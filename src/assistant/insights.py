"""Model-grounded insights: what-if analysis run through the trained model.

Every recommendation here comes from re-running the real model on a modified copy of the
student's input, so the assistant only ever talks about effects the model actually predicts.
"""
from typing import Dict, List

import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline

SCENARIOS = [
    {
        "id": "test_prep",
        "applies": lambda s: s["test_preparation_course"] != "completed",
        "change": lambda s: {**s, "test_preparation_course": "completed"},
        "action": "Complete the test preparation course",
    },
    {
        "id": "reading_up",
        "applies": lambda s: s["reading_score"] <= 90,
        "change": lambda s: {**s, "reading_score": min(100.0, s["reading_score"] + 10)},
        "action": "Raise the reading score by 10 points",
    },
    {
        "id": "writing_up",
        "applies": lambda s: s["writing_score"] <= 90,
        "change": lambda s: {**s, "writing_score": min(100.0, s["writing_score"] + 10)},
        "action": "Raise the writing score by 10 points",
    },
]


def performance_band(score: float) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 55:
        return "average"
    return "at risk"


def build_insights(student: Dict, prediction: float, schema: Dict, pipeline: PredictPipeline) -> Dict:
    mean = schema["target_mean"]
    scenarios = [s for s in SCENARIOS if s["applies"](student)]
    what_if: List[Dict] = []
    if scenarios:
        rows = pd.DataFrame([s["change"](student) for s in scenarios])
        new_preds = pipeline.predict(rows)
        for s, new in zip(scenarios, new_preds):
            what_if.append(
                {
                    "id": s["id"],
                    "action": s["action"],
                    "predicted_score": round(new, 1),
                    "gain": round(new - prediction, 1),
                }
            )
        what_if.sort(key=lambda w: w["gain"], reverse=True)

    strengths, risks = [], []
    lit = (student["reading_score"] + student["writing_score"]) / 2
    (strengths if lit >= 70 else risks).append(
        f"average reading/writing score of {lit:.0f}"
    )
    # Test prep is deliberately not labelled a strength/risk here: once reading and writing
    # scores are known, its effect on the math prediction is small and can even be negative,
    # so it is judged only through the what-if scenario below.
    if student["lunch"] == "free/reduced":
        risks.append("free/reduced lunch (associated with lower scores in the training data)")

    return {
        "predicted_math_score": round(prediction, 1),
        "band": performance_band(prediction),
        "cohort_average": round(mean, 1),
        "difference_from_average": round(prediction - mean, 1),
        "strengths": strengths,
        "risks": risks,
        "what_if": what_if,
        "recommended_actions": [w for w in what_if if w["gain"] > 0.5],
    }
