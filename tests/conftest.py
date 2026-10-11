import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.config import MODEL_PATH  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Train once if artifacts are missing (e.g. fresh CI checkout)."""
    if not os.path.exists(MODEL_PATH):
        from src.pipeline.train_pipeline import TrainPipeline
        TrainPipeline().run()


@pytest.fixture
def student():
    return {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "free/reduced",
        "test_preparation_course": "none",
        "reading_score": 62,
        "writing_score": 58,
    }


@pytest.fixture
def client():
    from application import create_app
    from src.assistant import AcademicAssistant

    app = create_app(assistant=AcademicAssistant(backend="template"), preload_assistant=False)
    return app.test_client()
