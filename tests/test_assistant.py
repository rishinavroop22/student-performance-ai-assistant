"""Assistant tests. The Hugging Face path is exercised with a stub model so CI does not need
to download weights; the real model is loaded at runtime via transformers."""
import pytest

from src.assistant import AcademicAssistant, build_insights
from src.pipeline.predict_pipeline import CustomData, PredictPipeline, get_schema


@pytest.fixture
def insights(student):
    pipeline = PredictPipeline()
    pred = pipeline.predict(CustomData(**student).get_data_as_data_frame())[0]
    return build_insights(CustomData(**student).to_dict(), pred, get_schema(), pipeline)


def test_insights_are_model_grounded(insights):
    assert insights["band"] in {"excellent", "good", "average", "at risk"}
    for w in insights["what_if"]:
        assert round(w["predicted_score"] - insights["predicted_math_score"], 1) == pytest.approx(w["gain"], abs=0.11)


def test_template_backend(insights):
    out = AcademicAssistant(backend="template").respond(insights)
    assert out["generated_by"] == "template"
    assert str(insights["predicted_math_score"]) in out["explanation"]


class _StubTokenizer:
    def __call__(self, prompt, **kw):
        self.last_prompt = prompt
        return {"input_ids": [[1, 2, 3]]}

    def decode(self, ids, skip_special_tokens=True):
        return "generated text"


class _StubModel:
    def generate(self, **kw):
        return [[4, 5, 6]]


def test_hf_backend_uses_model_and_grounded_prompt(insights):
    assistant = AcademicAssistant(backend="hf", model_name="stub")
    assistant._tokenizer, assistant._model = _StubTokenizer(), _StubModel()
    out = assistant.respond(insights)
    assert out == {"explanation": "generated text", "recommendations": "generated text", "generated_by": "stub"}
    assert str(insights["predicted_math_score"]) in assistant._tokenizer.last_prompt


def test_hf_backend_falls_back_when_model_unavailable(insights):
    assistant = AcademicAssistant(backend="hf", model_name="this/model-does-not-exist-xyz")
    assistant._load_error = "simulated"
    out = assistant.respond(insights)
    assert out["generated_by"] == "template"
