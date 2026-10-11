"""NLP academic assistant.

Turns the structured, model-grounded insights into a natural-language explanation and
personalised study recommendations using a Hugging Face seq2seq model (FLAN-T5 by default).

Environment variables:
    ASSISTANT_BACKEND   "hf" (default) or "template"
    ASSISTANT_MODEL     Hugging Face model id (default: google/flan-t5-base)

If the transformers/torch stack or the model weights are unavailable, it falls back to a
deterministic template so the API never fails because of the language model.
"""
import logging
import os
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/flan-t5-base"


def _facts(ins: Dict) -> str:
    lines = [
        f"Predicted math score: {ins['predicted_math_score']} out of 100 ({ins['band']}).",
        f"Class average: {ins['cohort_average']}. Difference: {ins['difference_from_average']:+}.",
    ]
    if ins["strengths"]:
        lines.append("Strengths: " + "; ".join(ins["strengths"]) + ".")
    if ins["risks"]:
        lines.append("Risks: " + "; ".join(ins["risks"]) + ".")
    for w in ins["recommended_actions"]:
        lines.append(f"If the student {w['action'].lower()}, predicted score becomes "
                     f"{w['predicted_score']} ({w['gain']:+} points).")
    return "\n".join(lines)


def template_explanation(ins: Dict) -> str:
    direction = "above" if ins["difference_from_average"] >= 0 else "below"
    text = (
        f"The model predicts a math score of {ins['predicted_math_score']}, which is "
        f"{abs(ins['difference_from_average'])} points {direction} the class average of "
        f"{ins['cohort_average']} and falls in the '{ins['band']}' band."
    )
    if ins["strengths"]:
        text += " Working in the student's favour: " + ", ".join(ins["strengths"]) + "."
    if ins["risks"]:
        text += " Holding the prediction back: " + ", ".join(ins["risks"]) + "."
    return text


def template_recommendations(ins: Dict) -> str:
    helpful = ins["recommended_actions"]
    if not helpful:
        return ("The student is already performing strongly on every factor the model uses. "
                "Keep practising regularly and attempt harder problem sets to stay ahead.")
    parts = [f"{w['action']} (about {w['gain']:+} points)" for w in helpful]
    return ("Highest-impact next steps according to the model: " + "; ".join(parts) +
            ". Short daily reading and written practice also supports math word problems.")


class AcademicAssistant:
    def __init__(self, backend: Optional[str] = None, model_name: Optional[str] = None):
        self.backend = (backend or os.environ.get("ASSISTANT_BACKEND", "hf")).lower()
        self.model_name = model_name or os.environ.get("ASSISTANT_MODEL", DEFAULT_MODEL)
        self._tokenizer = None
        self._model = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    # -- model loading -------------------------------------------------------------
    def _ensure_model(self) -> bool:
        if self.backend != "hf":
            return False
        if self._model is not None:
            return True
        if self._load_error:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self._model.eval()
                logger.info("Loaded assistant model %s", self.model_name)
                return True
            except Exception as e:  # missing torch, no network, bad model id...
                self._load_error = f"{type(e).__name__}: {e}"
                logger.warning("Assistant model unavailable, using template fallback: %s", self._load_error)
                return False

    def _generate(self, prompt: str, max_new_tokens: int = 120) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output = self._model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=4, no_repeat_ngram_size=3
        )
        return self._tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # -- public API ----------------------------------------------------------------
    def status(self) -> Dict:
        return {
            "backend": "huggingface" if self._model is not None else
                       ("template" if self.backend != "hf" or self._load_error else "huggingface (not loaded yet)"),
            "model": self.model_name if self.backend == "hf" else None,
            "load_error": self._load_error,
        }

    def respond(self, insights: Dict) -> Dict:
        facts = _facts(insights)
        if self._ensure_model():
            try:
                explanation = self._generate(
                    "You are an academic advisor. Using only the facts below, explain to the "
                    "student in two or three sentences why their math score is predicted this way.\n\n"
                    f"Facts:\n{facts}\n\nExplanation:"
                )
                recommendations = self._generate(
                    "You are an academic advisor. Using only the facts below, give the student "
                    "personalised study recommendations, starting with the action that gains the "
                    f"most points.\n\nFacts:\n{facts}\n\nRecommendations:"
                )
                if explanation and recommendations:
                    return {"explanation": explanation, "recommendations": recommendations,
                            "generated_by": self.model_name}
            except Exception as e:
                logger.warning("Generation failed, using template fallback: %s", e)
        return {
            "explanation": template_explanation(insights),
            "recommendations": template_recommendations(insights),
            "generated_by": "template",
        }
