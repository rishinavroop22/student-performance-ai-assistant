"""Flask REST API + static host for the React frontend.

    python application.py            # http://localhost:5000
"""
import os
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

from src.assistant import AcademicAssistant, build_insights
from src.config import METRICS_PATH, NUMERICAL_COLUMNS, ROOT_DIR
from src.pipeline.predict_pipeline import (
    CustomData,
    ModelNotTrainedError,
    PredictPipeline,
    get_schema,
)
from src.utils import load_json

FRONTEND_DIST = os.path.join(ROOT_DIR, "frontend", "dist")


def validate_payload(payload, schema):
    """Return (CustomData, None) or (None, errors)."""
    if not isinstance(payload, dict):
        return None, {"_": "Request body must be a JSON object."}
    errors, clean = {}, {}
    for field, allowed in schema["categories"].items():
        value = payload.get(field)
        if value not in allowed:
            errors[field] = f"Must be one of: {', '.join(allowed)}."
        else:
            clean[field] = value
    for field in NUMERICAL_COLUMNS:
        value = payload.get(field)
        try:
            number = float(value)
            if not 0 <= number <= 100:
                raise ValueError
            clean[field] = number
        except (TypeError, ValueError):
            errors[field] = "Must be a number between 0 and 100."
    if errors:
        return None, errors
    return CustomData(**clean), None


def create_app(assistant: AcademicAssistant = None, preload_assistant: bool = None):
    app = Flask(__name__, static_folder=None)
    app.config["JSON_SORT_KEYS"] = False
    app.assistant = assistant or AcademicAssistant()

    if preload_assistant is None:
        preload_assistant = os.environ.get("ASSISTANT_PRELOAD", "1") == "1"
    if preload_assistant:
        threading.Thread(target=app.assistant._ensure_model, daemon=True).start()

    def model_unavailable(e):
        return jsonify({"error": str(e)}), 503

    app.register_error_handler(ModelNotTrainedError, model_unavailable)

    @app.get("/api/health")
    def health():
        try:
            get_schema()
            model_ready = True
        except ModelNotTrainedError:
            model_ready = False
        return jsonify({"status": "ok", "model_ready": model_ready, "assistant": app.assistant.status()})

    @app.get("/api/options")
    def options():
        return jsonify(get_schema()["categories"])

    @app.get("/api/metrics")
    def metrics():
        if not os.path.exists(METRICS_PATH):
            raise ModelNotTrainedError("Metrics not found. Train the model first.")
        return jsonify(load_json(METRICS_PATH))

    @app.post("/api/predict")
    def predict():
        schema = get_schema()
        data, errors = validate_payload(request.get_json(silent=True), schema)
        if errors:
            return jsonify({"errors": errors}), 400
        start = time.perf_counter()
        prediction = PredictPipeline().predict(data.get_data_as_data_frame())[0]
        inference_ms = (time.perf_counter() - start) * 1000
        return jsonify({
            "predicted_math_score": round(prediction, 1),
            "inference_ms": round(inference_ms, 2),
        })

    @app.post("/api/explain")
    def explain():
        schema = get_schema()
        data, errors = validate_payload(request.get_json(silent=True), schema)
        if errors:
            return jsonify({"errors": errors}), 400
        start = time.perf_counter()
        pipeline = PredictPipeline()
        prediction = pipeline.predict(data.get_data_as_data_frame())[0]
        insights = build_insights(data.to_dict(), prediction, schema, pipeline)
        assistant = app.assistant.respond(insights)
        return jsonify({
            "insights": insights,
            "assistant": assistant,
            "total_ms": round((time.perf_counter() - start) * 1000, 1),
        })

    # React build (frontend/dist) served from the same origin.
    @app.get("/", defaults={"path": ""})
    @app.get("/<path:path>")
    def frontend(path):
        if path.startswith("api/"):
            return jsonify({"error": "Not found"}), 404
        if not os.path.isdir(FRONTEND_DIST):
            return (
                "Frontend not built. Run `npm install && npm run build` in ./frontend, "
                "or use the Vite dev server (npm run dev).",
                200,
            )
        target = os.path.join(FRONTEND_DIST, path)
        if path and os.path.isfile(target):
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, "index.html")

    return app


app = application = create_app() if __name__ == "__main__" else None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
