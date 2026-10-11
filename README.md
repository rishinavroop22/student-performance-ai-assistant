# AI-Powered Student Performance Prediction & Intelligent Academic Assistant

Predicts a student's **math score** from demographic and academic features. An **NLP academic
assistant** built on Hugging Face Transformers then explains the prediction and recommends what
to work on next. The stack is a Flask REST API with a React frontend, tested and benchmarked in
GitHub Actions.

![CI](https://github.com/rishinavroop22/Enterprise-AI-Knowledge-Assistant/actions/workflows/ci.yml/badge.svg)

## Results

Dataset: 1,000 students (`src/notebook/data/stud.csv`), split 80/20 with a fixed seed.

| | |
|---|---|
| Models compared | 10, each tuned with `GridSearchCV` |
| Selection method | 5-fold cross-validated R² on the training set |
| Selected model | Ridge Regression (CV R² 0.869 ± 0.014) |
| Held-out test set | **R² 0.880 · MAE 4.22 · RMSE 5.40** |
| `/api/predict` latency | **~8 ms p50 / ~12 ms p95** in-process, **~10 ms p50 / ~15 ms p95** over local HTTP (1,000 and 300 requests, CPU) |

<details><summary>Full cross-validation leaderboard</summary>

| Model | CV R² (mean ± std) |
|---|---|
| Ridge Regression | 0.869 ± 0.014 |
| Linear Regression | 0.869 ± 0.014 |
| Support Vector Machine (SVR) | 0.868 ± 0.014 |
| CatBoost | 0.860 ± 0.014 |
| Gradient Boosting | 0.856 ± 0.013 |
| XGBoost | 0.855 ± 0.015 |
| Random Forest | 0.840 ± 0.020 |
| AdaBoost | 0.818 ± 0.018 |
| Decision Tree | 0.800 ± 0.019 |
| K-Nearest Neighbors | 0.666 ± 0.020 |

</details>

Linear models win because math score is close to linear in reading and writing scores. The tree
ensembles have no non-linearity to exploit and only add variance.

## How it works

```
stud.csv ─► DataIngestion ─► DataTransformation ─► ModelTrainer ─► artifacts/
            dedupe, split    feature engineering    10 models × GridSearchCV
                             impute · one-hot       5-fold CV selection
                             scale                  test-set metrics → metrics.json

React UI ─► Flask API ─► PredictPipeline (cached model) ─► prediction
                     └─► insights (what-if analysis through the model)
                          └─► AcademicAssistant (FLAN-T5) ─► explanation + recommendations
```

- **Feature engineering.** Adds `avg_literacy_score` and `reading_writing_gap` alongside the raw
  scores and one-hot encoded categorical features.
- **Model-grounded assistant.** The assistant never makes up effects. `src/assistant/insights.py`
  re-runs the trained model on what-if variants of the student's input (complete test prep,
  +10 reading, +10 writing) and keeps only the actions that actually raise the prediction. Those
  facts go into a prompt for `google/flan-t5-base`, which writes the explanation and
  recommendations.
- **Graceful fallback.** If PyTorch or the model weights are unavailable, the assistant switches
  to deterministic template text, so the API keeps working.

## Project structure

```
application.py                 Flask API + serves the React build
src/
  components/                  data_ingestion, data_transformation, model_trainer
  pipeline/                    train_pipeline (CLI), predict_pipeline (cached inference)
  assistant/                   insights (what-if analysis), generator (Hugging Face model)
  config.py, logger.py, exception.py, utils.py
  notebook/                    EDA and model-training notebooks, dataset
frontend/                      React + Vite UI
scripts/benchmark_latency.py   latency benchmark (in-process or over HTTP)
tests/                         pytest: pipeline, API, assistant
.github/workflows/ci.yml       train → test → benchmark, plus frontend build
```

## Run it locally

```bash
# 1. Backend
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-assistant.txt            # includes torch for the HF model
python -m src.pipeline.train_pipeline                # trains and writes artifacts/

# 2. Frontend
cd frontend && npm install && npm run build && cd ..

# 3. Start
python application.py                                # http://localhost:5000
```

The first start downloads `google/flan-t5-base` (~1 GB) from Hugging Face. To skip the
language model, use `pip install -r requirements.txt` and `ASSISTANT_BACKEND=template`.
To use a lighter model, set `ASSISTANT_MODEL=google/flan-t5-small`.

For frontend development with hot reload, run `npm run dev` in `frontend/`. It proxies `/api` to
Flask on port 5000.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Model and assistant status |
| GET | `/api/options` | Allowed values for each categorical field |
| GET | `/api/metrics` | CV leaderboard and test metrics |
| POST | `/api/predict` | Predicted score and `inference_ms` |
| POST | `/api/explain` | Prediction, what-if insights, assistant explanation and recommendations |

```bash
curl -X POST localhost:5000/api/explain -H "Content-Type: application/json" -d '{
  "gender": "female", "race_ethnicity": "group B",
  "parental_level_of_education": "bachelor'"'"'s degree", "lunch": "standard",
  "test_preparation_course": "none", "reading_score": 72, "writing_score": 74 }'
```

Invalid input returns `400` with an error for each field.

## Tests and benchmark

```bash
pip install -r requirements-dev.txt
pytest -v
python scripts/benchmark_latency.py --requests 1000                       # in-process
python scripts/benchmark_latency.py --url http://localhost:5000           # live server
```

GitHub Actions runs training, the tests and the benchmark on every push and uploads
`metrics.json` and `latency.json` as build artifacts. It also builds the React app.
