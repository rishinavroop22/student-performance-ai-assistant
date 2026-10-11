def test_health(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok" and body["model_ready"] is True


def test_options(client):
    body = client.get("/api/options").get_json()
    assert "group B" in body["race_ethnicity"]


def test_predict(client, student):
    r = client.post("/api/predict", json=student)
    assert r.status_code == 200
    body = r.get_json()
    assert 0 <= body["predicted_math_score"] <= 100
    assert body["inference_ms"] >= 0


def test_predict_validation(client, student):
    r = client.post("/api/predict", json={**student, "reading_score": 150, "lunch": "caviar"})
    assert r.status_code == 400
    errors = r.get_json()["errors"]
    assert "reading_score" in errors and "lunch" in errors


def test_predict_rejects_non_json(client):
    assert client.post("/api/predict", data="nope").status_code == 400


def test_explain(client, student):
    r = client.post("/api/explain", json=student)
    assert r.status_code == 200
    body = r.get_json()
    assert body["assistant"]["explanation"]
    assert body["assistant"]["recommendations"]
    assert all(a["gain"] > 0 for a in body["insights"]["recommended_actions"])


def test_metrics_endpoint(client):
    assert client.get("/api/metrics").get_json()["best_model"]
