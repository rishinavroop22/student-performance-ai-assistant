"""Measure /api/predict latency.

In-process (Flask test client, no network):
    python scripts/benchmark_latency.py --requests 1000

Against a running server over HTTP:
    python application.py &
    python scripts/benchmark_latency.py --url http://localhost:5000 --requests 500
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def random_student(schema):
    s = {k: random.choice(v) for k, v in schema["categories"].items()}
    s["reading_score"] = random.randint(20, 100)
    s["writing_score"] = random.randint(20, 100)
    return s


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="Base URL of a running server (default: in-process test client)")
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--json", help="Write results to this JSON file")
    args = ap.parse_args()
    random.seed(0)

    if args.url:
        schema = {"categories": json.load(urllib.request.urlopen(f"{args.url}/api/options"))}

        def call(body):
            req = urllib.request.Request(
                f"{args.url}/api/predict", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            return json.load(urllib.request.urlopen(req))
        mode = f"HTTP ({args.url})"
    else:
        from application import create_app
        client = create_app(preload_assistant=False).test_client()
        schema = {"categories": client.get("/api/options").get_json()}

        def call(body):
            r = client.post("/api/predict", json=body)
            assert r.status_code == 200, r.get_data(as_text=True)
            return r.get_json()
        mode = "in-process Flask test client"

    for _ in range(args.warmup):
        call(random_student(schema))

    e2e, model = [], []
    for _ in range(args.requests):
        start = time.perf_counter()
        out = call(random_student(schema))
        e2e.append((time.perf_counter() - start) * 1000)
        model.append(out["inference_ms"])

    result = {
        "mode": mode,
        "requests": args.requests,
        "request_ms": {"p50": percentile(e2e, 50), "p95": percentile(e2e, 95),
                       "p99": percentile(e2e, 99), "mean": statistics.mean(e2e)},
        "model_inference_ms": {"p50": percentile(model, 50), "p95": percentile(model, 95),
                               "mean": statistics.mean(model)},
    }
    result = json.loads(json.dumps(result), parse_float=lambda x: round(float(x), 2))
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
