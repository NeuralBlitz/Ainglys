# Supervised training with held-out validation

Ainglys now has an authenticated classification endpoint for **user-provided numeric datasets**. This is an engineering prototype for small tabular datasets, not a claim of scientifically validated performance.

## Endpoints

- `POST /train`: existing synthetic-data demo, with random labels. Demo accuracy is **not** a meaningful benchmark.
- `POST /train/supervised`: train a new classifier on labeled numeric data and report performance on a separate, stratified validation holdout.
- `POST /predict`: use the currently active trained classifier for inference.
- `GET /models`, `POST /save`, `POST /load/{model_name}`: existing model lifecycle operations.

Both training endpoints require `X-API-Key` equal to the server's `AI_HUB_API_KEY` (at least 32 characters). An absent key fails closed.

## Input contract

JSON request object:

| Field | Required | Constraints |
| --- | --- | --- |
| `features` | Yes | 16–512 rows; 1–128 columns per row, same width; finite numeric values, each within ±1,000,000 |
| `labels` | Yes | One integer per row, 2–32 classes; contiguous IDs `0..C-1`; at least 2 samples in every class |
| `hidden_dim` | No | 1–256 (default 32) |
| `epochs` | No | 1–20 (default 5) |
| `learning_rate` | No | 0.000001–0.1 (default 0.001) |
| `validation_fraction` | No | 0.1–0.4 (default 0.2); actual size is rounded separately for each class |
| `seed` | No | Nonnegative 32-bit signed integer (default 42) |

Unknown fields are rejected. Empty classes, mismatched labels, NaN/infinite features, and nonconsecutive IDs receive HTTP 422. A concurrent training or model-load operation receives HTTP 429.

## CLI example

Generate a **synthetic demonstration dataset**; replace this file with your own real records to work with real data.

```bash
python - <<'PY'
import json

dataset = {
    "features": [[float(i) / 40, float(i % 3)] for i in range(40)],
    "labels": [i % 2 for i in range(40)],
    "hidden_dim": 12,
    "epochs": 2,
    "learning_rate": 0.01,
    "validation_fraction": 0.25,
    "seed": 101
}
with open("training-example.json", "w", encoding="utf-8") as output:
    json.dump(dataset, output)
PY

# Configure the exact same secret on the API server and CLI.
export AI_HUB_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"

# Start the local server in another terminal, inheriting AI_HUB_API_KEY.
ai-hub api start --port 8000

# Submit data, receive training and held-out evaluation metrics.
ai-hub model train-data training-example.json
```

For an explicitly selected remote host, use `--api-url https://your-trusted-host.example`. The CLI refuses non-TLS remote URLs and rejects credential-bearing URLs.

Equivalent authenticated request:

```bash
curl --fail-with-body \
  -H "X-API-Key: $AI_HUB_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @training-example.json \
  http://127.0.0.1:8000/train/supervised
```

## Response

```json
{
  "status": "trained",
  "data_source": "user_provided_supervised",
  "model_info": {"input_dim": 2, "hidden_dim": 12, "output_dim": 2},
  "dataset": {
    "samples": 40,
    "training_samples": 30,
    "validation_samples": 10,
    "classes": 2,
    "validation_fraction_requested": 0.25,
    "random_seed": 101,
    "split_strategy": "stratified_holdout"
  },
  "training": {"final_loss": 0.0, "epochs": 2},
  "validation": {"accuracy": 0.0, "cross_entropy": 0.0}
}
```

**Important:** The example numbers `0.0` in the last two sections are placeholders illustrating the response schema, **not measured results**. Actual values come from the supplied dataset and trained model.

## Experimental integrity and limitations

1. The stratified split uses record indices and a seeded local random generator. Each class has at least one record in training and validation, and index sets are disjoint. No validation record is sent to the training optimizer.
2. The train/test distinction is **not** sufficient to establish generalization if duplicate records, correlated observations, repeated patients, grouped subjects, or time-series leakage are present. Users must prepare their data appropriately and introduce entity/time-aware splits externally where necessary.
3. No preprocessing, standardization, cross-validation, class weighting, confidence intervals, fairness evaluation, or external independent test set is performed here. Such evaluations remain future work.
4. The seed stabilizes index selection and initializes the model RNG, but deterministic GPU results are not guaranteed across hardware/framework versions.
5. Training is bounded and executed within the API process. This initial endpoint is not a distributed queue; avoid exposing it publicly without production-grade identity, quotas, auditing and monitoring.
6. No metric in this endpoint should be described as state of the art, independently replicated, or production validated without a separate experimental study.

The existing model remains active if the new training or its validation fails. After a successful run, it becomes the active in-memory model. Persist it separately using `POST /save` (the existing model lifecycle API).

For follow-up engineering and security requirements, see [Ainglys issue #3](https://github.com/NeuralBlitz/Ainglys/issues/3), [issue #5](https://github.com/NeuralBlitz/Ainglys/issues/5) and [issue #6](https://github.com/NeuralBlitz/Ainglys/issues/6).
