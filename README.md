# Real-Time Fraud Detection Microservice

A production-style **FastAPI microservice** for real-time credit-card fraud scoring. The repository separates offline model training from online inference and packages the prediction service with Docker and CI.

## Problem

Fraud detection systems must score transactions quickly while handling extreme class imbalance. This project trains a supervised classifier on the widely used credit-card transaction dataset (~284K transactions), persists the model artifact, and exposes low-latency fraud probabilities through a REST API.

## Architecture

```text
Historical transactions
        |
        v
Validation + preprocessing
        |
        v
Stratified train/test split
        |
        v
XGBoost classifier
(class-imbalance weighting)
        |
        v
ROC-AUC / PR-AUC evaluation
        |
        v
joblib model artifact
        |
        v
FastAPI inference service
        |
        +--> /health
        +--> /model-info
        +--> /predict
        |
        v
Docker container
        |
        v
AWS deployment pattern
(API Gateway / ALB -> ECS or Lambda)
```

## Engineering focus

- **Offline/online separation:** training code is independent from serving code.
- **Imbalance-aware modeling:** positive-class weighting is derived from training data.
- **Stable feature contract:** the saved artifact stores the exact feature order used by the API.
- **Configurable decision threshold:** probability scoring is separated from the binary fraud decision.
- **Health and metadata endpoints:** useful for deployment probes and debugging.
- **Docker + CI:** reproducible runtime and automated tests.

## Dataset

The training script expects the Kaggle/ULB credit-card fraud dataset as `data/creditcard.csv`, with columns:

`Time, V1 ... V28, Amount, Class`

`Class=1` represents fraud.

The original dataset contains approximately **284,807 transactions**. The raw CSV is intentionally not committed to this repository.

## Train the model

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p data artifacts
# place creditcard.csv inside data/
python train.py --data data/creditcard.csv --output artifacts/fraud_model.joblib
```

Training prints ROC-AUC, PR-AUC, confusion matrix, and the selected threshold, then saves a deployable artifact.

> **Metric integrity:** the resume-level ~0.97 ROC-AUC figure should only be presented as verified after running this training/evaluation pipeline on the exact dataset/version used for that result. This repository does not hard-code an evaluation metric.

## Run the API

```bash
uvicorn app.main:app --reload
```

Swagger UI: `http://localhost:8000/docs`

### Example request

```json
POST /predict
{
  "features": {
    "Time": 406.0,
    "V1": -2.312,
    "V2": 1.952,
    "V3": -1.609,
    "Amount": 0.0
  }
}
```

For a trained artifact, the request must include every feature stored in the model contract.

Example response:

```json
{
  "fraud_probability": 0.9821,
  "is_fraud": true,
  "threshold": 0.50
}
```

## Project structure

```text
app/
  main.py
  model.py
artifacts/
  .gitkeep
data/
  .gitkeep
train.py
tests/
  test_api.py
deploy/
  aws.md
.github/workflows/
  ci.yml
Dockerfile
requirements.txt
```

## AWS deployment pattern

The service can be containerized and deployed to **Amazon ECS/Fargate** behind an Application Load Balancer, or adapted for **AWS Lambda + API Gateway** for serverless inference. See `deploy/aws.md`.

## Next improvements

- Add model registry/versioning
- Add drift monitoring and feature-distribution checks
- Add authentication and rate limiting
- Add asynchronous event scoring through SQS/Kinesis
- Add CloudWatch metrics and alarms
- Add canary deployment for new model versions
