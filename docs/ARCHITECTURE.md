# Architecture

```mermaid
flowchart LR
    D[(Historical transactions)] --> T[train.py]
    T --> P[Validation + preprocessing]
    P --> M[XGBoost classifier]
    M --> E[ROC-AUC / PR-AUC evaluation]
    E --> A[(joblib artifact)]
    C[Client transaction] --> F[FastAPI /predict]
    A --> F
    F --> S[Feature contract validation]
    S --> I[Probability inference]
    I --> R[Fraud probability + decision]
    R --> C
    F --> H[/health + /model-info]
    F --> K[Docker image]
    K --> W[AWS ECS/Fargate or Lambda pattern]
```

## Offline training path

1. `train.py` loads the credit-card transaction CSV and validates the expected target column.
2. The data is split with stratification so the rare fraud class is represented in train and test sets.
3. The classifier uses class weighting to account for extreme imbalance.
4. Evaluation reports probability-based metrics such as ROC-AUC and PR-AUC rather than relying on accuracy.
5. The trained estimator, feature order, threshold, and metadata are persisted as one artifact.

## Online inference path

1. A client sends transaction features to `POST /predict`.
2. The FastAPI layer verifies that a model artifact is available.
3. `FraudPredictor` validates incoming feature names against the saved feature contract and preserves training-time order.
4. The model returns a fraud probability.
5. The configured threshold converts the probability into an `is_fraud` decision.
6. `/health` and `/model-info` expose deployment readiness and artifact metadata without running a prediction.

## Reliability decisions

- Training and inference are separated so the API never retrains on startup.
- The feature contract is saved with the model to avoid silent column-order bugs.
- Probabilities are returned alongside binary decisions so downstream systems can tune policy thresholds.
- The raw dataset and trained artifact are excluded from source control.
- Docker provides a reproducible runtime; GitHub Actions validates the service on pushes and pull requests.

## AWS deployment pattern

A production deployment can push the Docker image to ECR, run it on ECS/Fargate behind an Application Load Balancer, and emit prediction latency/error metrics to CloudWatch. An event-driven variant can place SQS/Kinesis in front of workers for asynchronous scoring.
