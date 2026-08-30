# AWS Deployment Design

A practical production path for this service is:

```text
Client
  |
  v
Application Load Balancer / API Gateway
  |
  v
ECS Fargate task (FastAPI container)
  |
  +--> model artifact bundled in image or downloaded from S3 at startup
  +--> CloudWatch logs and metrics
  +--> Auto Scaling based on CPU/request volume
```

## Recommended AWS components

- **Amazon ECR** — Docker image registry
- **Amazon ECS + Fargate** — managed container runtime
- **Application Load Balancer** — HTTPS traffic routing and health checks
- **Amazon S3** — versioned model artifacts
- **AWS IAM** — least-privilege access to model artifacts
- **Amazon CloudWatch** — logs, latency, error rate, and alerting
- **GitHub Actions** — test/build pipeline; deployment can be added with OIDC instead of long-lived AWS keys

## Deployment flow

1. GitHub Actions runs tests.
2. Build the Docker image.
3. Authenticate to AWS using GitHub OIDC.
4. Push the image to ECR.
5. Update the ECS task definition/service.
6. ECS performs a rolling deployment.
7. ALB checks `/health` before routing production traffic.

For high-throughput event-driven scoring, transactions can also be published to Kinesis/SQS and scored asynchronously by a separate worker service.
