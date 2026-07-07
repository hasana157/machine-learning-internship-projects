# CineMatch AI - Deployment Guide

## Quick Start (Local Development)

```bash
# Clone repository
git clone <repository-url>
cd CineMatch

# Setup environment
cp .env.example .env

# Start all services
docker-compose up -d

# Initialize database
docker-compose exec backend python -m alembic upgrade head

# Load sample data (optional)
docker-compose exec backend python scripts/load_sample_data.py
```

**Services will be available at:**
- Frontend: http://localhost:3000
- API: http://localhost:8000
- Gradio Demo: http://localhost:7860
- PgAdmin: http://localhost:5050
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

---

## Production Deployment on AWS

### Prerequisites

- AWS Account with appropriate IAM permissions
- AWS CLI configured
- Docker and Docker Compose installed locally
- Terraform (optional, for Infrastructure as Code)

### Architecture

```
CloudFront CDN
    ↓
Application Load Balancer (ALB)
    ↓
ECS Fargate Cluster
├── Frontend (Next.js)
├── Backend (FastAPI)
└── ML Service (Gradio)
    ↓
RDS Aurora PostgreSQL
Redis ElastiCache
S3 (Model artifacts, static assets)
```

### Step 1: Prepare AWS Resources

#### 1.1 Create ECR Repositories

```bash
# Create repositories for each service
aws ecr create-repository --repository-name cinematch/frontend --region us-east-1
aws ecr create-repository --repository-name cinematch/backend --region us-east-1
aws ecr create-repository --repository-name cinematch/ml --region us-east-1
```

#### 1.2 Create RDS Aurora PostgreSQL Cluster

```bash
aws rds create-db-cluster \
  --db-cluster-identifier cinematch-prod-db \
  --engine aurora-postgresql \
  --engine-version 16.1 \
  --master-username admin \
  --master-user-password <SECURE_PASSWORD> \
  --database-name cinematch_db \
  --db-subnet-group-name <subnet-group> \
  --vpc-security-group-ids <security-group-id> \
  --region us-east-1
```

#### 1.3 Create ElastiCache Redis Cluster

```bash
aws elasticache create-cache-cluster \
  --cache-cluster-id cinematch-prod-redis \
  --cache-node-type cache.r7g.large \
  --engine redis \
  --engine-version 7.2 \
  --num-cache-nodes 3 \
  --cache-subnet-group-name <subnet-group> \
  --security-group-ids <security-group-id> \
  --region us-east-1
```

#### 1.4 Create S3 Buckets

```bash
# Model artifacts
aws s3 mb s3://cinematch-models-prod --region us-east-1
aws s3api put-bucket-versioning \
  --bucket cinematch-models-prod \
  --versioning-configuration Status=Enabled

# Frontend static assets
aws s3 mb s3://cinematch-frontend-prod --region us-east-1
```

### Step 2: Build and Push Docker Images

```bash
# Setup
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

# Frontend
docker build -t cinematch/frontend:latest ./frontend
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker tag cinematch/frontend:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/cinematch/frontend:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/cinematch/frontend:latest

# Backend
docker build -t cinematch/backend:latest ./backend
docker tag cinematch/backend:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/cinematch/backend:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/cinematch/backend:latest

# ML Service
docker build -t cinematch/ml:latest ./ml
docker tag cinematch/ml:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/cinematch/ml:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/cinematch/ml:latest
```

### Step 3: Create ECS Cluster and Services

```bash
# Create cluster
aws ecs create-cluster --cluster-name cinematch-prod

# Create task definitions (use CloudFormation template or AWS console)
# See terraform/ecs.tf for IaC example

# Create services with auto-scaling
```

### Step 4: Configure CloudFront CDN

```bash
# Create CloudFront distribution pointing to S3 + ALB
aws cloudfront create-distribution \
  --distribution-config file://cloudfront-config.json \
  --region us-east-1
```

### Step 5: Setup CI/CD Pipeline

#### Using GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Build and push Docker images
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin ...
          docker build -t cinematch/backend:latest ./backend
          docker push ...
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster cinematch-prod \
            --service backend \
            --force-new-deployment
```

### Step 6: Configure Environment Variables

Create production `.env` file with secure values:

```bash
ENVIRONMENT=production
SECRET_KEY=<64-character-secure-key>
DATABASE_URL=postgresql+asyncpg://admin:password@cinematch-prod-db.*.rds.amazonaws.com:5432/cinematch_db
REDIS_URL=redis://:password@cinematch-prod-redis.*.ng.0001.use1.cache.amazonaws.com:6379/0
AWS_REGION=us-east-1
AWS_S3_BUCKET=cinematch-models-prod
```

Use AWS Secrets Manager:

```bash
aws secretsmanager create-secret \
  --name cinematch/prod/db-password \
  --secret-string '<secure-password>' \
  --region us-east-1
```

### Step 7: Database Migrations

```bash
# Run migrations on production database
docker run --rm \
  -e DATABASE_URL='postgresql+asyncpg://...' \
  cinematch/backend:latest \
  alembic upgrade head
```

### Step 8: Load Initial Data

```bash
docker run --rm \
  -e DATABASE_URL='postgresql+asyncpg://...' \
  cinematch/backend:latest \
  python scripts/load_movielens_data.py
```

---

## Monitoring & Observability

### CloudWatch Logs

```bash
# View backend logs
aws logs tail /aws/ecs/cinematch-backend --follow

# View error logs
aws logs filter-log-events \
  --log-group-name /aws/ecs/cinematch-backend \
  --filter-pattern "ERROR"
```

### CloudWatch Dashboards

```bash
# Create custom dashboard
aws cloudwatch put-dashboard \
  --dashboard-name cinematch-prod \
  --dashboard-body file://dashboards/prod-dashboard.json
```

### Prometheus & Grafana

Access at `https://monitoring.cinematch.ai`

**Key Metrics:**
- API Response Latency
- Cache Hit Rate
- Database Connection Pool Usage
- Model Inference Latency
- Error Rates

### Alerts

Set up CloudWatch alarms:

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name cinematch-high-error-rate \
  --alarm-description "Alert when error rate > 1%" \
  --metric-name ErrorRate \
  --namespace CineMatch \
  --statistic Average \
  --period 300 \
  --threshold 1 \
  --comparison-operator GreaterThanThreshold \
  --alarm-actions arn:aws:sns:us-east-1:...:cinematch-alerts
```

---

## Scaling

### Auto-scaling Policies

```bash
# Backend service
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/cinematch-prod/backend \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 20 \
  --region us-east-1

# Create scaling policy
aws application-autoscaling put-scaling-policy \
  --policy-name backend-cpu-scaling \
  --service-namespace ecs \
  --resource-id service/cinematch-prod/backend \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

---

## Disaster Recovery

### Database Backup

```bash
# Enable automated backups
aws rds modify-db-cluster \
  --db-cluster-identifier cinematch-prod-db \
  --backup-retention-period 30 \
  --enable-cloudwatch-logs-exports postgresql

# Manual snapshot
aws rds create-db-cluster-snapshot \
  --db-cluster-snapshot-identifier cinematch-backup-2024-11-01 \
  --db-cluster-identifier cinematch-prod-db
```

### Restore from Backup

```bash
aws rds restore-db-cluster-from-snapshot \
  --db-cluster-identifier cinematch-restored-db \
  --snapshot-identifier cinematch-backup-2024-11-01 \
  --engine aurora-postgresql
```

---

## Cost Optimization

### Recommendations

1. **Use Spot Instances**: Save 70% on compute with ECS Spot
2. **Enable RDS Auto-scaling**: Resize based on workload
3. **CloudFront Caching**: Reduce origin requests
4. **Reserved Capacity**: Commit for 1-3 year discounts
5. **Monitor Costs**: Use AWS Cost Explorer

---

## Troubleshooting

### Check Service Status

```bash
# List running tasks
aws ecs list-tasks --cluster cinematch-prod

# Get task details
aws ecs describe-tasks --cluster cinematch-prod --tasks <task-arn>

# View logs
aws logs get-log-events --log-group-name /aws/ecs/cinematch-backend
```

### Common Issues

**502 Bad Gateway**
- Check ALB target health
- Verify backend is running: `docker ps`
- Check security groups allow traffic

**Database Connection Errors**
- Verify RDS security group allows port 5432
- Check connection string
- Verify database is running

**High Latency**
- Check CloudWatch metrics
- Review Prometheus dashboards
- Check cache hit rate
- Profile with X-Ray

---

## Production Checklist

- [ ] Environment variables configured securely
- [ ] Database backups enabled
- [ ] SSL/TLS certificates installed
- [ ] CloudFront distribution configured
- [ ] Auto-scaling policies enabled
- [ ] CloudWatch alarms configured
- [ ] Monitoring dashboards created
- [ ] Security groups properly configured
- [ ] IAM roles with least privilege
- [ ] Secrets Manager configured
- [ ] CI/CD pipeline tested
- [ ] Disaster recovery tested
- [ ] Load testing completed
- [ ] Security audit completed
- [ ] Documentation updated

---

## Rollback Procedure

```bash
# Update service to previous image version
aws ecs update-service \
  --cluster cinematch-prod \
  --service backend \
  --force-new-deployment \
  --image cinematch/backend:v1.0.0

# Monitor rollout
aws ecs describe-services --cluster cinematch-prod --services backend
```

---

## Support

For issues during deployment:

1. Check CloudWatch logs
2. Review Terraform state
3. Verify AWS credentials
4. Contact AWS support

Reference: AWS documentation for ECS, RDS, ElastiCache
