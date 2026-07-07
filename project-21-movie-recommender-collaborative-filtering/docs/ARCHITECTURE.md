# CineMatch AI - System Architecture

## Overview

CineMatch AI is a production-grade, scalable movie recommendation system built with modern cloud-native technologies. The system implements collaborative filtering with multiple recommendation algorithms, real-time caching, and comprehensive monitoring.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                       │
│          (Web Browser, Mobile, Third-party Apps)            │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTPS
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    CloudFront CDN                            │
│              (Caching, DDoS Protection)                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Application Load Balancer (ALB)                   │
│         (HTTPS, Request routing, Rate limiting)             │
└─┬─────────────────┬──────────────────┬─────────────────────┐
  │                 │                  │                     │
  ▼                 ▼                  ▼                     ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐
│ Frontend │  │ Backend  │  │    ML    │  │ Admin Panel  │
│(Next.js) │  │(FastAPI) │  │ (Gradio) │  │              │
└──┬───────┘  └──┬───────┘  └──┬───────┘  └──────────────┘
   │             │             │
   └─────────────┼─────────────┘
                 │
         ┌───────┴────────┐
         │                │
         ▼                ▼
    ┌─────────┐      ┌─────────┐
    │PostgreSQL │     │  Redis  │
    │    RDS    │     │  Cache  │
    └─────────┘      └─────────┘

┌─────────────────────────────────────────────────────────────┐
│              Monitoring & Observability                      │
│     Prometheus  │  Grafana  │  CloudWatch  │  X-Ray         │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Frontend Layer (Next.js 14)

**Responsibilities:**
- Server-side rendering (SSR) for SEO
- Real-time recommendation updates (SWR)
- User authentication & profile management
- Movie discovery with filters/search
- Gradio demo embedding

**Technology:**
- Next.js 14 App Router
- Tailwind CSS (cinematic dark theme)
- Framer Motion (animations)
- TypeScript (type safety)
- SWR (data fetching)

**Key Features:**
- Responsive design (mobile-first)
- Dark theme matching streaming platforms
- Optimized images (WebP/AVIF)
- Incremental Static Regeneration (ISR)
- API route handlers for backend proxying

### 2. Backend API Layer (FastAPI)

**Responsibilities:**
- REST API endpoint serving
- JWT authentication & authorization
- Business logic orchestration
- Database transaction management
- Cache invalidation
- Event logging
- Rate limiting

**Technology:**
- FastAPI (async Python)
- Pydantic (data validation)
- SQLAlchemy 2.x (async ORM)
- python-jose (JWT tokens)
- slowapi (rate limiting)

**Key Endpoints:**
```
POST   /api/v1/auth/login              - Authenticate user
POST   /api/v1/auth/register           - Create account
GET    /api/v1/recommendations/{id}    - Get recommendations
GET    /api/v1/movies/{id}             - Movie details
POST   /api/v1/events                  - Log interactions
POST   /api/v1/events/rate             - Submit ratings
GET    /health/ready                   - Readiness probe
```

**Response Times:**
- Cache hit: <20ms
- Cache miss: <250ms
- Database query: <50ms
- Model inference: <100ms

### 3. ML/Recommendation Engine

**Responsibilities:**
- Model training on historical data
- Recommendation generation
- Similarity calculations
- Model versioning & management
- Interactive ML exploration (Gradio)

**Algorithms Implemented:**

#### Tier 1: Baseline (Fallback)
- Global mean + bias
- Use case: Cold-start (< 5 ratings)
- Performance: RMSE < 1.0

#### Tier 2: KNN-Based
- User-User KNN with cosine similarity
- Item-Item similarity
- Use case: Sparse users (5-50 ratings)
- Performance: Hit@10 > 0.50

#### Tier 3: Matrix Factorization
- Singular Value Decomposition (SVD)
- n_factors = 128
- Biased SVD (user + item biases)
- Use case: Primary recommendation
- Performance: Hit@10 > 0.65

#### Tier 4: Implicit Feedback
- Alternating Least Squares (ALS)
- Incorporates clicks, views, watches
- Use case: Engagement signals
- Performance: Precision@10 > 0.45

#### Ensemble
- Weighted blend: SVD (70%) + ALS (30%)
- Dynamic weighting based on confidence
- Performance: NDCG@10 > 0.42

**Training Pipeline:**
```
1. Data Loading          → 25M ratings from PostgreSQL
2. Matrix Construction   → Sparse CSR format (162k × 62k)
3. Normalization         → Mean-centering per user
4. Train/Val/Test Split  → 80/10/10 by timestamp
5. Hyperparameter Tuning → Optuna (100 trials, Bayesian)
6. Model Training        → Surprise library
7. Evaluation            → Hit@K, NDCG@K, Coverage metrics
8. Versioning            → Save to S3 with metadata
9. Model Loading         → Hot-reload in inference service
```

**Inference Pipeline:**
```
User Request
    │
    ├─ Check Cache (Redis) ──► HIT ──► Return (< 20ms)
    │
    └─ MISS
        │
        ├─ Load user factor vector from PostgreSQL
        ├─ Compute dot-product with all item factors (Faiss)
        ├─ Get top-200 candidates
        ├─ Apply filters:
        │   ├─ Remove already-rated items
        │   ├─ Remove low-confidence (< threshold)
        │   ├─ Apply diversity constraint
        │   └─ Sort by score + recency bias
        ├─ Return top-10 with explanations
        ├─ Cache result (TTL: 1 hour)
        └─ Return (< 250ms)
```

### 4. Data Layer

#### PostgreSQL (RDS Aurora)
```
Tables:
├── users               - User accounts
├── movies              - Movie catalog (62k)
├── ratings             - User-movie ratings (25M)
├── user_events         - Interaction logs
├── model_runs          - Training metadata
└── recommendation_cache - Pre-computed recs

Indexes:
├── user_id on ratings (for per-user queries)
├── movie_id on ratings (for per-movie stats)
├── timestamp on events (for time-range queries)
├── composite indexes for common joins
└── partial indexes for active users
```

**Connections:**
- Pool size: 20
- Max overflow: 10
- Recycle time: 1 hour
- Async driver: asyncpg

#### Redis (ElastiCache)
```
Cache Keys:
├── rec:{user_id}:{k}           - Recommendations (1h TTL)
├── user:{user_id}:profile      - User data (24h TTL)
├── movie:{movie_id}:stats      - Movie stats (6h TTL)
├── session:{token}             - Auth sessions (30m TTL)
├── rate:{user_id}              - Rate limiting counter (1m)
└── model:current:version       - Active model version

Cluster Mode: 3 nodes
Replication: 2 replicas per shard
Eviction policy: allkeys-lru
```

### 5. Storage Layer

#### S3 (Model Artifacts)
```
Bucket: cinematch-models-prod

Structure:
├── models/
│   ├── svd/
│   │   ├── v1.pkl
│   │   ├── v1.metadata.json
│   │   ├── v2.pkl
│   │   └── ...
│   ├── als/
│   ├── ensemble/
│   └── active.txt (pointer to active version)
├── data/
│   ├── processed/
│   │   ├── ratings.parquet
│   │   ├── user_factors.npy
│   │   └── item_factors.npy
│   └── raw/
├── reports/
│   ├── training_metrics.json
│   └── model_cards/
└── backups/
```

**Versioning:** S3 versioning enabled, point-in-time restore possible

### 6. Caching Strategy

```
Layer 1: CDN Cache (CloudFront)
├── Static assets (JS, CSS, images)
├── TTL: 1 hour to 30 days
├── Compression: gzip, brotli
└── Invalidation: Manual on deploy

Layer 2: Application Cache (Redis)
├── Recommendations: 1 hour TTL
├── User data: 24 hours TTL
├── Movie stats: 6 hours TTL
├── Session tokens: 30 minutes TTL
└── Hit target: > 80%

Layer 3: HTTP Client Cache (SWR)
├── Frontend data fetching
├── Stale-while-revalidate
├── Auto-revalidation every 30s
└── Manual invalidation on mutation

Layer 4: Browser Cache
├── Service workers for offline support
├── Indexed DB for client state
└── Max age: Depends on asset type
```

**Cache Invalidation Strategy:**
```
Event Triggers:
├── User rates movie        → Invalidate user's recommendations
├── Model retrained         → Invalidate all recommendation caches
├── Movie metadata updated  → Invalidate specific movie cache
├── User profile changed    → Invalidate user session
└── Manual cache flush      → Admin operation
```

## Data Flow Diagrams

### Recommendation Request Flow

```
1. Frontend (Next.js)
   └─ Send: GET /api/v1/recommendations/123
      Headers: Authorization: Bearer {token}

2. ALB
   └─ Route to Backend service
   └─ Rate limit check (100/min)

3. Backend (FastAPI)
   ├─ Validate JWT token
   ├─ Check user exists
   └─ Call recommendation service

4. Recommendation Service
   ├─ Check Redis cache
   │  ├─ HIT → Return cached recommendations
   │  └─ MISS → Continue to inference
   │
   ├─ Load user factors from PostgreSQL
   ├─ Compute similarity scores (Faiss)
   ├─ Apply post-filters
   ├─ Generate explanations
   ├─ Cache result in Redis (1h)
   └─ Return recommendations

5. Response
   ├─ 200 OK with recommendations
   ├─ Include: scores, explanations, latency
   └─ Client renders recommendations
```

### Rating Submission Flow

```
1. User submits rating
   └─ Frontend: POST /api/v1/events/rate

2. Backend validation
   ├─ Check auth token
   ├─ Validate rating (0.5-5.0)
   └─ Check movie exists

3. Database write
   ├─ Insert/update rating in PostgreSQL
   └─ Transaction committed

4. Cache invalidation
   └─ Delete user's cached recommendations from Redis

5. Event logging
   └─ Async task: Log event for analytics

6. Model retraining trigger
   └─ If sufficient new ratings: Queue retraining job

7. Response
   └─ 200 OK with confirmation
```

### Model Training Flow

```
1. Nightly trigger (Celery beat)
   └─ Start training job

2. Data preparation
   ├─ Load all ratings from PostgreSQL
   ├─ Build sparse CSR matrix
   ├─ Apply preprocessing (normalization)
   └─ Create train/val/test split

3. Hyperparameter tuning
   ├─ Optuna study (100 trials)
   ├─ Bayesian optimization
   └─ Early stopping on val RMSE

4. Model training
   ├─ SVD with optimal params
   ├─ ALS on implicit feedback
   └─ Ensemble blending

5. Evaluation
   ├─ Hit@10, NDCG@10, Coverage
   ├─ Log metrics to MLflow
   └─ Generate model card

6. Model management
   ├─ Save to S3 with version tag
   ├─ Update active model pointer
   └─ Log metadata to PostgreSQL

7. Hot reload
   ├─ Inference service loads new model
   └─ Automatic without downtime

8. Monitoring
   ├─ Compare against baseline
   ├─ Alert if performance drops
   └─ Rollback if necessary
```

## Deployment Architecture

### Local Development
```
docker-compose
├── PostgreSQL container
├── Redis container
├── FastAPI container (with auto-reload)
├── Next.js container (dev server)
├── Gradio container
├── Prometheus container
└── Grafana container
```

### Production (AWS)

```
CloudFront → ALB → ECS Fargate Services
                    ├─ Frontend (2-20 tasks)
                    ├─ Backend (2-20 tasks)
                    └─ ML (2-5 tasks)
                        ↓
                    RDS Aurora Cluster
                    (db.r6g.large, Multi-AZ)
                    ↓
                    ElastiCache Redis
                    (cache.r7g.large, 3-node cluster)
                    ↓
                    S3 buckets
                    (Models, static assets)
```

**Auto-scaling:**
- Metric: CPU utilization (target: 70%)
- Min replicas: 2
- Max replicas: 20
- Scale-up time: 1 minute
- Scale-down time: 5 minutes

## Security Architecture

### Authentication & Authorization
```
┌─ User Login (Credentials)
   │
   ├─ Hash password + salt (bcrypt)
   ├─ Verify against DB
   └─ Return JWT token

┌─ JWT Token
   ├─ Algorithm: HS256
   ├─ Expiry: 30 minutes
   ├─ Contains: user_id, username, is_admin
   └─ Signed with: SECRET_KEY (64+ chars in prod)

┌─ Protected Routes
   ├─ Extract token from Authorization header
   ├─ Verify signature + expiry
   ├─ Load user from DB
   ├─ Check permissions
   └─ Proceed or return 401/403
```

### Data Security
```
At Rest:
├─ Database: Encrypted at EBS volume level (AWS KMS)
├─ Redis: Encrypted at transit (TLS)
└─ S3: Encryption enabled (AES-256)

In Transit:
├─ HTTPS everywhere (TLS 1.3)
├─ API to API: mTLS (optional)
└─ Database: SSL connections

Access Control:
├─ Principle of least privilege (IAM)
├─ Database role with limited permissions
├─ S3 bucket policies restrict access
└─ Security groups whitelist ports
```

### Network Security
```
VPC Isolation:
├─ Public subnets: ALB, NAT Gateway
├─ Private subnets: ECS services
├─ Database subnets: RDS isolated
└─ ElastiCache: Private subnet only

Security Groups:
├─ ALB: Allow 80/443 from internet
├─ ECS: Allow from ALB only
├─ RDS: Allow from ECS only
├─ Redis: Allow from ECS only
└─ No database exposed to internet
```

## Monitoring & Observability

### Metrics Collection (Prometheus)
```
Application Metrics:
├─ API latency (histogram)
├─ Request count (counter)
├─ Error rate (counter)
├─ Cache hit rate (gauge)
├─ Model inference time (histogram)
└─ Recommendation quality (gauge)

Infrastructure Metrics:
├─ CPU utilization
├─ Memory usage
├─ Network I/O
├─ Disk usage
├─ Database connections
└─ Redis memory
```

### Dashboards (Grafana)
```
Dashboard 1: Overview
├─ API health status
├─ Request rates
├─ Error rates
├─ Latency percentiles
└─ Service availability

Dashboard 2: ML Metrics
├─ Hit@10 trend
├─ NDCG@10 trend
├─ Model inference latency
├─ Cache hit rate
└─ Model training status

Dashboard 3: Infrastructure
├─ CPU/Memory per service
├─ Database performance
├─ Redis memory usage
├─ Network I/O
└─ Disk utilization
```

### Alerting
```
High Priority (PagerDuty):
├─ API error rate > 5%
├─ API latency p95 > 1s
├─ Database unavailable
├─ Redis unavailable
└─ Hit@10 drops > 10%

Medium Priority (Email):
├─ API error rate > 1%
├─ Cache hit rate < 70%
├─ Disk usage > 80%
└─ Replica lag > 10s

Low Priority (Slack):
├─ Model training completed
├─ Unusual request patterns
└─ Scheduled maintenance
```

### Logging
```
Structured JSON Logs:
├─ Timestamp (ISO 8601)
├─ Log level (INFO/WARN/ERROR)
├─ Request ID (for tracing)
├─ User ID (when applicable)
├─ Service name
├─ Message
└─ Metadata (latency, status, etc.)

Log Retention:
├─ Real-time: CloudWatch Logs
├─ Long-term: S3 (Glacier after 90 days)
└─ Search: ElasticSearch/Splunk integration
```

## Performance Optimization

### Database Optimization
```
Indexing:
├─ user_id on ratings (100+ queries/sec)
├─ movie_id on ratings (50+ queries/sec)
├─ timestamp on events (range queries)
├─ composite indexes for joins
└─ partial indexes for active users

Query Optimization:
├─ Use connection pooling
├─ Prepared statements (prevent SQL injection)
├─ Query result caching
├─ Batch operations
└─ Denormalized views for reporting
```

### ML Model Optimization
```
Faiss Index:
├─ Index type: IVF (Inverted File)
├─ n_clusters: 100
├─ Enables sub-10ms similarity search
└─ Memory: ~200MB for 62k movies × 128 dimensions

Model Quantization:
├─ 32-bit float → 16-bit float (50% memory reduction)
├─ Accuracy impact: < 0.1%
└─ Inference speedup: ~20%

Inference Batching:
├─ Process multiple users simultaneously
├─ Reduced per-request overhead
└─ Better GPU utilization
```

### Frontend Optimization
```
Code Splitting:
├─ Route-based: Load only needed JavaScript
├─ Component lazy loading
└─ Vendor bundle optimization

Image Optimization:
├─ Format: WebP/AVIF with fallbacks
├─ Lazy loading: Load below-fold images on scroll
├─ Responsive sizes: Multiple resolutions
└─ Compression: Reduce by 70-80%

Network Optimization:
├─ Compression: gzip/brotli enabled
├─ HTTP/2 push for critical assets
├─ Async JavaScript loading
└─ CSS critical path inlining
```

## Disaster Recovery & Business Continuity

### Backup Strategy
```
Database:
├─ Automated daily snapshots (30-day retention)
├─ Point-in-time recovery enabled
├─ Replicas in multiple AZs
└─ Cross-region backup to secondary region

Models:
├─ Version control in S3
├─ Previous versions retained (last 10)
├─ Can rollback in < 5 minutes
└─ Model card and metadata preserved

Configuration:
├─ Infrastructure as Code (Terraform)
├─ Secrets in AWS Secrets Manager
├─ Configuration in parameter store
└─ Version controlled
```

### Failover Procedures
```
Scenario: Database unavailable
├─ Automatic failover to read replica (< 2 min)
├─ Read-only mode activated
├─ Recommendations served from cache
├─ Alerts sent to on-call engineer

Scenario: Cache unavailable
├─ Direct queries to database (slower)
├─ Degraded performance mode
├─ Automatic recovery when cache available

Scenario: Model inference unavailable
├─ Fallback to popularity-based recommendations
├─ Service remains available
├─ User notified of limited recommendations
```

## Scalability & Growth Planning

### Horizontal Scaling
```
Current capacity (with 10 tasks per service):
├─ 5,000 concurrent users
├─ 500 requests/second
├─ <100ms p95 latency
└─ 80%+ cache hit rate

Auto-scaling will handle:
├─ 10x traffic spike (up to 50k concurrent)
├─ 100x user base growth (1.6M → 16M users)
└─ Peak hours without performance degradation
```

### Vertical Scaling
```
Database:
├─ Current: db.r6g.large (2 vCPU, 16GB RAM)
├─ Next step: db.r6g.xlarge (4 vCPU, 32GB RAM)
└─ Path to: db.r6g.2xlarge as needed

Redis:
├─ Current: cache.r7g.large per node (2GB)
├─ Cluster can scale to 50+ nodes
└─ Memory: Add nodes before hitting 80% utilization
```

---

## Future Enhancements

- Deep learning models (Neural Collaborative Filtering)
- Real-time model updates (no retraining delay)
- Multi-model ensemble with dynamic weights
- Graph neural networks for social recommendations
- Content-based recommendations (NLP on overviews)
- Cold-start improvements (content + collaborative)
- Real-time experimentation platform (A/B testing)

---

## References

- [Collaborative Filtering Research](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [FastAPI Performance](https://fastapi.tiangolo.com/deployment/concepts/)
- [Next.js Optimization](https://nextjs.org/docs/advanced-features/measuring-performance)
- [PostgreSQL Best Practices](https://wiki.postgresql.org/wiki/Performance_Optimization)
