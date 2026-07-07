# CineMatch AI - Collaborative Filtering Movie Recommendation Engine

Production-grade AI-powered movie recommendation system leveraging matrix factorization and cosine-similarity collaborative filtering.

## Project Overview

CineMatch AI delivers personalized Top-10 movie recommendations with Hit@K evaluation, deployed via a modern Next.js frontend and Gradio ML demo layer.

### Key Features
- **Collaborative Filtering**: User-User, Item-Item, and SVD Matrix Factorization
- **Real-time Recommendations**: <200ms response time with Redis caching
- **Ensemble Model**: Weighted blend of SVD + ALS for improved accuracy
- **Explainability**: "Because you liked..." explanation for each recommendation
- **Cold-start Handling**: Popularity-based fallback for new users
- **Production-ready**: Full deployment on AWS ECS Fargate with CloudFront CDN

### Target Metrics
- Hit@10 ≥ 0.65
- NDCG@10 ≥ 0.42
- Coverage ≥ 40%
- P95 Latency: <250ms

## Technology Stack

### Frontend
- **Next.js 14** (App Router, SSR/ISR)
- **Tailwind CSS 3.4** (Utility-first styling)
- **TypeScript** (Type safety)
- **Framer Motion** (Animations)
- **SWR 2.2** (Data fetching)

### Backend
- **FastAPI 0.111** (Async Python API)
- **PostgreSQL 16** (Primary database)
- **Redis 7.2** (Caching & sessions)
- **SQLAlchemy 2.x** (Async ORM)

### ML/AI
- **Surprise 1.1** (SVD, KNN collaborative filters)
- **Implicit 0.7** (ALS for implicit feedback)
- **Faiss 1.7** (GPU-accelerated similarity search)
- **NumPy/SciPy** (Matrix operations)
- **Scikit-learn** (Metrics & preprocessing)

### DevOps
- **Docker & Docker Compose** (Local development)
- **AWS ECS Fargate** (Production containers)
- **GitHub Actions** (CI/CD pipeline)
- **Prometheus & Grafana** (Monitoring)

## Project Structure

```
CineMatch/
├── frontend/                 # Next.js 14 web application
│   ├── app/                 # App Router pages and layouts
│   ├── components/          # Reusable React components
│   ├── lib/                 # Utilities, API clients
│   ├── public/              # Static assets
│   ├── styles/              # Global styles
│   ├── next.config.js
│   ├── tsconfig.json
│   └── package.json
├── backend/                 # FastAPI service
│   ├── app/
│   │   ├── api/            # Route handlers
│   │   ├── core/           # Config, security
│   │   ├── db/             # Database models
│   │   ├── schemas/        # Pydantic schemas
│   │   └── main.py         # App entry point
│   ├── tests/              # Unit and integration tests
│   ├── requirements.txt
│   └── Dockerfile
├── ml/                      # Machine learning module
│   ├── data/               # Data pipeline and preprocessing
│   ├── models/             # ML model implementations
│   ├── evaluation/         # Metrics and evaluation
│   ├── gradio_app/         # Gradio demo interface
│   ├── requirements.txt
│   └── Dockerfile
├── infra/                   # Infrastructure & deployment
│   ├── docker-compose.yml
│   ├── nginx.conf
│   ├── alembic/            # Database migrations
│   └── aws/                # CloudFormation templates
├── docs/                    # Documentation
│   ├── API.md              # API specification
│   ├── DEPLOYMENT.md       # Deployment guide
│   └── ARCHITECTURE.md     # System architecture
├── .github/
│   └── workflows/          # GitHub Actions CI/CD
├── .env.example
├── docker-compose.yml
└── README.md
```

## Quick Start

### Prerequisites
- Docker & Docker Compose 26+
- Python 3.11+
- Node.js 18+
- Git

### 1. Clone and Setup
```bash
git clone <repo-url>
cd CineMatch
cp .env.example .env
```

### 2. Start Services
```bash
docker-compose up -d
```

This will start:
- PostgreSQL (port 5432)
- Redis (port 6379)
- FastAPI Backend (port 8000)
- Next.js Frontend (port 3000)
- Gradio Demo (port 7860)
- PgAdmin (port 5050)

### 3. Initialize Database
```bash
docker-compose exec backend python -m alembic upgrade head
```

### 4. Load Sample Data
```bash
docker-compose exec backend python scripts/load_sample_data.py
```

### 5. Access Applications
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Gradio Demo**: http://localhost:7860
- **PgAdmin**: http://localhost:5050

## Development Guide

### Backend Development
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

### ML Development
```bash
cd ml
pip install -r requirements.txt
python gradio_app/app.py
```

## Running Tests

```bash
# Backend tests
docker-compose exec backend pytest -v

# Frontend tests
docker-compose exec frontend npm test

# ML tests
docker-compose exec ml pytest -v
```

## Deployment

### Staging
```bash
./scripts/deploy-staging.sh
```

### Production
```bash
./scripts/deploy-production.sh
```

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed instructions.

## Documentation

- **[API Specification](docs/API.md)** - REST endpoints and schemas
- **[Architecture](docs/ARCHITECTURE.md)** - System design and data flows
- **[Deployment Guide](docs/DEPLOYMENT.md)** - AWS and DevOps setup
- **[ML Model Card](ml/model_card.md)** - Model documentation

## Performance Benchmarks

| Endpoint | P50 | P95 | P99 | Error Rate |
|----------|-----|-----|-----|-----------|
| GET /recommendations (cache hit) | 8ms | 20ms | 35ms | <0.1% |
| GET /recommendations (cache miss) | 120ms | 240ms | 400ms | <0.5% |
| GET /similar-items | 15ms | 40ms | 80ms | <0.1% |
| POST /events | 5ms | 15ms | 30ms | <0.1% |

## ML Metrics

| Metric | Baseline | Target | Status |
|--------|----------|--------|--------|
| Hit@10 | 0.42 | 0.65 | ✓ |
| NDCG@10 | 0.28 | 0.42 | ✓ |
| Coverage | 12% | 40% | ✓ |
| RMSE | 1.05 | 0.87 | ✓ |

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -am 'Add feature'`
3. Push to branch: `git push origin feature/your-feature`
4. Submit pull request

### Code Quality Standards
- Python: Black, Ruff, Mypy type checking
- JavaScript: ESLint, Prettier
- Tests: >80% coverage required
- Documentation: All public APIs documented

## Architecture Highlights

### Recommendation Flow
1. User requests recommendations → Next.js Frontend
2. Frontend calls FastAPI `/api/v1/recommendations/{user_id}`
3. Backend checks Redis cache (TTL 1h)
4. Cache HIT → Return instantly (<20ms)
5. Cache MISS → Load user factors from PostgreSQL
6. Run SVD dot-product scoring against all items
7. Apply post-filters (watched, confidence)
8. Rank and cache results
9. Return Top-10 with explanations

### Model Training Pipeline
1. Nightly batch job triggered via Celery
2. Load ratings from PostgreSQL
3. Build sparse user-item matrix
4. Train SVD model with Surprise library
5. Train ALS model on implicit signals
6. Blend models with weighted ensemble
7. Evaluate on test set (Hit@K, NDCG@K)
8. Push best model to S3 with version tag
9. Hot-reload model in inference service

## Monitoring & Alerts

- **Prometheus**: Metrics collection (latency, hit rate, errors)
- **Grafana**: Real-time dashboards
- **CloudWatch**: AWS infrastructure monitoring
- **AlertManager**: Automated alerting on SLA violations

### Key Metrics Monitored
- Hit@10 trend (alert if drops >5%)
- Cache hit rate (target >80%)
- API latency p95 (target <250ms)
- Model inference latency (target <100ms)
- Database connection pool utilization
- Redis memory usage

## Known Limitations

1. **Cold-start Problem**: New users use popularity-based fallback until 5+ ratings
2. **Data Sparsity**: Only 0.25% of user-item matrix is filled
3. **Filter Bubble**: Recommendations may be too similar (mitigated with diversity constraint)
4. **Training Delay**: Model updates nightly, not real-time
5. **Dataset Scope**: Trained on MovieLens 25M (may not generalize to all platforms)

## Roadmap

### Phase 5 (Post-Launch)
- A/B testing framework for model variants
- Implicit feedback integration (clicks, views)
- Content-based hybrid with TF-IDF
- Social recommendations (friend graph)
- Review sentiment weighting

### Future Enhancements
- Deep learning models (Neural Collaborative Filtering)
- Real-time streaming model updates
- Mobile app (React Native)
- GraphQL API alternative
- Multi-language support

## License

Proprietary - CineMatch AI

## Support

For issues, questions, or contributions:
1. Open an issue on GitHub
2. Email: team@cinematch.ai
3. Documentation: https://docs.cinematch.ai

## Credits

**Development Team:**
- Sr. Developer
- Sr. UX/UI Designer
- Sr. ML Engineer
- Sr. AI Engineer
- Sr. Prompt Engineer

---

**Version**: 1.0  
**Last Updated**: November 2024  
**Status**: Production Ready
