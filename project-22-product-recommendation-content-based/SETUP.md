# 🚀 ShopSense AI - Complete Setup Guide

**Setup time**: 15-30 minutes (first time)  
**Difficulty**: Beginner-friendly (no ML knowledge required)

---

## ✅ Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows (WSL2)
- **RAM**: 8GB minimum (the system is optimized for this)
- **Disk**: 20GB free (10GB for full dataset)
- **Docker**: 4.0+
- **Git**: 2.0+
- **Internet**: 10Mbps+ (for dataset download)

### Check Prerequisites

```bash
# Docker installed?
docker --version
# Docker version 25.0.0 or later

# Python installed?
python3 --version
# Python 3.10 or later

# Git installed?
git --version
# git version 2.0 or later
```

---

## 📥 Step 1: Clone & Setup Project

```bash
# Clone repository
git clone <repo-url> shopsense-ai
cd shopsense-ai

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies (for local data processing)
pip install -r backend/requirements.txt
pip install datasets huggingface-hub  # For dataset download
```

---

## 🗂️ Step 2: Download Dataset

### Option A: Quick Start (500 products, 30 seconds)

```bash
# Download sample dataset (great for testing)
python ml/download_dataset.py --sample-size 500
# Output: data/amazon_products_500_sample.jsonl (2MB)

# ✅ This is enough for local testing
# Proceed to Step 3
```

### Option B: Load Testing (50k products, 5 minutes)

```bash
# Download larger sample
python ml/download_dataset.py --sample-size 50000
# Output: data/amazon_products_50000_sample.jsonl (200MB)

# Use this to test auto-scaling and caching behavior
```

### Option C: Full Production (2.5M products, 2-3 hours)

```bash
# Download complete dataset (requires 10GB disk)
python ml/download_dataset.py --sample-size null
# Output: data/amazon_products_full.jsonl (8GB)

# This is for production deployment only
# Requires 16GB+ RAM to process
```

**Console Output Example:**
```
2024-07-06 10:30:00 | INFO     | ShopSense AI - Dataset Downloader
2024-07-06 10:30:01 | INFO     | Loading Amazon Reviews 2023 - Electronics category...
2024-07-06 10:30:02 | INFO     | Streaming products from Electronics...
2024-07-06 10:30:05 | INFO     | Streamed 10000 products...
2024-07-06 10:30:07 | INFO     | ✅ Wrote 500 products to data/amazon_products_500_sample.jsonl
```

**Troubleshooting:**
```bash
# Error: "datasets library not found"
pip install datasets huggingface-hub

# Error: "Connection timeout"
# → Check internet connection
# → Try manual download from Kaggle (see README.md)

# Error: "Out of memory"
# → Use smaller sample: --sample-size 1000
```

---

## 🧠 Step 3: Generate ML Artefacts

### Run Data Ingestion Pipeline

```bash
# Generate TF-IDF matrix, embeddings, and product mappings
# This takes 2-3 minutes for 500 products
python ml/data_ingestion.py \
  --jsonl data/amazon_products_500_sample.jsonl \
  --output ml/artefacts
```

**What it does:**
1. Parses JSONL file (validates with Pydantic)
2. Combines fields: title×3 + brand×2 + features×1.5 + description + category
3. Fits TF-IDF vectorizer (50,000 vocab)
4. Generates embeddings (all-MiniLM-L6-v2, 384-dim)
5. Saves artefacts to `ml/artefacts/`:
   - `tfidf_vectorizer.pkl` (10MB)
   - `tfidf_matrix.npz` (sparse matrix, 50MB for 500 products)
   - `embeddings.npy` (100MB for 500 products)
   - `product_id_map.pkl` (50KB)

**Expected Output:**
```
2024-07-06 10:35:00 | INFO     | Starting ingestion pipeline...
2024-07-06 10:35:01 | INFO     | Step 1: Parsing and validating products...
2024-07-06 10:35:02 | INFO     | Loaded 500 valid products
2024-07-06 10:35:03 | INFO     | Step 2: Fitting TF-IDF vectorizer...
2024-07-06 10:35:04 | INFO     | TF-IDF matrix shape: (500, 50000) (density: 0.28%)
2024-07-06 10:35:05 | INFO     | Step 3: Generating embeddings...
2024-07-06 10:35:35 | INFO     | Generated embeddings shape: (500, 384)
2024-07-06 10:35:36 | INFO     | Step 4: Saving artefacts...
2024-07-06 10:35:37 | INFO     | Pipeline complete in 37.5s
```

**Verify Artefacts:**
```bash
ls -lh ml/artefacts/
# total 150M
# 10M  tfidf_vectorizer.pkl
# 50M  tfidf_matrix.npz
# 100M embeddings.npy
# 0.5M product_id_map.pkl
```

**Troubleshooting:**
```bash
# Error: "Out of memory during embedding generation"
# → Reduce batch_size in data_ingestion.py (line 330)
#   from batch_size=256 to batch_size=128

# Error: "CUDA out of memory" (if GPU)
# → Don't worry, falls back to CPU
# → Or disable GPU: export CUDA_VISIBLE_DEVICES=""

# Error: "embeddings.npy not found"
# → Check ml/artefacts/ directory
# → Re-run data_ingestion.py with fresh --jsonl file
```

---

## 🐳 Step 4: Start Docker Services

### Build & Run

```bash
# Start all 6 services (MongoDB, Redis, Elasticsearch, Django, Streamlit, Nginx)
docker compose up --build

# This will:
# 1. Build Docker images (~5 min first time)
# 2. Pull base images
# 3. Start containers
# 4. Run health checks
# 5. Wait for all services to be healthy
```

**Expected Console Output:**
```
[+] Building 5.2s (25/25) FINISHED
[+] Running 6/6
  ✓ mongodb
  ✓ redis
  ✓ elasticsearch
  ✓ django_api
  ✓ streamlit_dashboard
  ✓ nginx
```

### Verify Services

```bash
# In a new terminal:
docker compose ps

# Output:
# NAME                    STATUS
# shopsense_mongodb       Up (healthy)
# shopsense_redis         Up (healthy)
# shopsense_elasticsearch Up (healthy)
# shopsense_django        Up (healthy)
# shopsense_streamlit     Up (healthy)
# shopsense_nginx         Up (healthy)
```

### Health Check

```bash
# Test API health endpoint
curl http://localhost:8000/api/v1/health/

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "2024-07-06T10:40:00Z",
#   "services": {
#     "ml_engine": {
#       "status": "loaded",
#       "products": 500
#     },
#     "redis": "connected"
#   }
# }
```

**Troubleshooting:**
```bash
# Error: "Port 8000 already in use"
docker compose down -v
# Then: docker compose up --build

# Error: "Out of memory"
# → Check Docker desktop settings
# → Increase memory limit to 8GB+
# → Or use smaller dataset (--sample-size 100)

# Service won't start
# → Check logs: docker compose logs django_api
# → Common: Waiting for MongoDB (retry logic handles this)
```

---

## 🌐 Step 5: Access Applications

Open your browser and visit:

### 1. **Streamlit Dashboard** (Recommended starting point)

```
http://localhost:8501
```

**Features:**
- ✅ Live similarity explorer (search product → see 8 similar items)
- ✅ Similarity score visualization
- ✅ System metrics & monitoring
- ✅ No ML knowledge required

**Quick Test:**
1. Open dashboard
2. In sidebar, select method = "hybrid"
3. Enter product ID: `B08N5WRWNW` (Sony headphones)
4. Click "Find Similar"
5. See top-8 similar products with scores!

### 2. **Django REST API** (For developers)

```
http://localhost:8000/api/v1/
```

**Key Endpoints:**
- `GET /api/v1/health/` - Service status
- `GET /api/v1/stats/` - Engine statistics
- `GET /api/v1/products/{id}/similar/?k=8&method=hybrid` - Main endpoint
- `POST /api/v1/batch-similar/` - Batch processing

**Quick Test:**
```bash
curl "http://localhost:8000/api/v1/products/B08N5WRWNW/similar/?k=8"
```

### 3. **Swagger API Docs** (For integration)

```
http://localhost:8000/api/schema/swagger-ui/
```

**Use this to:**
- Understand request/response formats
- Try endpoints interactively
- Generate client code

### 4. **MongoDB Express** (Database visualization)

```
http://localhost:8081
```

**Note:** Only available if you uncomment in docker-compose.yml

---

## 🧪 Step 6: Run Tests (Optional)

```bash
# Test similarity engine
python -m pytest tests/test_similarity_engine.py -v

# Test API endpoints
python -m pytest tests/test_api.py -v

# Full test suite with coverage
python -m pytest tests/ --cov=. --cov-report=html
# Open htmlcov/index.html in browser
```

---

## 🔍 Step 7: Explore the System

### A. Test Different Methods

```bash
# TF-IDF only (fastest, keyword-based)
curl "http://localhost:8000/api/v1/products/B08N5WRWNW/similar/?method=tfidf_only"

# Embedding only (slowest, semantic)
curl "http://localhost:8000/api/v1/products/B08N5WRWNW/similar/?method=embedding_only"

# Hybrid (balanced, recommended)
curl "http://localhost:8000/api/v1/products/B08N5WRWNW/similar/?method=hybrid"
```

**Notice:**
- TF-IDF: Fast, keyword matches
- Embedding: Slow, semantic similarity
- Hybrid: Both, best results

### B. Check Cache Performance

```bash
# First request (cache MISS)
time curl "http://localhost:8000/api/v1/products/B08N5WRWNW/similar/"
# real: 0m0.142s (142ms)

# Second request (cache HIT)
time curl "http://localhost:8000/api/v1/products/B08N5WRWNW/similar/"
# real: 0m0.015s (15ms) ← 10x faster!
```

### C. Monitor with Logs

```bash
# Watch Django logs
docker compose logs -f django_api

# Watch all services
docker compose logs -f

# Follow specific pattern
docker compose logs -f | grep "Similar products"
```

---

## 🛑 Step 8: Shutdown

```bash
# Stop all services (keep data)
docker compose stop

# Stop and remove containers (keep data)
docker compose down

# Stop and remove everything (wipe data)
docker compose down -v
```

---

## 🚀 What's Next?

### For Learning
1. Read `ARCHITECTURE.md` to understand design
2. Study `ml/similarity_engine.py` (the core algorithm)
3. Trace a request: API call → similarity computation → response

### For Production
1. Download full dataset (2.5M products)
2. Configure environment variables (.env.prod)
3. Deploy to AWS ECS (see `docs/DEPLOYMENT.md`)

### For Extension
1. Add personalization (track user clicks)
2. Add visual similarity (CLIP embeddings)
3. A/B test different methods
4. Implement real-time updates

---

## 📊 Benchmark Your System

```bash
# Measure latency on your machine
python3 <<'EOF'
import time
import requests

url = "http://localhost:8000/api/v1/products/B08N5WRWNW/similar/"

# Warm up
requests.get(url)

# Benchmark 10 requests
times = []
for i in range(10):
    start = time.time()
    resp = requests.get(url)
    times.append((time.time() - start) * 1000)
    
import statistics
print(f"Min: {min(times):.1f}ms")
print(f"Avg: {statistics.mean(times):.1f}ms")
print(f"Max: {max(times):.1f}ms")
print(f"Cache hit: {resp.json().get('cache_hit')}")
EOF
```

**Expected Results:**
- First 1-2 requests: 150ms (cache miss)
- Remaining requests: 15ms (cache hit)

---

## ❓ FAQ

### Q: What if I get "Out of Memory"?
**A:** Use smaller dataset: `--sample-size 100` instead of 500

### Q: Can I use different product data?
**A:** Yes! Replace `amazon_products.jsonl` with your own JSONL file with same schema

### Q: How do I update products?
**A:** Re-run data_ingestion.py with new JSONL file, restart Docker

### Q: Can I run on GPU?
**A:** Yes! Embedding generation will auto-detect CUDA. Install `torch[cuda]`

### Q: How do I deploy to production?
**A:** See `docs/DEPLOYMENT.md` for AWS ECS guide

### Q: Is this suitable for my use case?
**A:** ShopSense is best for: <10M products, <50 categories, content-based matching
**Not suitable for:** Collaborative filtering (needs user history), deep personalization

---

## 📞 Support

**Something broken?**
1. Check logs: `docker compose logs -f <service>`
2. Review troubleshooting sections above
3. Check GitHub issues: [repo-url]/issues
4. Ask on Discussions: [repo-url]/discussions

**Want to contribute?**
1. Fork repository
2. Create feature branch
3. Submit PR with tests
4. Follow code style: `ruff check .`

---

## ✅ Checklist

- [ ] Prerequisites installed (Docker, Python, Git)
- [ ] Project cloned
- [ ] Dataset downloaded
- [ ] ML artefacts generated
- [ ] Docker services running
- [ ] Dashboard accessible (http://localhost:8501)
- [ ] API health check passing
- [ ] Test endpoint working

**If all ✅, you're ready to use ShopSense AI!**

---

**Built for developers who value clarity and sustainability.** 🚀
