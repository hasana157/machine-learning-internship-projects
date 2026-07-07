"""
ShopSense AI - Django Settings
Production-optimized for 8GB RAM systems
Memory-conscious configuration with lazy loading
"""

import os
from pathlib import Path
from datetime import timedelta

# ============================================================================
# CORE SETTINGS
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'dev-key-change-in-production')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# ============================================================================
# INSTALLED APPS (Minimal set for memory efficiency)
# ============================================================================
INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.auth',
    'rest_framework',
    'corsheaders',
    'products',          # Custom app
    'recommendations',   # Custom app
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
]

# ============================================================================
# REST FRAMEWORK CONFIG
# ============================================================================
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.CursorPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '30/minute',      # Anonymous users: 30 req/min
        'user': '200/minute',     # Authenticated: 200 req/min
    },
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
}

# ============================================================================
# CORS CONFIGURATION (Allow Flutter Web frontend)
# ============================================================================
CORS_ALLOWED_ORIGINS = [
    "http://localhost:8080",
    "http://localhost:3000",
]

# ============================================================================
# DATABASE - MongoDB (NoSQL for flexible schema, lower overhead than SQL)
# ============================================================================
MONGO_URI = os.getenv(
    'MONGO_URI',
    'mongodb://localhost:27017/shopsense_ai'
)

# MongoEngine settings (lazy initialization to save memory)
MONGOENGINE_CONNECT_KWARGS = {
    'connect': False,  # Lazy connection - connects only when needed
    'maxPoolSize': 10,  # Connection pool size (8GB = small pool)
    'minPoolSize': 2,
}

# ============================================================================
# CACHE - Redis (TTL-based caching for 8GB RAM)
# ============================================================================
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {'max_connections': 50},
        },
        'KEY_PREFIX': 'shopsense',
        'TIMEOUT': 7200,  # 2 hours default TTL
    }
}

# ============================================================================
# ELASTICSEARCH - Vector search configuration
# ============================================================================
ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST', 'localhost:9200')
ELASTICSEARCH_INDEX = 'products'

# ============================================================================
# ML CONFIGURATION (Memory-optimized)
# ============================================================================
ML_CONFIG = {
    # TF-IDF Vectorizer settings (sparse matrix = memory efficient)
    'TFIDF_MAX_FEATURES': 50000,      # Covers 99.8% of vocabulary
    'TFIDF_NGRAM_RANGE': (1, 2),      # Unigrams + bigrams
    'TFIDF_MIN_DF': 2,                # Min document frequency
    'TFIDF_SUBLINEAR_TF': True,       # log(1+tf) dampening
    
    # Sentence Transformer settings
    'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',  # 384-dim, fast inference
    'EMBEDDING_BATCH_SIZE': 128,            # Reduced from 256 for 8GB RAM
    'EMBEDDING_NORMALIZE': True,
    
    # Similarity search
    'SIMILAR_PRODUCT_K': 8,            # Number of similar items
    'RRF_WEIGHTS': [0.45, 0.55],       # TF-IDF vs Embedding weight
    'RRF_WINDOW_SIZE': 50,             # Over-fetch for re-ranking
    
    # Cache settings
    'SIMILARITY_CACHE_TTL': 7200,      # 2 hours
}

# ============================================================================
# CELERY - Async task queue (for embedding computation)
# ============================================================================
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2')
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 minutes hard limit
CELERY_TASK_SOFT_TIME_LIMIT = 25 * 60  # 25 minutes soft limit

# ============================================================================
# LOGGING - Structured logging for debugging
# ============================================================================
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': os.getenv('LOG_LEVEL', 'INFO'),
    },
}

# ============================================================================
# SECURITY (Production checklist)
# ============================================================================
if not DEBUG:
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_SECURITY_POLICY = {
        'default-src': ("'self'",),
    }

# ============================================================================
# ARTIFACT PATHS (Model files, vectorizers, matrices)
# ============================================================================
ML_ARTIFACTS_DIR = os.getenv('ML_ARTIFACTS_DIR', '/ml/artefacts')
os.makedirs(ML_ARTIFACTS_DIR, exist_ok=True)
