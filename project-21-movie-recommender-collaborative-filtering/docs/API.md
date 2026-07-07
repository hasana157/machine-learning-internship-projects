# CineMatch AI - API Specification

## Overview

**Base URL**: `http://localhost:8000` (development) | `https://api.cinematch.ai` (production)

**API Version**: `v1.0.0`

**Authentication**: JWT Bearer Token

**Rate Limit**: 100 requests/minute per user

**Response Format**: JSON

---

## Authentication

### Login Endpoint

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password123"
}
```

**Response (201 Created):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Register Endpoint

```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "newuser",
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "full_name": "John Doe"
}
```

---

## Recommendations API

### Get User Recommendations

```http
GET /api/v1/recommendations/{user_id}?k=10&strategy=ensemble
Authorization: Bearer {token}
```

**Query Parameters:**
- `k` (integer, default: 10): Number of recommendations (1-50)
- `strategy` (string, default: "ensemble"): Recommendation strategy
  - `svd`: Matrix factorization
  - `knn`: K-nearest neighbors
  - `ensemble`: Weighted blend (SVD + ALS)
  - `popularity`: Popular movies (cold-start)

**Response (200 OK):**
```json
{
  "user_id": 123,
  "strategy": "ensemble",
  "generated_at": "2024-11-01T10:30:00Z",
  "cache_hit": false,
  "latency_ms": 187,
  "recommendations": [
    {
      "rank": 1,
      "movie_id": 318,
      "title": "The Shawshank Redemption",
      "genres": ["Drama"],
      "year": 1994,
      "poster_url": "https://image.tmdb.org/...",
      "predicted_score": 4.87,
      "match_percent": 97,
      "explanation": "Because you rated The Green Mile 5.0 and Se7en 4.5"
    }
  ]
}
```

**HTTP Status Codes:**
- `200`: Success (cached or fresh)
- `401`: Unauthorized (missing/invalid token)
- `404`: User not found
- `500`: Server error

---

## Movies API

### List Movies

```http
GET /api/v1/movies?skip=0&limit=20&sort_by=popularity
```

**Query Parameters:**
- `skip` (integer, default: 0): Number to skip for pagination
- `limit` (integer, default: 20, max: 100): Items per page
- `sort_by` (string): popularity, rating, title, year

**Response (200 OK):**
```json
{
  "items": [
    {
      "id": 318,
      "title": "The Shawshank Redemption",
      "year": 1994,
      "poster_url": "https://...",
      "rating": 8.7,
      "genres": ["Drama"],
      "popularity": 85.5
    }
  ],
  "skip": 0,
  "limit": 20,
  "total": 62423
}
```

### Search Movies

```http
GET /api/v1/movies/search?q=inception&limit=10
```

**Query Parameters:**
- `q` (string, required): Search query
- `limit` (integer, default: 20, max: 100): Max results

**Response (200 OK):**
```json
{
  "query": "inception",
  "count": 3,
  "results": [
    {
      "id": 27205,
      "title": "Inception",
      "year": 2010,
      "poster_url": "https://...",
      "rating": 8.8,
      "genres": ["Action", "Sci-Fi", "Thriller"]
    }
  ]
}
```

### Get Movie Details

```http
GET /api/v1/movies/{movie_id}
```

**Response (200 OK):**
```json
{
  "id": 318,
  "title": "The Shawshank Redemption",
  "year": 1994,
  "overview": "Two imprisoned men bond over...",
  "poster_url": "https://...",
  "runtime": 142,
  "genres": ["Drama"],
  "rating": 8.7,
  "popularity": 85.5,
  "ratings_count": 15234,
  "average_rating": 4.82
}
```

---

## Events API

### Log User Event

```http
POST /api/v1/events
Authorization: Bearer {token}
Content-Type: application/json

{
  "event_type": "click",
  "movie_id": 318,
  "duration_seconds": null,
  "metadata": {
    "source": "recommendation_feed",
    "position": 1
  }
}
```

**Event Types:**
- `click`: User clicked on recommendation
- `view`: User viewed movie details
- `watch`: User watched movie
- `add_to_list`: Added to watchlist
- `remove_from_list`: Removed from watchlist

**Response (201 Created):**
```json
{
  "id": 1,
  "user_id": 123,
  "movie_id": 318,
  "event_type": "click",
  "timestamp": "2024-11-01T10:30:00Z"
}
```

### Rate Movie

```http
POST /api/v1/events/rate
Authorization: Bearer {token}
Content-Type: application/json

{
  "movie_id": 318,
  "rating": 4.5
}
```

**Response (200 OK):**
```json
{
  "user_id": 123,
  "movie_id": 318,
  "rating": 4.5,
  "timestamp": "2024-11-01T10:35:00Z",
  "message": "Rating submitted successfully"
}
```

### Get User Ratings

```http
GET /api/v1/events/ratings?limit=50&offset=0
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "user_id": 123,
  "total": 42,
  "offset": 0,
  "limit": 50,
  "ratings": [
    {
      "movie_id": 318,
      "movie_title": "The Shawshank Redemption",
      "rating": 4.5,
      "timestamp": "2024-11-01T10:35:00Z"
    }
  ]
}
```

---

## Health Check Endpoints

### Basic Health Check

```http
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy"
}
```

### Readiness Probe

```http
GET /health/ready
```

**Response (200 OK):**
```json
{
  "status": "ready",
  "database": "healthy",
  "redis": "healthy"
}
```

### Detailed Health

```http
GET /health/detailed
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "database": {
    "status": "healthy",
    "version": "PostgreSQL 16.0"
  },
  "redis": {
    "status": "healthy",
    "uptime_seconds": 3600,
    "connected_clients": 5,
    "used_memory": "2.5M"
  }
}
```

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Invalid request parameters"
}
```

### 401 Unauthorized

```json
{
  "detail": "Could not validate credentials"
}
```

### 403 Forbidden

```json
{
  "detail": "Insufficient permissions"
}
```

### 404 Not Found

```json
{
  "detail": "Resource not found"
}
```

### 422 Unprocessable Entity

```json
{
  "detail": "Validation error",
  "errors": [
    {
      "loc": ["body", "rating"],
      "msg": "Value must be between 0.5 and 5.0",
      "type": "value_error"
    }
  ]
}
```

### 429 Too Many Requests

```json
{
  "detail": "Rate limit exceeded (100 requests/minute)"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Internal server error"
}
```

---

## Rate Limiting

**Limits per minute:**
- Authenticated users: 100 requests/minute
- Anonymous: 20 requests/minute

**Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1635591600
```

---

## Pagination

All list endpoints support cursor/offset-based pagination:

```http
GET /api/v1/movies?skip=20&limit=10
```

**Response:**
```json
{
  "items": [...],
  "skip": 20,
  "limit": 10,
  "total": 62423
}
```

---

## Filtering & Sorting

### Sorting

```http
GET /api/v1/movies?sort_by=popularity
GET /api/v1/movies?sort_by=rating
GET /api/v1/movies?sort_by=title
```

### Filtering

```http
GET /api/v1/movies/search?q=action&limit=20
```

---

## Performance SLAs

| Endpoint | P50 | P95 | P99 | Target |
|----------|-----|-----|-----|--------|
| GET /recommendations (cache hit) | 8ms | 20ms | 35ms | <20ms |
| GET /recommendations (cache miss) | 120ms | 240ms | 400ms | <250ms |
| GET /similar-items | 15ms | 40ms | 80ms | <40ms |
| POST /events | 5ms | 15ms | 30ms | <15ms |
| GET /movies | 50ms | 100ms | 200ms | <100ms |

---

## Versioning

Current API version is `v1`. Future versions will be available at `/api/v2`, etc.

Backward compatibility will be maintained within major versions.

---

## Documentation

- **Interactive Docs**: `/docs` (Swagger UI)
- **ReDoc**: `/redoc` (ReDoc documentation)
- **OpenAPI Schema**: `/openapi.json`

---

## Changelog

### v1.0.0 (2024-11-01)
- Initial release
- Recommendations, Movies, Events endpoints
- JWT authentication
- Redis caching
