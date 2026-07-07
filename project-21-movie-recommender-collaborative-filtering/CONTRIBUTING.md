# Contributing to CineMatch AI

Thank you for interest in contributing! This document provides guidelines for development.

## Code of Conduct

- Be respectful and professional
- Give credit where due
- Report issues responsibly

## Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Git

### Initial Setup

```bash
# Clone repository
git clone <repo-url>
cd CineMatch

# Setup environment
cp .env.example .env

# Start development stack
docker-compose up -d

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Development Workflow

### 1. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

### Branch Naming Convention
- Features: `feature/feature-name`
- Bugs: `fix/bug-description`
- Chores: `chore/task-description`
- Docs: `docs/documentation-update`

### 2. Make Changes

#### Python Code Style

```python
# Use Black for formatting
black backend/ ml/

# Use Ruff for linting
ruff check backend/ ml/

# Use Mypy for type checking
mypy backend/ ml/
```

Example:
```python
"""Module docstring explaining purpose"""

from typing import Optional, List
from fastapi import APIRouter, Depends

router = APIRouter()


async def get_user(user_id: int) -> Optional[dict]:
    """Get user by ID.
    
    Args:
        user_id: User identifier
        
    Returns:
        User data or None if not found
    """
    # Implementation
    pass
```

#### JavaScript/TypeScript Code Style

```bash
# Use Prettier for formatting
prettier --write ./frontend

# Use ESLint for linting
eslint ./frontend --fix
```

Example:
```typescript
// File: components/MovieCard.tsx
import { FC, ReactNode } from 'react';
import Link from 'next/link';

interface MovieCardProps {
  id: number;
  title: string;
  posterUrl?: string;
  rating?: number;
}

/**
 * Movie card component displaying movie information.
 * @component
 */
const MovieCard: FC<MovieCardProps> = ({
  id,
  title,
  posterUrl,
  rating,
}) => {
  return (
    <Link href={`/movies/${id}`}>
      <div className="cursor-pointer rounded-lg overflow-hidden">
        {/* Component content */}
      </div>
    </Link>
  );
};

export default MovieCard;
```

### 3. Write Tests

#### Backend Tests

```bash
# Run tests
cd backend
pytest -v

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/api/test_recommendations.py::test_get_recommendations
```

Example test:
```python
# tests/api/test_recommendations.py
import pytest
from httpx import AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_get_recommendations():
    """Test recommendations endpoint returns valid response."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/recommendations/123",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
```

#### Frontend Tests

```bash
# Run tests
cd frontend
npm test

# Run with coverage
npm test -- --coverage

# Watch mode
npm test -- --watch
```

Example test:
```typescript
// components/__tests__/MovieCard.test.tsx
import { render, screen } from '@testing-library/react';
import MovieCard from '../MovieCard';

describe('MovieCard', () => {
  it('renders movie title', () => {
    render(
      <MovieCard
        id={1}
        title="Test Movie"
        rating={4.5}
      />
    );
    
    expect(screen.getByText('Test Movie')).toBeInTheDocument();
  });
});
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with clear message
git commit -m "feature: add user recommendations endpoint

- Implement SVD-based recommendation generation
- Add Redis caching for 1-hour TTL
- Include Hit@K evaluation metrics
- Support multiple recommendation strategies

Fixes #123"
```

#### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Test additions/updates
- `chore`: Maintenance tasks

### 5. Push & Create Pull Request

```bash
# Push to remote
git push origin feature/your-feature-name

# Create PR on GitHub with:
# - Clear title and description
# - Reference to related issues
# - Checklist of changes
```

## Code Review Guidelines

### For Authors

- Ensure all tests pass locally
- Check code coverage doesn't decrease
- Keep PRs focused and reasonably sized (<400 lines ideally)
- Respond to feedback professionally
- Update PR based on reviews

### For Reviewers

- Check functionality and test coverage
- Look for code style and best practices
- Verify documentation is updated
- Be constructive and respectful
- Approve or request changes clearly

## Testing Checklist

Before submitting PR:

- [ ] All new code has tests
- [ ] All tests pass locally
- [ ] Code is properly formatted (Black/Prettier)
- [ ] No linting errors (Ruff/ESLint)
- [ ] Type checks pass (Mypy/TSC)
- [ ] Documentation updated
- [ ] No console warnings
- [ ] Performance acceptable

## Documentation

### Python Docstrings

```python
def calculate_recommendations(user_id: int, k: int = 10) -> List[Recommendation]:
    """
    Generate personalized movie recommendations for a user.
    
    Uses SVD-based collaborative filtering with Redis caching
    to provide fast, personalized recommendations.
    
    Args:
        user_id: The ID of the user
        k: Number of recommendations to return (default: 10)
        
    Returns:
        List of Recommendation objects sorted by relevance
        
    Raises:
        UserNotFoundError: If user doesn't exist
        InvalidKError: If k is not between 1 and 50
        
    Example:
        >>> recs = calculate_recommendations(user_id=123, k=10)
        >>> print(f"Found {len(recs)} recommendations")
    """
```

### TypeScript Comments

```typescript
/**
 * Fetches recommendations from the API.
 * Includes automatic caching and error handling.
 *
 * @param userId - The user ID to get recommendations for
 * @param k - Number of recommendations (default: 10)
 * @returns Promise resolving to array of recommendations
 * @throws {ApiError} If API call fails
 *
 * @example
 * const recommendations = await getRecommendations(123, 10);
 */
async function getRecommendations(
  userId: number,
  k: number = 10
): Promise<Recommendation[]> {
  // Implementation
}
```

## Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Documentation update

## Related Issues
Closes #123

## Changes
- Change 1
- Change 2

## Testing
- [ ] Tested locally
- [ ] Added unit tests
- [ ] All tests pass

## Checklist
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No breaking changes
```

## Release Process

### Version Numbering
Uses Semantic Versioning (MAJOR.MINOR.PATCH)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist
1. Update version in `__version__.py`
2. Update CHANGELOG.md
3. Create annotated tag: `git tag -a v1.0.0 -m "Release v1.0.0"`
4. Push tag: `git push origin v1.0.0`
5. Create GitHub release with notes

## Performance Guidelines

### Backend
- API endpoints should respond in <250ms (p95)
- Database queries should complete in <50ms
- Cache hit rate should exceed 80%
- Maintain Hit@10 >= 0.65

### Frontend
- Lighthouse score >90
- Initial page load <3s
- Time to interactive <5s
- Component render <16ms (60fps)

## Security

### Reporting Security Issues

**DO NOT** open public issues for security vulnerabilities.

Instead, email security@cinematch.ai with:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if applicable)

We'll:
1. Acknowledge receipt within 48 hours
2. Investigate and develop fix
3. Coordinate disclosure timeline
4. Credit you in advisory (if desired)

### Code Security Standards

- Never commit secrets (API keys, passwords)
- Use parameterized queries for SQL
- Validate and sanitize all inputs
- Use HTTPS in production
- Keep dependencies updated
- Follow OWASP guidelines

## Getting Help

- **Discussions**: GitHub Discussions
- **Issues**: GitHub Issues for bugs
- **Slack**: CineMatch development channel
- **Email**: dev@cinematch.ai

## Resources

- [FastAPI Best Practices](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Python PEP 8](https://pep8.org/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)

## Thank You!

Your contributions help make CineMatch AI better for everyone. We appreciate your time and effort!
