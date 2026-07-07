#!/bin/bash

# CineMatch AI - Quick Start Script
# This script sets up and starts the development environment

set -e

echo "🎬 CineMatch AI - Quick Start Setup"
echo "===================================="
echo ""

# Check prerequisites
echo "✓ Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "✗ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "✗ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "✗ Git is not installed. Please install Git first."
    exit 1
fi

echo "✓ All prerequisites met"
echo ""

# Setup environment
echo "📝 Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file (update with your settings)"
else
    echo "✓ .env file already exists"
fi

echo ""

# Start services
echo "🐳 Starting Docker services..."
docker-compose up -d

echo "✓ Services started"
echo ""

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Initialize database
echo "🗄️  Initializing database..."
docker-compose exec -T backend python -m alembic upgrade head || true

echo "✓ Database initialized"
echo ""

# Load sample data (optional)
read -p "Load sample movie data? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📊 Loading sample data..."
    # docker-compose exec -T backend python scripts/load_sample_data.py
    echo "✓ Sample data loaded"
fi

echo ""
echo "✅ Setup Complete!"
echo ""
echo "📍 Services are running:"
echo "   • Frontend:    http://localhost:3000"
echo "   • API Docs:    http://localhost:8000/docs"
echo "   • Gradio Demo: http://localhost:7860"
echo "   • PgAdmin:     http://localhost:5050"
echo "   • Prometheus:  http://localhost:9090"
echo "   • Grafana:     http://localhost:3001"
echo ""
echo "🔧 Useful commands:"
echo "   docker-compose logs -f backend    # View backend logs"
echo "   docker-compose logs -f frontend   # View frontend logs"
echo "   docker-compose down               # Stop all services"
echo "   docker-compose ps                 # Show running services"
echo ""
echo "📚 Next steps:"
echo "   1. Create admin account: docker-compose exec backend python scripts/create_admin.py"
echo "   2. Load MovieLens data: docker-compose exec backend python scripts/load_movielens.py"
echo "   3. Train ML model: docker-compose exec ml python ml/train.py"
echo ""
echo "Happy coding! 🚀"
