#!/bin/bash
# LeadForge AI - Setup Script
# Automates local development environment setup

set -e  # Exit on error

echo "=========================================="
echo "LeadForge AI - Setup Script"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python3 --version || { echo -e "${RED}Python 3 not found${NC}"; exit 1; }

# 2. Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# 3. Activate virtual environment
source venv/bin/activate || { echo -e "${RED}Failed to activate venv${NC}"; exit 1; }
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# 4. Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip --quiet

# 5. Install dependencies
echo -e "${YELLOW}Installing backend dependencies...${NC}"
pip install -r backend/requirements.txt

echo -e "${YELLOW}Installing dashboard dependencies...${NC}"
cd dashboard
pip install -r requirements.txt
cd ..

echo -e "${GREEN}✓ Dependencies installed${NC}"

# 6. Create directory structure
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p models logs datasets/raw notebooks
echo -e "${GREEN}✓ Directories created${NC}"

# 7. Copy environment file
echo -e "${YELLOW}Setting up environment variables...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env file from template${NC}"
    echo -e "${YELLOW}⚠️  Update .env with your settings (database, Redis URLs, etc.)${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# 8. Check Docker
echo -e "${YELLOW}Checking Docker...${NC}"
if command -v docker &> /dev/null; then
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        echo -e "${GREEN}✓ Docker and Docker Compose are installed${NC}"
        
        # Start services
        read -p "Start PostgreSQL and Redis with docker-compose? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose up -d postgres redis
            echo -e "${GREEN}✓ PostgreSQL and Redis started${NC}"
            
            # Wait for services to be ready
            echo -e "${YELLOW}Waiting for services to be ready...${NC}"
            sleep 10
        fi
    else
        echo -e "${YELLOW}⚠️  Docker Compose not found. Manual setup required.${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Docker not installed. Skipping Docker setup.${NC}"
    echo -e "${YELLOW}Install Docker: https://docs.docker.com/get-docker/${NC}"
fi

# 9. Dataset check
echo -e "${YELLOW}Checking dataset...${NC}"
if [ -f "datasets/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv" ]; then
    echo -e "${GREEN}✓ Dataset found${NC}"
    
    # Ask to train model
    read -p "Train ML model now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 datasets/train_model.py
    fi
else
    echo -e "${YELLOW}⚠️  Dataset not found${NC}"
    echo -e "${YELLOW}Download from Kaggle:${NC}"
    echo -e "${YELLOW}https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset${NC}"
    echo -e "${YELLOW}Place CSV in: datasets/raw/${NC}"
fi

# 10. Summary
echo ""
echo "=========================================="
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download dataset from Kaggle (if not done)"
echo "2. Update .env with your settings"
echo "3. Train model: python3 datasets/train_model.py"
echo "4. Start backend: cd backend && python -m uvicorn app:app --reload"
echo "5. Start dashboard: cd dashboard && python app.py"
echo "6. Start frontend: cd frontend && npm install && npm run dev"
echo ""
echo "API docs: http://localhost:8000/api/docs"
echo "Dashboard: http://localhost:8050"
echo "Frontend: http://localhost:5173"
echo ""
