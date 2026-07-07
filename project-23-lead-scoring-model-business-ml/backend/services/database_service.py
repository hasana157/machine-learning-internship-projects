"""
Database Service
Handles PostgreSQL operations for lead scores and metadata
"""

import logging
import json
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

import asyncpg
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class LeadScore(Base):
    """SQLAlchemy model for lead scores"""
    __tablename__ = 'lead_scores'
    
    id = Column(String, primary_key=True, default=str(uuid.uuid4()))
    lead_id = Column(String, unique=True, index=True, nullable=False)
    score = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    tier = Column(String, nullable=False)
    features = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BatchJob(Base):
    """SQLAlchemy model for batch scoring jobs"""
    __tablename__ = 'batch_jobs'
    
    id = Column(String, primary_key=True, default=str(uuid.uuid4()))
    status = Column(String, nullable=False)  # queued, processing, completed, failed
    total_leads = Column(Integer, nullable=False)
    processed_leads = Column(Integer, default=0)
    leads_data = Column(JSON, nullable=False)
    results = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class DatabaseService:
    """
    Database Service
    Manages all database operations
    """
    
    def __init__(self, database_url: str):
        """
        Initialize Database Service
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url
        self.pool = None
        self.engine = None
        self.SessionLocal = None
    
    async def connect(self) -> None:
        """Create async connection pool"""
        try:
            # Parse async connection string
            async_url = self.database_url.replace('postgresql://', 'postgresql+asyncpg://')
            
            self.pool = await asyncpg.create_pool(
                self.database_url.replace('postgresql://', ''),
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info("Connected to PostgreSQL database")
        
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Disconnected from PostgreSQL")
    
    async def health_check(self) -> bool:
        """
        Check database health
        
        Returns:
            True if database is accessible
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def save_score(
        self,
        lead_id: str,
        score: int,
        probability: float,
        tier: str,
        features: Optional[Dict] = None
    ) -> str:
        """
        Save lead score to database
        
        Args:
            lead_id: Unique lead identifier
            score: Score 0-100
            probability: Conversion probability 0-1
            tier: Lead tier (hot/warm/cold)
            features: Original features dictionary
        
        Returns:
            Record ID
        """
        try:
            record_id = str(uuid.uuid4())
            
            async with self.pool.acquire() as conn:
                # Insert or update
                await conn.execute("""
                    INSERT INTO lead_scores (id, lead_id, score, probability, tier, features, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (lead_id) DO UPDATE SET
                        score = $3,
                        probability = $4,
                        tier = $5,
                        features = $6,
                        updated_at = $8
                """, 
                record_id,
                lead_id,
                score,
                probability,
                tier,
                json.dumps(features) if features else None,
                datetime.utcnow(),
                datetime.utcnow()
                )
            
            logger.info(f"Saved score for lead {lead_id}")
            return record_id
        
        except Exception as e:
            logger.error(f"Error saving score: {e}")
            raise
    
    async def get_score(self, lead_id: str) -> Optional[Dict]:
        """
        Retrieve a lead score
        
        Args:
            lead_id: Unique lead identifier
        
        Returns:
            Score record or None
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM lead_scores WHERE lead_id = $1",
                    lead_id
                )
            
            if row:
                return dict(row)
            return None
        
        except Exception as e:
            logger.error(f"Error retrieving score: {e}")
            return None
    
    async def queue_batch_job(self, leads: List[Dict]) -> str:
        """
        Queue a batch scoring job
        
        Args:
            leads: List of lead dictionaries
        
        Returns:
            Job ID
        """
        try:
            job_id = str(uuid.uuid4())
            
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO batch_jobs (id, status, total_leads, leads_data, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                """,
                job_id,
                'queued',
                len(leads),
                json.dumps(leads),
                datetime.utcnow()
                )
            
            logger.info(f"Queued batch job {job_id} with {len(leads)} leads")
            return job_id
        
        except Exception as e:
            logger.error(f"Error queueing batch job: {e}")
            raise
    
    async def get_batch_job(self, job_id: str) -> Optional[Dict]:
        """
        Get batch job status
        
        Args:
            job_id: Job identifier
        
        Returns:
            Job record or None
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM batch_jobs WHERE id = $1",
                    job_id
                )
            
            if row:
                return dict(row)
            return None
        
        except Exception as e:
            logger.error(f"Error retrieving job: {e}")
            return None
    
    async def update_batch_job(
        self,
        job_id: str,
        status: str,
        processed_count: int = None,
        results: Optional[List] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update batch job status
        
        Args:
            job_id: Job identifier
            status: New status
            processed_count: Number of processed leads
            results: Batch results
            error_message: Error message if failed
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE batch_jobs
                    SET status = $1, processed_leads = $2, results = $3, 
                        error_message = $4, completed_at = $5
                    WHERE id = $6
                """,
                status,
                processed_count or 0,
                json.dumps(results) if results else None,
                error_message,
                datetime.utcnow() if status == 'completed' else None,
                job_id
                )
            
            logger.info(f"Updated batch job {job_id} to {status}")
        
        except Exception as e:
            logger.error(f"Error updating batch job: {e}")
            raise
    
    async def get_lead_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            async with self.pool.acquire() as conn:
                stats = {
                    'total_leads': await conn.fetchval("SELECT COUNT(*) FROM lead_scores"),
                    'avg_score': await conn.fetchval("SELECT AVG(score) FROM lead_scores"),
                    'hot_leads': await conn.fetchval("SELECT COUNT(*) FROM lead_scores WHERE tier = 'hot'"),
                    'warm_leads': await conn.fetchval("SELECT COUNT(*) FROM lead_scores WHERE tier = 'warm'"),
                    'cold_leads': await conn.fetchval("SELECT COUNT(*) FROM lead_scores WHERE tier = 'cold'"),
                }
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    async def get_recent_scores(self, limit: int = 100) -> List[Dict]:
        """
        Get most recent lead scores
        
        Args:
            limit: Number of records to retrieve
        
        Returns:
            List of score records
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM lead_scores
                    ORDER BY updated_at DESC
                    LIMIT $1
                """, limit)
            
            return [dict(row) for row in rows]
        
        except Exception as e:
            logger.error(f"Error retrieving recent scores: {e}")
            return []
    
    async def delete_old_records(self, days: int = 90) -> int:
        """
        Delete records older than N days
        
        Args:
            days: Age threshold in days
        
        Returns:
            Number of deleted records
        """
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM lead_scores
                    WHERE updated_at < NOW() - INTERVAL '%s days'
                """ % days)
            
            deleted_count = int(result.split()[-1])
            logger.info(f"Deleted {deleted_count} old records")
            return deleted_count
        
        except Exception as e:
            logger.error(f"Error deleting old records: {e}")
            return 0


def init_db(database_url: str) -> None:
    """
    Initialize database tables
    
    Args:
        database_url: PostgreSQL connection string
    """
    try:
        engine = create_engine(database_url, echo=False)
        
        # Create tables
        Base.metadata.create_all(engine)
        
        logger.info("Database tables initialized")
    
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
