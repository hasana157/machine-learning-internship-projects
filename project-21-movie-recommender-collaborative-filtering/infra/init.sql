-- ============================================================================
-- CineMatch AI - PostgreSQL Initialization Script
-- ============================================================================
-- This script initializes the database with extensions and initial setup

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- For full-text search
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- For composite indexes
CREATE EXTENSION IF NOT EXISTS "btree_gist"; -- For exclusion constraints

-- Create indexes schema
CREATE SCHEMA IF NOT EXISTS indices;

-- Create audit schema for tracking changes
CREATE SCHEMA IF NOT EXISTS audit;

-- ============================================================================
-- AUDIT FUNCTION (for tracking modifications)
-- ============================================================================

CREATE OR REPLACE FUNCTION audit.audit_function() RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO audit.audit_log (
    table_name,
    record_id,
    action,
    old_data,
    new_data,
    changed_at,
    changed_by
  ) VALUES (
    TG_TABLE_NAME,
    NEW.id,
    TG_OP,
    to_jsonb(OLD),
    to_jsonb(NEW),
    NOW(),
    CURRENT_USER
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- AUDIT TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit.audit_log (
  id BIGSERIAL PRIMARY KEY,
  table_name TEXT NOT NULL,
  record_id INTEGER,
  action TEXT NOT NULL,
  old_data JSONB,
  new_data JSONB,
  changed_at TIMESTAMP DEFAULT NOW(),
  changed_by TEXT DEFAULT CURRENT_USER,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_audit_log_table ON audit.audit_log(table_name);
CREATE INDEX idx_audit_log_record ON audit.audit_log(record_id);
CREATE INDEX idx_audit_log_timestamp ON audit.audit_log(changed_at DESC);

-- ============================================================================
-- MATERIALIZED VIEWS (for reporting and analytics)
-- ============================================================================

-- User statistics materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS public.user_stats AS
SELECT
  u.id,
  u.username,
  COALESCE(r.rating_count, 0) as rating_count,
  COALESCE(ROUND(r.avg_rating::numeric, 2), 0) as avg_rating,
  COALESCE(e.event_count, 0) as event_count,
  u.created_at,
  NOW() as stats_updated_at
FROM users u
LEFT JOIN (
  SELECT user_id, COUNT(*) as rating_count, AVG(rating) as avg_rating
  FROM ratings
  GROUP BY user_id
) r ON u.id = r.user_id
LEFT JOIN (
  SELECT user_id, COUNT(*) as event_count
  FROM user_events
  GROUP BY user_id
) e ON u.id = e.user_id;

CREATE UNIQUE INDEX idx_user_stats_user_id ON user_stats(id);

-- Movie statistics materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS public.movie_stats AS
SELECT
  m.id,
  m.title,
  COALESCE(r.rating_count, 0) as rating_count,
  COALESCE(ROUND(r.avg_rating::numeric, 2), 0) as avg_rating,
  COALESCE(r.min_rating, 0) as min_rating,
  COALESCE(r.max_rating, 0) as max_rating,
  COALESCE(e.view_count, 0) as view_count,
  NOW() as stats_updated_at
FROM movies m
LEFT JOIN (
  SELECT
    movie_id,
    COUNT(*) as rating_count,
    AVG(rating) as avg_rating,
    MIN(rating) as min_rating,
    MAX(rating) as max_rating
  FROM ratings
  GROUP BY movie_id
) r ON m.id = r.movie_id
LEFT JOIN (
  SELECT movie_id, COUNT(*) as view_count
  FROM user_events
  WHERE event_type = 'view'
  GROUP BY movie_id
) e ON m.id = e.movie_id;

CREATE UNIQUE INDEX idx_movie_stats_movie_id ON movie_stats(id);

-- ============================================================================
-- AGGREGATE FUNCTIONS (for custom metrics)
-- ============================================================================

-- Function to calculate Hit@K
CREATE OR REPLACE FUNCTION calculate_hit_at_k(
  recommendations INT[],
  relevant_items INT[],
  k INT DEFAULT 10
) RETURNS FLOAT AS $$
DECLARE
  hit_count INT := 0;
  relevant_item INT;
BEGIN
  FOREACH relevant_item IN ARRAY relevant_items LOOP
    IF relevant_item = ANY(recommendations[1:k]) THEN
      hit_count := hit_count + 1;
      EXIT;
    END IF;
  END LOOP;
  
  RETURN CASE WHEN hit_count > 0 THEN 1.0 ELSE 0.0 END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================================
-- INITIAL SEED DATA (optional)
-- ============================================================================

-- Insert initial admin user (password: admin, must be changed in production)
INSERT INTO users (username, email, hashed_password, full_name, is_admin, is_active)
VALUES (
  'admin',
  'admin@cinematch.ai',
  '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5oeIS2xo.L28m', -- bcrypt hash of 'admin'
  'Administrator',
  true,
  true
)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- GRANTS & PERMISSIONS
-- ============================================================================

-- Create application role with limited permissions
CREATE ROLE cinematch_app WITH LOGIN PASSWORD 'change_this_password';

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO cinematch_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO cinematch_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO cinematch_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO cinematch_app;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE users IS 'User accounts with authentication details';
COMMENT ON TABLE movies IS 'Movie catalog with metadata and popularity scores';
COMMENT ON TABLE ratings IS 'User ratings of movies (0.5-5.0 scale)';
COMMENT ON TABLE user_events IS 'User interaction events (clicks, views, watches)';
COMMENT ON TABLE model_runs IS 'ML model training runs and versions';
COMMENT ON TABLE recommendation_cache IS 'Pre-computed recommendations cache';

-- ============================================================================
-- COMPLETION
-- ============================================================================

-- Set search path for convenience
ALTER DATABASE cinematch_db SET search_path = public, audit, indices;

-- Log initialization
DO $$ BEGIN
  RAISE NOTICE 'CineMatch AI Database Initialized Successfully';
  RAISE NOTICE 'Created % tables with proper indexing', (
    SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'
  );
END $$;
