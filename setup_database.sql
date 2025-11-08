-- ============================================================================
-- AI Surveillance Camera - Database Setup SQL
-- PostgreSQL with pgvector extension
-- ============================================================================

-- Create database (run this as postgres superuser)
-- Uncomment if database doesn't exist:
-- CREATE DATABASE surveillance;

-- Connect to the database
\c surveillance

-- ============================================================================
-- 1. Enable pgvector extension
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extension
SELECT * FROM pg_extension WHERE extname = 'vector';

-- ============================================================================
-- 2. Create faces table
-- ============================================================================
DROP TABLE IF EXISTS faces CASCADE;

CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    gender VARCHAR(10),
    age_approx INTEGER,
    embedding vector(512),  -- 512-dimensional face embedding
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Constraints
    CONSTRAINT valid_age CHECK (age_approx IS NULL OR (age_approx >= 0 AND age_approx <= 150)),
    CONSTRAINT valid_gender CHECK (gender IS NULL OR gender IN ('Male', 'Female', 'Unknown'))
);

-- Add comments for documentation
COMMENT ON TABLE faces IS 'Stores face embeddings with metadata for face recognition';
COMMENT ON COLUMN faces.id IS 'Primary key (auto-increment)';
COMMENT ON COLUMN faces.name IS 'Person''s name';
COMMENT ON COLUMN faces.gender IS 'Gender (Male/Female/Unknown)';
COMMENT ON COLUMN faces.age_approx IS 'Approximate age';
COMMENT ON COLUMN faces.embedding IS '512-dimensional face embedding vector from InsightFace';
COMMENT ON COLUMN faces.metadata IS 'Additional metadata (recording info, camera ID, etc.)';

-- ============================================================================
-- 3. Create indexes for performance
-- ============================================================================

-- HNSW index for fast vector similarity search
-- m=16: number of connections per layer (affects recall vs speed)
-- ef_construction=64: quality during index build (higher = better quality, slower build)
CREATE INDEX faces_embedding_idx 
ON faces USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

COMMENT ON INDEX faces_embedding_idx IS 'HNSW index for fast approximate nearest neighbor search using cosine distance';

-- B-tree index for name searches
CREATE INDEX faces_name_idx ON faces (name);

-- B-tree index for timestamps
CREATE INDEX faces_created_at_idx ON faces (created_at DESC);

-- GIN index for JSONB metadata searches
CREATE INDEX faces_metadata_idx ON faces USING GIN (metadata);

-- ============================================================================
-- 4. Create triggers for automatic timestamp updates
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to call the function before updates
DROP TRIGGER IF EXISTS update_faces_updated_at ON faces;
CREATE TRIGGER update_faces_updated_at
    BEFORE UPDATE ON faces
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 5. Create useful views
-- ============================================================================

-- View for face statistics
CREATE OR REPLACE VIEW face_statistics AS
SELECT 
    COUNT(*) as total_faces,
    COUNT(DISTINCT name) as unique_names,
    COUNT(CASE WHEN gender = 'Male' THEN 1 END) as male_count,
    COUNT(CASE WHEN gender = 'Female' THEN 1 END) as female_count,
    ROUND(AVG(age_approx), 1) as avg_age,
    MIN(created_at) as first_face_added,
    MAX(created_at) as last_face_added
FROM faces;

COMMENT ON VIEW face_statistics IS 'Summary statistics of stored faces';

-- ============================================================================
-- 6. Create helper functions
-- ============================================================================

-- Function to search faces by similarity
CREATE OR REPLACE FUNCTION search_similar_faces(
    query_embedding vector(512),
    match_threshold FLOAT DEFAULT 0.6,
    max_results INT DEFAULT 5
)
RETURNS TABLE (
    face_id INT,
    face_name VARCHAR(255),
    face_gender VARCHAR(10),
    face_age INTEGER,
    similarity FLOAT,
    face_metadata JSONB,
    face_created_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        id,
        name,
        gender,
        age_approx,
        (1 - (embedding <=> query_embedding))::FLOAT as sim,
        metadata,
        created_at
    FROM faces
    WHERE (1 - (embedding <=> query_embedding)) >= match_threshold
    ORDER BY embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION search_similar_faces IS 'Search for faces by vector similarity using cosine distance';

-- ============================================================================
-- 7. Insert sample data for testing (optional)
-- ============================================================================

-- You can add sample faces here for testing
-- Example:
-- INSERT INTO faces (name, gender, age_approx, embedding, metadata)
-- VALUES ('Test Person', 'Male', 30, array_fill(0.1, ARRAY[512])::vector(512), '{"test": true}'::jsonb);

-- ============================================================================
-- 8. Grant permissions (adjust as needed)
-- ============================================================================

-- Grant all privileges to the application user
-- GRANT ALL PRIVILEGES ON TABLE faces TO your_app_user;
-- GRANT USAGE, SELECT ON SEQUENCE faces_id_seq TO your_app_user;

-- ============================================================================
-- 9. Display setup summary
-- ============================================================================

\echo '============================================================================'
\echo 'Database Setup Complete!'
\echo '============================================================================'

SELECT 'Database' as component, current_database() as name;
SELECT 'Extension' as component, extname as name FROM pg_extension WHERE extname = 'vector';
SELECT 'Table' as component, 'faces' as name;
SELECT 'Indexes' as component, COUNT(*)::text as name FROM pg_indexes WHERE tablename = 'faces';
SELECT 'Triggers' as component, COUNT(*)::text as name FROM pg_trigger WHERE tgrelid = 'faces'::regclass;

\echo ''
\echo 'Table structure:'
\d faces

\echo ''
\echo 'Indexes:'
\di faces_*

\echo ''
\echo 'Statistics:'
SELECT * FROM face_statistics;

\echo ''
\echo '============================================================================'
\echo 'Ready to use!'
\echo '============================================================================'
\echo 'You can now run: python gui_app_qt.py'
\echo ''

