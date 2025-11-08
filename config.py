"""
Configuration file for AI Surveillance Camera
"""

# PostgreSQL Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'surveillance',
    'user': 'postgres',
    'password': 'postgres'  # Change this to your PostgreSQL password
}

# Face Matching Configuration
FACE_MATCH_THRESHOLD = 0.6  # Minimum similarity for face matching (0-1)
FACE_MATCH_TOP_K = 5  # Number of top matches to retrieve

# Recording Configuration
RECORDING_DURATION = 10.0  # seconds
RECORDING_AUTO_SAVE = True  # Automatically save to database after recording

# Vector Search Configuration
VECTOR_SEARCH_ENABLED = True  # Enable database vector search
FALLBACK_TO_MEMORY = True  # Fallback to in-memory matching if database unavailable

