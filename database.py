import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from typing import List, Tuple, Optional, Dict
import json
from contextlib import contextmanager


class FaceDatabase:
    """
    PostgreSQL database manager with pgvector extension for face embeddings.
    
    Features:
    - Store face embeddings (512D vectors) with metadata (name, gender, age)
    - Vector similarity search using cosine distance
    - Efficient indexing with HNSW (Hierarchical Navigable Small World)
    """
    
    def __init__(self, host='localhost', port=5432, database='surveillance', 
                 user='postgres', password='postgres'):
        """
        Initialize database connection parameters.
        
        Args:
            host: PostgreSQL server host
            port: PostgreSQL server port
            database: Database name
            user: Database user
            password: Database password
        """
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None
        self.embedding_dim = 512  # InsightFace embedding dimension
    
    def connect(self) -> bool:
        """
        Establish connection to PostgreSQL database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            print(f"[OK] Connected to database: {self.connection_params['database']}")
            return True
        except psycopg2.OperationalError as e:
            print(f"[FAIL] Database connection failed: {e}")
            return False
        except Exception as e:
            print(f"[FAIL] Unexpected error: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("[OK] Database disconnected")
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor."""
        if not self.conn:
            raise ConnectionError("Database not connected. Call connect() first.")
        
        cursor = self.conn.cursor()
        try:
            yield cursor
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def initialize_database(self) -> bool:
        """
        Initialize database schema with pgvector extension.
        Creates tables and indexes if they don't exist.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            with self.get_cursor() as cursor:
                # Enable pgvector extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                print("[OK] pgvector extension enabled")
                
                # Create faces table
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS faces (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        gender VARCHAR(10),
                        age_approx INTEGER,
                        embedding vector({self.embedding_dim}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB DEFAULT '{{}}'::jsonb
                    );
                """)
                print("[OK] Faces table created")
                
                # Create index for faster vector search (HNSW - Hierarchical Navigable Small World)
                # m=16 (connections per layer), ef_construction=64 (quality during index build)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS faces_embedding_idx 
                    ON faces USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
                print("[OK] HNSW index created for vector search")
                
                # Create index for name searches
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS faces_name_idx 
                    ON faces (name);
                """)
                print("[OK] Name index created")
                
                # Create updated_at trigger
                cursor.execute("""
                    CREATE OR REPLACE FUNCTION update_updated_at_column()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ language 'plpgsql';
                """)
                
                cursor.execute("""
                    DROP TRIGGER IF EXISTS update_faces_updated_at ON faces;
                """)
                
                cursor.execute("""
                    CREATE TRIGGER update_faces_updated_at 
                    BEFORE UPDATE ON faces 
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                """)
                print("[OK] Triggers created")
            
            return True
            
        except Exception as e:
            print(f"[FAIL] Database initialization failed: {e}")
            return False
    
    def save_face(self, name: str, embedding: np.ndarray, 
                  gender: Optional[str] = None, 
                  age: Optional[int] = None,
                  metadata: Optional[Dict] = None) -> Optional[int]:
        """
        Save a face embedding to the database.
        
        Args:
            name: Person's name
            embedding: Face embedding vector (512D numpy array)
            gender: Gender ('Male' or 'Female')
            age: Approximate age
            metadata: Additional metadata as dictionary
        
        Returns:
            Face ID if successful, None otherwise
        """
        try:
            if embedding.shape[0] != self.embedding_dim:
                raise ValueError(f"Embedding must be {self.embedding_dim}D, got {embedding.shape[0]}D")
            
            # Convert numpy array to list for PostgreSQL
            embedding_list = embedding.tolist()
            
            # Convert metadata to JSON
            metadata_json = json.dumps(metadata) if metadata else '{}'
            
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO faces (name, gender, age_approx, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    RETURNING id;
                """, (name, gender, age, embedding_list, metadata_json))
                
                face_id = cursor.fetchone()[0]
                print(f"[OK] Face saved: {name} (ID: {face_id})")
                return face_id
                
        except Exception as e:
            print(f"[FAIL] Failed to save face: {e}")
            return None
    
    def search_similar_faces(self, embedding: np.ndarray, 
                            limit: int = 5, 
                            threshold: float = 0.6) -> List[Tuple]:
        """
        Search for similar faces using vector similarity (cosine distance).
        
        Args:
            embedding: Query embedding vector (512D numpy array)
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0-1, where 1 is identical)
        
        Returns:
            List of tuples: (id, name, gender, age_approx, similarity, metadata)
        """
        try:
            if embedding.shape[0] != self.embedding_dim:
                raise ValueError(f"Embedding must be {self.embedding_dim}D, got {embedding.shape[0]}D")
            
            # Convert numpy array to list for PostgreSQL
            embedding_list = embedding.tolist()
            
            with self.get_cursor() as cursor:
                # Using cosine distance: 1 - cosine_similarity
                # We convert it back to similarity for easier interpretation
                cursor.execute("""
                    SELECT 
                        id, 
                        name, 
                        gender, 
                        age_approx,
                        1 - (embedding <=> %s::vector) as similarity,
                        metadata,
                        created_at
                    FROM faces
                    WHERE 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (embedding_list, embedding_list, threshold, embedding_list, limit))
                
                results = cursor.fetchall()
                return results
                
        except Exception as e:
            print(f"[FAIL] Search failed: {e}")
            return []
    
    def get_face_by_id(self, face_id: int) -> Optional[Dict]:
        """
        Retrieve a face record by ID.
        
        Args:
            face_id: Face ID
        
        Returns:
            Dictionary with face data or None
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT id, name, gender, age_approx, embedding, metadata, created_at, updated_at
                    FROM faces
                    WHERE id = %s;
                """, (face_id,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'id': result[0],
                        'name': result[1],
                        'gender': result[2],
                        'age_approx': result[3],
                        'embedding': np.array(result[4]),
                        'metadata': result[5],
                        'created_at': result[6],
                        'updated_at': result[7]
                    }
                return None
                
        except Exception as e:
            print(f"[FAIL] Failed to retrieve face: {e}")
            return None
    
    def get_all_faces(self, limit: int = 100) -> List[Dict]:
        """
        Get all faces from database.
        
        Args:
            limit: Maximum number of records to retrieve
        
        Returns:
            List of face dictionaries
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT id, name, gender, age_approx, created_at, updated_at
                    FROM faces
                    ORDER BY created_at DESC
                    LIMIT %s;
                """, (limit,))
                
                results = cursor.fetchall()
                return [{
                    'id': r[0],
                    'name': r[1],
                    'gender': r[2],
                    'age_approx': r[3],
                    'created_at': r[4],
                    'updated_at': r[5]
                } for r in results]
                
        except Exception as e:
            print(f"[FAIL] Failed to retrieve faces: {e}")
            return []
    
    def delete_face(self, face_id: int) -> bool:
        """
        Delete a face record by ID.
        
        Args:
            face_id: Face ID to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("DELETE FROM faces WHERE id = %s;", (face_id,))
                print(f"[OK] Face deleted: ID {face_id}")
                return True
                
        except Exception as e:
            print(f"[FAIL] Failed to delete face: {e}")
            return False
    
    def update_face(self, face_id: int, 
                   name: Optional[str] = None,
                   gender: Optional[str] = None,
                   age: Optional[int] = None,
                   metadata: Optional[Dict] = None) -> bool:
        """
        Update face metadata (not embedding).
        
        Args:
            face_id: Face ID to update
            name: New name
            gender: New gender
            age: New age
            metadata: New metadata
        
        Returns:
            True if successful, False otherwise
        """
        try:
            updates = []
            params = []
            
            if name is not None:
                updates.append("name = %s")
                params.append(name)
            if gender is not None:
                updates.append("gender = %s")
                params.append(gender)
            if age is not None:
                updates.append("age_approx = %s")
                params.append(age)
            if metadata is not None:
                updates.append("metadata = %s::jsonb")
                params.append(json.dumps(metadata))
            
            if not updates:
                return False
            
            params.append(face_id)
            query = f"UPDATE faces SET {', '.join(updates)} WHERE id = %s;"
            
            with self.get_cursor() as cursor:
                cursor.execute(query, params)
                print(f"[OK] Face updated: ID {face_id}")
                return True
                
        except Exception as e:
            print(f"[FAIL] Failed to update face: {e}")
            return False
    
    def get_face_count(self) -> int:
        """
        Get total number of faces in database.
        
        Returns:
            Number of faces
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM faces;")
                count = cursor.fetchone()[0]
                return count
        except Exception as e:
            print(f"[FAIL] Failed to get face count: {e}")
            return 0
    
    def search_by_name(self, name: str, exact: bool = False) -> List[Dict]:
        """
        Search faces by name.
        
        Args:
            name: Name to search for
            exact: If True, exact match; if False, partial match
        
        Returns:
            List of face dictionaries
        """
        try:
            with self.get_cursor() as cursor:
                if exact:
                    cursor.execute("""
                        SELECT id, name, gender, age_approx, created_at, updated_at
                        FROM faces
                        WHERE name = %s
                        ORDER BY created_at DESC;
                    """, (name,))
                else:
                    cursor.execute("""
                        SELECT id, name, gender, age_approx, created_at, updated_at
                        FROM faces
                        WHERE name ILIKE %s
                        ORDER BY created_at DESC;
                    """, (f'%{name}%',))
                
                results = cursor.fetchall()
                return [{
                    'id': r[0],
                    'name': r[1],
                    'gender': r[2],
                    'age_approx': r[3],
                    'created_at': r[4],
                    'updated_at': r[5]
                } for r in results]
                
        except Exception as e:
            print(f"[FAIL] Search by name failed: {e}")
            return []


def test_connection(host='localhost', port=5432, database='surveillance',
                   user='postgres', password='postgres'):
    """
    Test database connection and setup.
    
    Args:
        host: PostgreSQL server host
        port: PostgreSQL server port
        database: Database name
        user: Database user
        password: Database password
    
    Returns:
        True if test successful, False otherwise
    """
    print("=" * 70)
    print("PostgreSQL + pgvector Connection Test")
    print("=" * 70)
    
    db = FaceDatabase(host=host, port=port, database=database, user=user, password=password)
    
    # Test connection
    if not db.connect():
        print("\n[FAIL] Connection failed!")
        print("\nMake sure:")
        print("  1. PostgreSQL is installed and running")
        print("  2. pgvector extension is installed: https://github.com/pgvector/pgvector")
        print("  3. Database credentials are correct")
        print("  4. Database exists (or user has CREATE DATABASE permission)")
        return False
    
    # Initialize database
    if not db.initialize_database():
        print("\n[FAIL] Database initialization failed!")
        db.disconnect()
        return False
    
    # Test operations
    print("\nTesting database operations...")
    
    # Create test embedding
    test_embedding = np.random.rand(512).astype(np.float64)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)  # Normalize
    
    # Save test face
    face_id = db.save_face(
        name="Test Person",
        embedding=test_embedding,
        gender="Male",
        age=30,
        metadata={"test": True}
    )
    
    if face_id:
        # Search similar faces
        results = db.search_similar_faces(test_embedding, limit=1)
        if results and len(results) > 0:
            print(f"[OK] Vector search working: Found {len(results)} result(s)")
        
        # Clean up test data
        db.delete_face(face_id)
        print("[OK] Test data cleaned up")
    
    # Get statistics
    count = db.get_face_count()
    print(f"\n[OK] Database has {count} face(s)")
    
    db.disconnect()
    
    print("\n" + "=" * 70)
    print("[OK] All tests passed! Database is ready.")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    # Run connection test
    test_connection()

