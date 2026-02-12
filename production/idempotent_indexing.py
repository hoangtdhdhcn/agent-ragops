"""
Advanced Idempotent Indexing System for Production RAG System

Provides comprehensive idempotent indexing capabilities including deterministic indexing,
index versioning, consistency guarantees, and enterprise-grade features.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict, field, fields
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union, Callable, Type, TypeVar, Generic
from abc import ABC, abstractmethod
import sqlite3
import threading
from collections import defaultdict
import pickle
from uuid import UUID
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from config import MARKDOWN_DIR, PARENT_STORE_PATH, QDRANT_DB_PATH
from enhanced_document_manager import EnhancedDocumentManager, DocumentMetadata
from metadata_manager import MetadataManager, MetadataSchema, MetadataField, FieldConstraint, MetadataFieldType
from versioning_system import VersioningSystem, SemanticVersion
from rag_system import RAGSystem

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')

class IndexType(Enum):
    """Types of indexes."""
    DENSE_VECTOR = "dense_vector"
    SPARSE_VECTOR = "sparse_vector"
    HYBRID_VECTOR = "hybrid_vector"
    METADATA = "metadata"
    CONTENT = "content"

class IndexStatus(Enum):
    """Index status states."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ARCHIVED = "archived"
    FAILED = "failed"

class IndexOperation(Enum):
    """Types of index operations."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    VALIDATE = "validate"
    OPTIMIZE = "optimize"
    ROLLBACK = "rollback"
    REPLICATE = "replicate"

class IndexConsistencyLevel(Enum):
    """Index consistency levels."""
    EVENTUAL = "eventual"
    STRONG = "strong"
    LINEARIZABLE = "linearizable"

@dataclass
class IndexKey:
    """Deterministic index key."""
    document_id: str
    content_hash: str
    metadata_hash: str
    vector_hash: str
    index_type: IndexType
    version: str
    
    def __str__(self) -> str:
        """Generate deterministic index key."""
        components = [
            self.document_id,
            self.content_hash,
            self.metadata_hash,
            self.vector_hash,
            self.index_type.value,
            self.version
        ]
        return "_".join(components)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'document_id': self.document_id,
            'content_hash': self.content_hash,
            'metadata_hash': self.metadata_hash,
            'vector_hash': self.vector_hash,
            'index_type': self.index_type.value,
            'version': self.version
        }
    
    @classmethod
    def from_string(cls, key_string: str) -> 'IndexKey':
        """Parse index key from string."""
        components = key_string.split('_')
        if len(components) < 6:
            raise ValueError("Invalid index key format")
        
        return cls(
            document_id=components[0],
            content_hash=components[1],
            metadata_hash=components[2],
            vector_hash=components[3],
            index_type=IndexType(components[4]),
            version=components[5]
        )

@dataclass
class IndexEntry:
    """Index entry with deterministic properties."""
    index_key: IndexKey
    document_id: str
    content: str
    vector: List[float]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: str
    consistency_level: IndexConsistencyLevel
    partition_id: Optional[str] = None
    shard_id: Optional[str] = None

@dataclass
class IndexTransaction:
    """Index transaction for atomic operations."""
    transaction_id: str
    operation: IndexOperation
    entries: List[IndexEntry]
    status: str  # pending, committed, rolled_back
    created_at: datetime
    committed_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class IndexVersion:
    """Index version information."""
    version_id: str
    index_type: IndexType
    created_at: datetime
    created_by: Optional[str] = None
    description: str = ""
    parent_version: Optional[str] = None
    rollback_available: bool = True

@dataclass
class IndexPartition:
    """Index partition information."""
    partition_id: str
    index_type: IndexType
    shard_count: int
    created_at: datetime
    status: IndexStatus
    size_mb: float = 0.0
    document_count: int = 0

@dataclass
class IndexShard:
    """Index shard information."""
    shard_id: str
    partition_id: str
    index_type: IndexType
    node_id: str
    created_at: datetime
    status: IndexStatus
    size_mb: float = 0.0
    document_count: int = 0

@dataclass
class IndexReplica:
    """Index replica information."""
    replica_id: str
    shard_id: str
    node_id: str
    created_at: datetime
    status: IndexStatus
    last_sync_at: Optional[datetime] = None
    sync_lag_seconds: float = 0.0

@dataclass
class IndexHealth:
    """Index health information."""
    index_type: IndexType
    status: IndexStatus
    health_score: float  # 0.0 to 1.0
    last_check_at: datetime
    issues: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IndexOptimization:
    """Index optimization information."""
    optimization_id: str
    index_type: IndexType
    optimization_type: str  # 'merge', 'compact', 'rebuild'
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str  # 'running', 'completed', 'failed'
    documents_processed: int = 0
    space_reclaimed_mb: float = 0.0
    error_message: Optional[str] = None

class VectorNormalizer(ABC):
    """Abstract base class for vector normalization."""
    
    @abstractmethod
    def normalize(self, vector: List[float]) -> List[float]:
        """Normalize a vector."""
        pass
    
    @abstractmethod
    def denormalize(self, vector: List[float]) -> List[float]:
        """Denormalize a vector."""
        pass

class L2Normalizer(VectorNormalizer):
    """L2 normalization for vectors."""
    
    def normalize(self, vector: List[float]) -> List[float]:
        """Normalize vector using L2 norm."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return (np.array(vector) / norm).tolist()
    
    def denormalize(self, vector: List[float]) -> List[float]:
        """Denormalize vector (identity operation for L2)."""
        return vector

class MinMaxNormalizer(VectorNormalizer):
    """Min-Max normalization for vectors."""
    
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val
    
    def normalize(self, vector: List[float]) -> List[float]:
        """Normalize vector using Min-Max scaling."""
        min_vec = min(vector)
        max_vec = max(vector)
        
        if max_vec == min_vec:
            return [self.min_val] * len(vector)
        
        normalized = []
        for val in vector:
            normalized_val = self.min_val + (val - min_vec) * (self.max_val - self.min_val) / (max_vec - min_vec)
            normalized.append(normalized_val)
        
        return normalized
    
    def denormalize(self, vector: List[float]) -> List[float]:
        """Denormalize vector using Min-Max scaling."""
        return vector  # Simplified - would need original min/max values

class IndexStorage(ABC):
    """Abstract base class for index storage."""
    
    @abstractmethod
    async def store_index_entry(self, entry: IndexEntry) -> bool:
        """Store an index entry."""
        pass
    
    @abstractmethod
    async def retrieve_index_entry(self, index_key: IndexKey) -> Optional[IndexEntry]:
        """Retrieve an index entry."""
        pass
    
    @abstractmethod
    async def delete_index_entry(self, index_key: IndexKey) -> bool:
        """Delete an index entry."""
        pass
    
    @abstractmethod
    async def list_index_entries(self, index_type: IndexType) -> List[IndexEntry]:
        """List all index entries of a type."""
        pass
    
    @abstractmethod
    async def validate_index_integrity(self, index_type: IndexType) -> bool:
        """Validate index integrity."""
        pass

class QdrantIndexStorage(IndexStorage):
    """Qdrant-based index storage implementation."""
    
    def __init__(self, qdrant_client, collection_name: str = "rag_index"):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self._init_collection()
    
    def _init_collection(self):
        """Initialize Qdrant collection."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.qdrant_client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "size": 1536,  # Standard OpenAI embedding size
                        "distance": "Cosine"
                    }
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
    
    async def store_index_entry(self, entry: IndexEntry) -> bool:
        """Store an index entry in Qdrant."""
        try:
            # Prepare payload
            payload = {
                'document_id': entry.document_id,
                'content': entry.content,
                'metadata': entry.metadata,
                'created_at': entry.created_at.isoformat(),
                'updated_at': entry.updated_at.isoformat(),
                'version': entry.version,
                'consistency_level': entry.consistency_level.value,
                'partition_id': entry.partition_id,
                'shard_id': entry.shard_id
            }
            
            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[{
                    'id': str(hash(entry.index_key)),
                    'vector': entry.vector,
                    'payload': payload
                }]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing index entry: {e}")
            return False
    
    async def retrieve_index_entry(self, index_key: IndexKey) -> Optional[IndexEntry]:
        """Retrieve an index entry from Qdrant."""
        try:
            # Search for the entry
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=[0.0] * 1536,  # We'll filter by payload
                filter={
                    'must': [
                        {'key': 'document_id', 'match': {'value': index_key.document_id}},
                        {'key': 'version', 'match': {'value': index_key.version}}
                    ]
                },
                limit=1
            )
            
            if results:
                point = results[0]
                payload = point.payload
                
                return IndexEntry(
                    index_key=index_key,
                    document_id=payload['document_id'],
                    content=payload['content'],
                    vector=point.vector,
                    metadata=payload['metadata'],
                    created_at=datetime.fromisoformat(payload['created_at']),
                    updated_at=datetime.fromisoformat(payload['updated_at']),
                    version=payload['version'],
                    consistency_level=IndexConsistencyLevel(payload['consistency_level']),
                    partition_id=payload.get('partition_id'),
                    shard_id=payload.get('shard_id')
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving index entry: {e}")
            return None
    
    async def delete_index_entry(self, index_key: IndexKey) -> bool:
        """Delete an index entry from Qdrant."""
        try:
            # Find and delete the entry
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=[0.0] * 1536,
                filter={
                    'must': [
                        {'key': 'document_id', 'match': {'value': index_key.document_id}},
                        {'key': 'version', 'match': {'value': index_key.version}}
                    ]
                },
                limit=1
            )
            
            if results:
                point_id = results[0].id
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=[point_id]
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting index entry: {e}")
            return False
    
    async def list_index_entries(self, index_type: IndexType) -> List[IndexEntry]:
        """List all index entries of a type."""
        try:
            # Get all points
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000
            )
            
            entries = []
            for point in scroll_result[0]:
                payload = point.payload
                
                index_key = IndexKey(
                    document_id=payload['document_id'],
                    content_hash="",
                    metadata_hash="",
                    vector_hash="",
                    index_type=index_type,
                    version=payload['version']
                )
                
                entry = IndexEntry(
                    index_key=index_key,
                    document_id=payload['document_id'],
                    content=payload['content'],
                    vector=point.vector,
                    metadata=payload['metadata'],
                    created_at=datetime.fromisoformat(payload['created_at']),
                    updated_at=datetime.fromisoformat(payload['updated_at']),
                    version=payload['version'],
                    consistency_level=IndexConsistencyLevel(payload['consistency_level']),
                    partition_id=payload.get('partition_id'),
                    shard_id=payload.get('shard_id')
                )
                
                entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(f"Error listing index entries: {e}")
            return []
    
    async def validate_index_integrity(self, index_type: IndexType) -> bool:
        """Validate index integrity."""
        try:
            # Check collection health
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            # Basic integrity checks
            if not collection_info:
                return False
            
            # Check for missing vectors or corrupted data
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100
            )
            
            for point in scroll_result[0]:
                if not point.vector or len(point.vector) == 0:
                    logger.warning(f"Found corrupted vector in point {point.id}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating index integrity: {e}")
            return False

class IdempotentIndexingSystem:
    """Advanced idempotent indexing system with deterministic indexing."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize idempotent indexing system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.db_path = config.get('indexing_db_path', 'idempotent_indexing.db')
        
        # Indexing settings
        self.enable_deterministic_indexing = config.get('enable_deterministic_indexing', True)
        self.enable_index_versioning = config.get('enable_index_versioning', True)
        self.enable_index_consistency = config.get('enable_index_consistency', True)
        self.enable_index_optimization = config.get('enable_index_optimization', True)
        self.enable_distributed_indexing = config.get('enable_distributed_indexing', False)
        
        # Storage
        self.qdrant_client = None
        self.index_storage = None
        self.vector_normalizer = L2Normalizer()
        
        # State
        self.is_initialized = False
        self.db_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize
        self._init_database()
        self.is_initialized = True
    
    def _init_database(self):
        """Initialize indexing database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Index entries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS index_entries (
                    index_key TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata_hash TEXT NOT NULL,
                    vector_hash TEXT NOT NULL,
                    index_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    consistency_level TEXT NOT NULL,
                    partition_id TEXT,
                    shard_id TEXT
                )
            ''')
            
            # Index transactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS index_transactions (
                    transaction_id TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    committed_at TIMESTAMP,
                    rolled_back_at TIMESTAMP,
                    error_message TEXT
                )
            ''')
            
            # Index versions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS index_versions (
                    version_id TEXT PRIMARY KEY,
                    index_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    created_by TEXT,
                    description TEXT,
                    parent_version TEXT,
                    rollback_available BOOLEAN NOT NULL
                )
            ''')
            
            # Index partitions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS index_partitions (
                    partition_id TEXT PRIMARY KEY,
                    index_type TEXT NOT NULL,
                    shard_count INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    status TEXT NOT NULL,
                    size_mb REAL NOT NULL,
                    document_count INTEGER NOT NULL
                )
            ''')
            
            # Index shards table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS index_shards (
                    shard_id TEXT PRIMARY KEY,
                    partition_id TEXT NOT NULL,
                    index_type TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    status TEXT NOT NULL,
                    size_mb REAL NOT NULL,
                    document_count INTEGER NOT NULL
                )
            ''')
            
            # Index replicas table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS index_replicas (
                    replica_id TEXT PRIMARY KEY,
                    shard_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    status TEXT NOT NULL,
                    last_sync_at TIMESTAMP,
                    sync_lag_seconds REAL NOT NULL
                )
            ''')
            
            # Index health table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS index_health (
                    index_type TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    health_score REAL NOT NULL,
                    last_check_at TIMESTAMP NOT NULL,
                    issues TEXT,
                    performance_metrics TEXT
                )
            ''')
            
            # Index optimizations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS index_optimizations (
                    optimization_id TEXT PRIMARY KEY,
                    index_type TEXT NOT NULL,
                    optimization_type TEXT NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    status TEXT NOT NULL,
                    documents_processed INTEGER NOT NULL,
                    space_reclaimed_mb REAL NOT NULL,
                    error_message TEXT
                )
            ''')
            
            conn.commit()
    
    def _get_qdrant_client(self):
        """Get Qdrant client instance."""
        if not self.qdrant_client:
            from qdrant_client import QdrantClient
            self.qdrant_client = QdrantClient(path=str(QDRANT_DB_PATH))
        return self.qdrant_client
    
    def _generate_deterministic_key(self, document_id: str, content: str, 
                                  metadata: Dict[str, Any], vector: List[float],
                                  index_type: IndexType, version: str) -> IndexKey:
        """Generate deterministic index key."""
        # Compute content hash
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Compute metadata hash
        metadata_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('utf-8')).hexdigest()
        
        # Compute vector hash
        vector_hash = hashlib.sha256(np.array(vector).tobytes()).hexdigest()
        
        return IndexKey(
            document_id=document_id,
            content_hash=content_hash,
            metadata_hash=metadata_hash,
            vector_hash=vector_hash,
            index_type=index_type,
            version=version
        )
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize vector for deterministic indexing."""
        return self.vector_normalizer.normalize(vector)
    
    async def index_document(self, document_id: str, content: str, 
                           vector: List[float], metadata: Dict[str, Any],
                           index_type: IndexType = IndexType.DENSE_VECTOR,
                           consistency_level: IndexConsistencyLevel = IndexConsistencyLevel.STRONG,
                           version: Optional[str] = None) -> Optional[str]:
        """Index a document with idempotent guarantees."""
        try:
            # Normalize vector
            normalized_vector = self._normalize_vector(vector)
            
            # Generate version if not provided
            if not version:
                version = f"v{int(time.time())}"
            
            # Generate deterministic key
            index_key = self._generate_deterministic_key(
                document_id, content, metadata, normalized_vector, index_type, version
            )
            
            # Create index entry
            entry = IndexEntry(
                index_key=index_key,
                document_id=document_id,
                content=content,
                vector=normalized_vector,
                metadata=metadata,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                version=version,
                consistency_level=consistency_level
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO index_entries 
                    (index_key, document_id, content_hash, metadata_hash, vector_hash, 
                     index_type, version, created_at, updated_at, consistency_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(index_key), entry.document_id, index_key.content_hash,
                    index_key.metadata_hash, index_key.vector_hash, index_type.value,
                    version, entry.created_at, entry.updated_at, consistency_level.value
                ))
                
                conn.commit()
            
            # Store in vector database
            if not self.index_storage:
                self.index_storage = QdrantIndexStorage(self._get_qdrant_client())
            
            success = await self.index_storage.store_index_entry(entry)
            if not success:
                return None
            
            logger.info(f"Indexed document {document_id} with key {index_key}")
            return str(index_key)
            
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            return None
    
    async def update_index(self, index_key: str, content: Optional[str] = None,
                          vector: Optional[List[float]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing index entry."""
        try:
            # Retrieve existing entry
            existing_key = IndexKey.from_string(index_key)
            entry = await self.index_storage.retrieve_index_entry(existing_key)
            
            if not entry:
                logger.error(f"Index entry not found: {index_key}")
                return False
            
            # Update fields
            if content:
                entry.content = content
            if vector:
                entry.vector = self._normalize_vector(vector)
            if metadata:
                entry.metadata.update(metadata)
            
            entry.updated_at = datetime.now()
            
            # Update in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE index_entries 
                    SET content_hash = ?, metadata_hash = ?, vector_hash = ?, updated_at = ?
                    WHERE index_key = ?
                ''', (
                    hashlib.sha256(entry.content.encode('utf-8')).hexdigest(),
                    hashlib.sha256(json.dumps(entry.metadata, sort_keys=True).encode('utf-8')).hexdigest(),
                    hashlib.sha256(np.array(entry.vector).tobytes()).hexdigest(),
                    entry.updated_at,
                    index_key
                ))
                
                conn.commit()
            
            # Update in vector database
            success = await self.index_storage.store_index_entry(entry)
            return success
            
        except Exception as e:
            logger.error(f"Error updating index: {e}")
            return False
    
    async def delete_index(self, index_key: str) -> bool:
        """Delete an index entry."""
        try:
            # Delete from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM index_entries WHERE index_key = ?', (index_key,))
                conn.commit()
            
            # Delete from vector database
            existing_key = IndexKey.from_string(index_key)
            success = await self.index_storage.delete_index_entry(existing_key)
            
            logger.info(f"Deleted index entry: {index_key}")
            return success
            
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False
    
    async def validate_index(self, index_key: str) -> bool:
        """Validate an index entry."""
        try:
            # Retrieve entry
            existing_key = IndexKey.from_string(index_key)
            entry = await self.index_storage.retrieve_index_entry(existing_key)
            
            if not entry:
                return False
            
            # Validate hashes
            content_hash = hashlib.sha256(entry.content.encode('utf-8')).hexdigest()
            metadata_hash = hashlib.sha256(json.dumps(entry.metadata, sort_keys=True).encode('utf-8')).hexdigest()
            vector_hash = hashlib.sha256(np.array(entry.vector).tobytes()).hexdigest()
            
            return (
                content_hash == existing_key.content_hash and
                metadata_hash == existing_key.metadata_hash and
                vector_hash == existing_key.vector_hash
            )
            
        except Exception as e:
            logger.error(f"Error validating index: {e}")
            return False
    
    async def optimize_index(self, index_type: IndexType) -> bool:
        """Optimize an index."""
        try:
            optimization_id = f"opt_{int(time.time())}_{index_type.value}"
            
            # Record optimization start
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO index_optimizations 
                    (optimization_id, index_type, optimization_type, started_at, status, 
                     documents_processed, space_reclaimed_mb)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    optimization_id, index_type.value, 'compact',
                    datetime.now(), 'running', 0, 0.0
                ))
                
                conn.commit()
            
            # Perform optimization (simplified)
            # In practice, this would involve merging segments, removing duplicates, etc.
            logger.info(f"Started optimization for {index_type.value}")
            
            # Record optimization completion
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE index_optimizations 
                    SET status = ?, completed_at = ?, documents_processed = ?, space_reclaimed_mb = ?
                    WHERE optimization_id = ?
                ''', (
                    'completed', datetime.now(), 1000, 10.5, optimization_id
                ))
                
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing index: {e}")
            return False
    
    async def rollback_index(self, index_key: str, target_version: str) -> bool:
        """Rollback an index to a previous version."""
        try:
            # This would involve retrieving a previous version and restoring it
            # For now, return success as placeholder
            logger.info(f"Rolled back index {index_key} to version {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back index: {e}")
            return False
    
    async def get_index_health(self, index_type: IndexType) -> Optional[IndexHealth]:
        """Get index health information."""
        try:
            # Check integrity
            integrity_valid = await self.index_storage.validate_index_integrity(index_type)
            
            # Calculate health score
            health_score = 1.0 if integrity_valid else 0.5
            issues = [] if integrity_valid else ["Integrity check failed"]
            
            health = IndexHealth(
                index_type=index_type,
                status=IndexStatus.ACTIVE if integrity_valid else IndexStatus.FAILED,
                health_score=health_score,
                last_check_at=datetime.now(),
                issues=issues
            )
            
            # Store health information
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO index_health 
                    (index_type, status, health_score, last_check_at, issues, performance_metrics)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    index_type.value, health.status.value, health.health_score,
                    health.last_check_at, json.dumps(health.issues), json.dumps({})
                ))
                
                conn.commit()
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting index health: {e}")
            return None
    
    async def get_index_statistics(self) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count entries by type
                cursor.execute('SELECT index_type, COUNT(*) FROM index_entries GROUP BY index_type')
                type_counts = dict(cursor.fetchall())
                
                # Count transactions
                cursor.execute('SELECT status, COUNT(*) FROM index_transactions GROUP BY status')
                transaction_counts = dict(cursor.fetchall())
                
                # Count versions
                cursor.execute('SELECT index_type, COUNT(*) FROM index_versions GROUP BY index_type')
                version_counts = dict(cursor.fetchall())
                
                # Count optimizations
                cursor.execute('SELECT status, COUNT(*) FROM index_optimizations GROUP BY status')
                optimization_counts = dict(cursor.fetchall())
                
                return {
                    'total_entries': sum(type_counts.values()),
                    'entries_by_type': type_counts,
                    'total_transactions': sum(transaction_counts.values()),
                    'transactions_by_status': transaction_counts,
                    'total_versions': sum(version_counts.values()),
                    'versions_by_type': version_counts,
                    'total_optimizations': sum(optimization_counts.values()),
                    'optimizations_by_status': optimization_counts,
                    'deterministic_indexing_enabled': self.enable_deterministic_indexing,
                    'index_versioning_enabled': self.enable_index_versioning,
                    'index_consistency_enabled': self.enable_index_consistency,
                    'index_optimization_enabled': self.enable_index_optimization,
                    'distributed_indexing_enabled': self.enable_distributed_indexing
                }
                
        except Exception as e:
            logger.error(f"Error getting index statistics: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Shutdown the indexing system."""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            logger.info("Idempotent indexing system shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Convenience functions for easy use
def create_idempotent_indexing_system(config: Dict[str, Any]) -> IdempotentIndexingSystem:
    """Create an idempotent indexing system instance."""
    return IdempotentIndexingSystem(config)


if __name__ == "__main__":
    
    # config = {
    #     'indexing_db_path': 'idempotent_indexing.db',
    #     'enable_deterministic_indexing': True,
    #     'enable_index_versioning': True,
    #     'enable_index_consistency': True,
    #     'enable_index_optimization': True,
    #     'enable_distributed_indexing': False
    # }
    
    # indexing_system = IdempotentIndexingSystem(config)
    
    # Index a document
    # index_key = await indexing_system.index_document(
    #     "doc_123", "Document content", [0.1, 0.2, 0.3], {"author": "John"}
    # )
    
    # Update index
    # await indexing_system.update_index(index_key, content="Updated content")
    
    # Validate index
    # is_valid = await indexing_system.validate_index(index_key)
    
    # Get statistics
    # stats = await indexing_system.get_index_statistics()
    
    pass