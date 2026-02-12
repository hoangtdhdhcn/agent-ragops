"""
Advanced Deduplication Engine for Production RAG System

Provides comprehensive deduplication capabilities including exact deduplication,
semantic similarity detection, fuzzy matching, and cross-document deduplication.
"""

import asyncio
import hashlib
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

from config import MARKDOWN_DIR, PARENT_STORE_PATH, QDRANT_DB_PATH
from enhanced_document_manager import EnhancedDocumentManager, DocumentMetadata
from multi_format_ingestion import DocumentTypeDetector

logger = logging.getLogger(__name__)

@dataclass
class DuplicateRecord:
    """Record of a duplicate detection."""
    document_id: str
    duplicate_of: str
    similarity_score: float
    detection_method: str  # 'exact', 'semantic', 'fuzzy'
    detected_at: datetime
    resolved: bool = False
    resolution_method: Optional[str] = None
    resolution_details: Optional[Dict] = None

@dataclass
class DeduplicationConfig:
    """Configuration for deduplication engine."""
    # Exact deduplication
    enable_exact_dedup: bool = True
    
    # Semantic similarity
    enable_semantic_dedup: bool = True
    semantic_threshold: float = 0.85  # Minimum similarity for duplicate
    semantic_model: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Fuzzy matching
    enable_fuzzy_dedup: bool = True
    fuzzy_threshold: float = 0.90
    
    # Performance settings
    batch_size: int = 100
    max_documents_for_similarity: int = 10000
    similarity_index_update_interval: int = 3600  # seconds
    
    # Resolution settings
    auto_resolve_duplicates: bool = True
    resolution_strategy: str = "keep_newest"  # 'keep_newest', 'keep_oldest', 'keep_largest', 'manual'
    
    # Storage settings
    enable_similarity_index: bool = True
    similarity_index_path: str = "deduplication_similarity_index.faiss"
    metadata_db_path: str = "deduplication_metadata.db"

class SimilarityIndex:
    """Efficient similarity index for large-scale deduplication."""
    
    def __init__(self, config: DeduplicationConfig):
        self.config = config
        self.index = None
        self.document_ids = []  # Maps index position to document ID
        self.embeddings_cache = {}  # Cache for embeddings
        self.is_built = False
        self.last_update = None
        
        # Initialize database for metadata
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for deduplication metadata."""
        self.conn = sqlite3.connect(self.config.metadata_db_path)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS duplicate_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                duplicate_of TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                detection_method TEXT NOT NULL,
                detected_at TIMESTAMP NOT NULL,
                resolved BOOLEAN DEFAULT 0,
                resolution_method TEXT,
                resolution_details TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_embeddings (
                document_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        ''')
        
        self.conn.commit()
    
    def add_document(self, document_id: str, embedding: np.ndarray):
        """Add a document to the similarity index."""
        # Cache embedding
        self.embeddings_cache[document_id] = embedding
        
        # Store in database
        cursor = self.conn.cursor()
        embedding_blob = pickle.dumps(embedding)
        cursor.execute(
            "INSERT OR REPLACE INTO document_embeddings (document_id, embedding, updated_at) VALUES (?, ?, ?)",
            (document_id, embedding_blob, datetime.now())
        )
        self.conn.commit()
        
        # Add to FAISS index
        if self.index is None:
            dimension = embedding.shape[0]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        self.document_ids.append(document_id)
        self.is_built = True
        self.last_update = datetime.now()
    
    def find_similar(self, embedding: np.ndarray, threshold: float) -> List[Tuple[str, float]]:
        """Find documents similar to the given embedding."""
        if not self.is_built or self.index.ntotal == 0:
            return []
        
        # Search in FAISS index
        embedding = embedding.reshape(1, -1).astype('float32')
        similarities, indices = self.index.search(embedding, self.index.ntotal)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity >= threshold and idx < len(self.document_ids):
                document_id = self.document_ids[idx]
                results.append((document_id, float(similarity)))
        
        return results
    
    def get_embedding(self, document_id: str) -> Optional[np.ndarray]:
        """Get embedding for a document."""
        # Check cache first
        if document_id in self.embeddings_cache:
            return self.embeddings_cache[document_id]
        
        # Check database
        cursor = self.conn.cursor()
        cursor.execute("SELECT embedding FROM document_embeddings WHERE document_id = ?", (document_id,))
        result = cursor.fetchone()
        
        if result:
            embedding = pickle.loads(result[0])
            self.embeddings_cache[document_id] = embedding
            return embedding
        
        return None
    
    def remove_document(self, document_id: str):
        """Remove a document from the similarity index."""
        if document_id in self.embeddings_cache:
            del self.embeddings_cache[document_id]
        
        # Remove from database
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM document_embeddings WHERE document_id = ?", (document_id,))
        self.conn.commit()
        
        # Rebuild index (simplified approach)
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild the FAISS index from database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT document_id, embedding FROM document_embeddings")
        results = cursor.fetchall()
        
        if not results:
            self.index = None
            self.document_ids = []
            self.is_built = False
            return
        
        # Rebuild index
        embeddings = []
        document_ids = []
        
        for document_id, embedding_blob in results:
            embedding = pickle.loads(embedding_blob)
            embeddings.append(embedding)
            document_ids.append(document_id)
        
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings_array)
        self.document_ids = document_ids
        self.is_built = True
    
    def save_index(self):
        """Save the similarity index to disk."""
        if self.index and self.config.enable_similarity_index:
            faiss.write_index(self.index, self.config.similarity_index_path)
    
    def load_index(self):
        """Load the similarity index from disk."""
        if self.config.enable_similarity_index and Path(self.config.similarity_index_path).exists():
            self.index = faiss.read_index(self.config.similarity_index_path)
            self.is_built = True
            self._load_document_ids()
    
    def _load_document_ids(self):
        """Load document IDs from database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT document_id FROM document_embeddings ORDER BY updated_at")
        results = cursor.fetchall()
        self.document_ids = [row[0] for row in results]
    
    def close(self):
        """Close the similarity index and database connection."""
        if self.index:
            self.save_index()
        self.conn.close()

class EmbeddingManager:
    """Manages document embeddings for semantic similarity."""
    
    def __init__(self, config: DeduplicationConfig):
        self.config = config
        self.model = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.semantic_model)
            self.is_initialized = True
            logger.info(f"Initialized semantic similarity model: {self.config.semantic_model}")
        except ImportError:
            logger.warning("Sentence transformers not available. Semantic deduplication disabled.")
            self.is_initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.is_initialized = False
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text."""
        if not self.is_initialized or not self.model:
            return None
        
        try:
            # Truncate very long texts to avoid memory issues
            max_length = 2000
            if len(text) > max_length:
                text = text[:max_length]
            
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding for text: {e}")
            return None
    
    def get_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        try:
            # Normalize embeddings
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

class FuzzyMatcher:
    """Handles fuzzy matching for near-duplicates."""
    
    def __init__(self, config: DeduplicationConfig):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        self.is_trained = False
        self.document_texts = {}
    
    def add_document(self, document_id: str, text: str):
        """Add document text for fuzzy matching."""
        self.document_texts[document_id] = text
    
    def find_similar(self, document_id: str, text: str, threshold: float) -> List[Tuple[str, float]]:
        """Find similar documents using fuzzy matching."""
        if not self.document_texts:
            return []
        
        # Add current document to comparison set
        all_texts = list(self.document_texts.values()) + [text]
        all_ids = list(self.document_texts.keys()) + [document_id]
        
        try:
            # Vectorize all texts
            if not self.is_trained:
                tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                self.is_trained = True
            else:
                tfidf_matrix = self.vectorizer.transform(all_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
            
            results = []
            for i, similarity in enumerate(similarities[0]):
                if similarity >= threshold:
                    results.append((all_ids[i], float(similarity)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in fuzzy matching: {e}")
            return []

class DeduplicationEngine:
    """Main deduplication engine with multi-level detection."""
    
    def __init__(self, config: DeduplicationConfig, enhanced_doc_manager: EnhancedDocumentManager):
        self.config = config
        self.doc_manager = enhanced_doc_manager
        
        # Initialize components
        self.similarity_index = SimilarityIndex(config)
        self.embedding_manager = EmbeddingManager(config)
        self.fuzzy_matcher = FuzzyMatcher(config)
        
        # State
        self.is_initialized = False
        self.duplicate_records = []
        self.resolution_callbacks = []
        
        # Threading
        self.processing_lock = threading.Lock()
        self.update_thread = None
        self.stop_update_event = threading.Event()
    
    async def initialize(self):
        """Initialize the deduplication engine."""
        try:
            # Initialize embedding manager
            await self.embedding_manager.initialize()
            
            # Load existing similarity index
            self.similarity_index.load_index()
            
            # Load existing documents into fuzzy matcher
            self._load_existing_documents()
            
            self.is_initialized = True
            logger.info("Deduplication engine initialized successfully")
            
            # Start background update thread
            self._start_background_update()
            
        except Exception as e:
            logger.error(f"Failed to initialize deduplication engine: {e}")
            raise
    
    def _load_existing_documents(self):
        """Load existing documents into the deduplication system."""
        try:
            documents = self.doc_manager.list_documents()
            
            for doc_info in documents:
                document_id = doc_info['document_id']
                
                # Get document content
                md_path = Path(self.doc_manager.markdown_dir) / f"{document_id}.md"
                if md_path.exists():
                    with open(md_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Add to fuzzy matcher
                    self.fuzzy_matcher.add_document(document_id, content)
                    
                    # Add to similarity index if semantic dedup is enabled
                    if self.config.enable_semantic_dedup and self.embedding_manager.is_initialized:
                        embedding = self.embedding_manager.get_embedding(content)
                        if embedding is not None:
                            self.similarity_index.add_document(document_id, embedding)
            
            logger.info(f"Loaded {len(documents)} existing documents into deduplication engine")
            
        except Exception as e:
            logger.error(f"Error loading existing documents: {e}")
    
    def _start_background_update(self):
        """Start background thread for index updates."""
        if self.config.similarity_index_update_interval > 0:
            self.update_thread = threading.Thread(target=self._background_update_loop, daemon=True)
            self.update_thread.start()
    
    def _background_update_loop(self):
        """Background loop for updating similarity index."""
        while not self.stop_update_event.is_set():
            try:
                # Check if index needs updating
                if (self.similarity_index.last_update is None or 
                    (datetime.now() - self.similarity_index.last_update).total_seconds() > 
                    self.config.similarity_index_update_interval):
                    
                    self.similarity_index.save_index()
                    logger.debug("Updated similarity index")
                
                # Sleep until next update
                self.stop_update_event.wait(self.config.similarity_index_update_interval)
                
            except Exception as e:
                logger.error(f"Error in background update loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    async def check_duplicates(self, document_id: str, content: str, 
                             metadata: Optional[DocumentMetadata] = None) -> List[DuplicateRecord]:
        """Check for duplicates of a document."""
        if not self.is_initialized:
            raise RuntimeError("Deduplication engine not initialized")
        
        duplicates = []
        
        async with self.processing_lock:
            # 1. Exact deduplication (hash-based)
            if self.config.enable_exact_dedup:
                exact_duplicates = await self._check_exact_duplicates(document_id, content)
                duplicates.extend(exact_duplicates)
            
            # 2. Semantic similarity
            if self.config.enable_semantic_dedup and self.embedding_manager.is_initialized:
                semantic_duplicates = await self._check_semantic_duplicates(document_id, content)
                duplicates.extend(semantic_duplicates)
            
            # 3. Fuzzy matching
            if self.config.enable_fuzzy_dedup:
                fuzzy_duplicates = await self._check_fuzzy_duplicates(document_id, content)
                duplicates.extend(fuzzy_duplicates)
        
        # Store duplicate records
        for duplicate in duplicates:
            self._store_duplicate_record(duplicate)
        
        # Auto-resolve if configured
        if self.config.auto_resolve_duplicates and duplicates:
            await self._auto_resolve_duplicates(duplicates, metadata)
        
        return duplicates
    
    async def _check_exact_duplicates(self, document_id: str, content: str) -> List[DuplicateRecord]:
        """Check for exact duplicates using hash comparison."""
        try:
            # Calculate hash
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # Check hash index
            existing_doc_id = self.doc_manager.hash_index.get(content_hash)
            
            if existing_doc_id and existing_doc_id != document_id:
                return [DuplicateRecord(
                    document_id=document_id,
                    duplicate_of=existing_doc_id,
                    similarity_score=1.0,
                    detection_method='exact',
                    detected_at=datetime.now()
                )]
            
            return []
            
        except Exception as e:
            logger.error(f"Error checking exact duplicates: {e}")
            return []
    
    async def _check_semantic_duplicates(self, document_id: str, content: str) -> List[DuplicateRecord]:
        """Check for semantic duplicates using embeddings."""
        try:
            # Get embedding for current document
            current_embedding = self.embedding_manager.get_embedding(content)
            if current_embedding is None:
                return []
            
            # Find similar documents
            similar_docs = self.similarity_index.find_similar(
                current_embedding, 
                self.config.semantic_threshold
            )
            
            duplicates = []
            for similar_id, similarity in similar_docs:
                if similar_id != document_id:  # Don't compare with self
                    duplicates.append(DuplicateRecord(
                        document_id=document_id,
                        duplicate_of=similar_id,
                        similarity_score=similarity,
                        detection_method='semantic',
                        detected_at=datetime.now()
                    ))
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Error checking semantic duplicates: {e}")
            return []
    
    async def _check_fuzzy_duplicates(self, document_id: str, content: str) -> List[DuplicateRecord]:
        """Check for fuzzy duplicates using TF-IDF similarity."""
        try:
            # Find similar documents
            similar_docs = self.fuzzy_matcher.find_similar(
                document_id, 
                content, 
                self.config.fuzzy_threshold
            )
            
            duplicates = []
            for similar_id, similarity in similar_docs:
                if similar_id != document_id:  # Don't compare with self
                    duplicates.append(DuplicateRecord(
                        document_id=document_id,
                        duplicate_of=similar_id,
                        similarity_score=similarity,
                        detection_method='fuzzy',
                        detected_at=datetime.now()
                    ))
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Error checking fuzzy duplicates: {e}")
            return []
    
    def _store_duplicate_record(self, duplicate: DuplicateRecord):
        """Store duplicate record in database."""
        try:
            cursor = self.similarity_index.conn.cursor()
            resolution_details_json = json.dumps(duplicate.resolution_details) if duplicate.resolution_details else None
            
            cursor.execute('''
                INSERT INTO duplicate_records 
                (document_id, duplicate_of, similarity_score, detection_method, detected_at, resolved, resolution_method, resolution_details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                duplicate.document_id,
                duplicate.duplicate_of,
                duplicate.similarity_score,
                duplicate.detection_method,
                duplicate.detected_at,
                duplicate.resolved,
                duplicate.resolution_method,
                resolution_details_json
            ))
            
            self.similarity_index.conn.commit()
            self.duplicate_records.append(duplicate)
            
        except Exception as e:
            logger.error(f"Error storing duplicate record: {e}")
    
    async def _auto_resolve_duplicates(self, duplicates: List[DuplicateRecord], 
                                     metadata: Optional[DocumentMetadata]):
        """Auto-resolve duplicates based on configured strategy."""
        try:
            for duplicate in duplicates:
                resolution_method = self._determine_resolution_method(duplicate, metadata)
                
                if resolution_method == 'keep_newest':
                    await self._resolve_keep_newest(duplicate)
                elif resolution_method == 'keep_oldest':
                    await self._resolve_keep_oldest(duplicate)
                elif resolution_method == 'keep_largest':
                    await self._resolve_keep_largest(duplicate)
                elif resolution_method == 'merge':
                    await self._resolve_merge(duplicate)
                
                # Update duplicate record
                duplicate.resolved = True
                duplicate.resolution_method = resolution_method
                
                # Notify callbacks
                for callback in self.resolution_callbacks:
                    try:
                        callback(duplicate)
                    except Exception as e:
                        logger.error(f"Error in resolution callback: {e}")
        
        except Exception as e:
            logger.error(f"Error in auto-resolve duplicates: {e}")
    
    def _determine_resolution_method(self, duplicate: DuplicateRecord, 
                                   metadata: Optional[DocumentMetadata]) -> str:
        """Determine resolution method based on strategy and context."""
        if self.config.resolution_strategy != 'manual':
            return self.config.resolution_strategy
        
        # For manual resolution, return based on some logic or default
        return 'keep_newest'
    
    async def _resolve_keep_newest(self, duplicate: DuplicateRecord):
        """Resolve by keeping the newest document."""
        try:
            # Get document metadata
            current_doc = self.doc_manager.get_document_info(duplicate.document_id)
            existing_doc = self.doc_manager.get_document_info(duplicate.duplicate_of)
            
            if not current_doc or not existing_doc:
                return
            
            # Compare modification dates
            current_date = datetime.fromisoformat(current_doc.get('modification_date', ''))
            existing_date = datetime.fromisoformat(existing_doc.get('modification_date', ''))
            
            if current_date > existing_date:
                # Keep current, remove existing
                await self._remove_duplicate_document(duplicate.duplicate_of, duplicate)
            else:
                # Keep existing, remove current
                await self._remove_duplicate_document(duplicate.document_id, duplicate)
        
        except Exception as e:
            logger.error(f"Error resolving keep newest: {e}")
    
    async def _resolve_keep_oldest(self, duplicate: DuplicateRecord):
        """Resolve by keeping the oldest document."""
        try:
            # Similar to keep_newest but opposite logic
            current_doc = self.doc_manager.get_document_info(duplicate.document_id)
            existing_doc = self.doc_manager.get_document_info(duplicate.duplicate_of)
            
            if not current_doc or not existing_doc:
                return
            
            current_date = datetime.fromisoformat(current_doc.get('modification_date', ''))
            existing_date = datetime.fromisoformat(existing_doc.get('modification_date', ''))
            
            if current_date < existing_date:
                # Keep current, remove existing
                await self._remove_duplicate_document(duplicate.duplicate_of, duplicate)
            else:
                # Keep existing, remove current
                await self._remove_duplicate_document(duplicate.document_id, duplicate)
        
        except Exception as e:
            logger.error(f"Error resolving keep oldest: {e}")
    
    async def _resolve_keep_largest(self, duplicate: DuplicateRecord):
        """Resolve by keeping the document with more content."""
        try:
            # Get document sizes
            current_doc = self.doc_manager.get_document_info(duplicate.document_id)
            existing_doc = self.doc_manager.get_document_info(duplicate.duplicate_of)
            
            if not current_doc or not existing_doc:
                return
            
            current_size = current_doc.get('file_size', 0)
            existing_size = existing_doc.get('file_size', 0)
            
            if current_size > existing_size:
                # Keep current, remove existing
                await self._remove_duplicate_document(duplicate.duplicate_of, duplicate)
            else:
                # Keep existing, remove current
                await self._remove_duplicate_document(duplicate.document_id, duplicate)
        
        except Exception as e:
            logger.error(f"Error resolving keep largest: {e}")
    
    async def _resolve_merge(self, duplicate: DuplicateRecord):
        """Resolve by merging content from both documents."""
        try:
            # This is a complex operation that would need to be implemented
            # based on specific requirements for content merging
            logger.info(f"Merging documents {duplicate.document_id} and {duplicate.duplicate_of}")
            
            # For now, just keep the current document
            await self._remove_duplicate_document(duplicate.duplicate_of, duplicate)
        
        except Exception as e:
            logger.error(f"Error resolving merge: {e}")
    
    async def _remove_duplicate_document(self, document_id: str, duplicate: DuplicateRecord):
        """Remove a duplicate document."""
        try:
            # Remove from document manager
            # Note: This would need to be implemented in the document manager
            # For now, just log the action
            logger.info(f"Would remove duplicate document: {document_id}")
            
            # Remove from similarity index
            self.similarity_index.remove_document(document_id)
            
            # Remove from fuzzy matcher
            if document_id in self.fuzzy_matcher.document_texts:
                del self.fuzzy_matcher.document_texts[document_id]
        
        except Exception as e:
            logger.error(f"Error removing duplicate document {document_id}: {e}")
    
    def add_resolution_callback(self, callback: Callable[[DuplicateRecord], None]):
        """Add a callback function for duplicate resolution events."""
        self.resolution_callbacks.append(callback)
    
    def get_duplicate_stats(self) -> Dict[str, Any]:
        """Get statistics about duplicate detection."""
        try:
            cursor = self.similarity_index.conn.cursor()
            
            # Total duplicates detected
            cursor.execute("SELECT COUNT(*) FROM duplicate_records")
            total_duplicates = cursor.fetchone()[0]
            
            # Resolved duplicates
            cursor.execute("SELECT COUNT(*) FROM duplicate_records WHERE resolved = 1")
            resolved_duplicates = cursor.fetchone()[0]
            
            # Detection methods
            cursor.execute("SELECT detection_method, COUNT(*) FROM duplicate_records GROUP BY detection_method")
            detection_methods = dict(cursor.fetchall())
            
            # Resolution methods
            cursor.execute("SELECT resolution_method, COUNT(*) FROM duplicate_records WHERE resolved = 1 GROUP BY resolution_method")
            resolution_methods = dict(cursor.fetchall())
            
            return {
                'total_detected': total_duplicates,
                'resolved': resolved_duplicates,
                'unresolved': total_duplicates - resolved_duplicates,
                'detection_methods': detection_methods,
                'resolution_methods': resolution_methods,
                'resolution_rate': resolved_duplicates / max(total_duplicates, 1)
            }
        
        except Exception as e:
            logger.error(f"Error getting duplicate stats: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown the deduplication engine."""
        try:
            # Stop background update thread
            self.stop_update_event.set()
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=30)
            
            # Save similarity index
            self.similarity_index.save_index()
            
            # Close database connection
            self.similarity_index.close()
            
            logger.info("Deduplication engine shutdown complete")
        
        except Exception as e:
            logger.error(f"Error during deduplication engine shutdown: {e}")


# Convenience functions for easy use
def create_deduplication_engine(config: DeduplicationConfig, 
                               enhanced_doc_manager: EnhancedDocumentManager) -> DeduplicationEngine:
    """Create a deduplication engine instance."""
    return DeduplicationEngine(config, enhanced_doc_manager)


if __name__ == "__main__":
    
    # config = DeduplicationConfig()
    # doc_manager = EnhancedDocumentManager(rag_system)
    # dedup_engine = DeduplicationEngine(config, doc_manager)
    # await dedup_engine.initialize()
    
    # Check for duplicates
    # duplicates = await dedup_engine.check_duplicates("doc_id", "document content")
    # print(f"Found {len(duplicates)} duplicates")
    
    pass