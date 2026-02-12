"""
Enhanced Document Manager with production-ready features.

Includes: incremental ingestion, deduplication, stable IDs, rich metadata, 
versioning, and idempotent indexing.
"""

import os
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from config import MARKDOWN_DIR, PARENT_STORE_PATH, QDRANT_DB_PATH
from multi_format_ingestion import MultiFormatIngestionPipeline, DocumentTypeDetector

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Rich metadata for documents."""
    document_id: str
    original_filename: str
    file_path: str
    file_size: int
    file_hash: str
    document_type: str
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    tags: List[str] = None
    version: int = 1
    ingestion_date: str = ""
    source_url: Optional[str] = None
    processing_status: str = "completed"
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if not self.ingestion_date:
            self.ingestion_date = datetime.now().isoformat()

class EnhancedDocumentManager:
    """Production-ready document manager with advanced features."""
    
    def __init__(self, rag_system, enable_deduplication=True, enable_versioning=True):
        self.rag_system = rag_system
        self.markdown_dir = Path(MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        
        self.ingestion_pipeline = MultiFormatIngestionPipeline(self.markdown_dir)
        
        # Metadata storage
        self.metadata_dir = self.markdown_dir / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Deduplication settings
        self.enable_deduplication = enable_deduplication
        self.enable_versioning = enable_versioning
        
        # Hash to document ID mapping for deduplication
        self.hash_index_file = self.metadata_dir / "hash_index.json"
        self.hash_index = self._load_hash_index()
        
        # Document registry
        self.document_registry_file = self.metadata_dir / "document_registry.json"
        self.document_registry = self._load_document_registry()
    
    def _load_hash_index(self) -> Dict[str, str]:
        """Load hash to document ID mapping."""
        if self.hash_index_file.exists():
            try:
                with open(self.hash_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load hash index: {e}")
        return {}
    
    def _save_hash_index(self):
        """Save hash to document ID mapping."""
        try:
            with open(self.hash_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.hash_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save hash index: {e}")
    
    def _load_document_registry(self) -> Dict[str, Dict]:
        """Load document registry."""
        if self.document_registry_file.exists():
            try:
                with open(self.document_registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load document registry: {e}")
        return {}
    
    def _save_document_registry(self):
        """Save document registry."""
        try:
            with open(self.document_registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save document registry: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def _generate_document_id(self, original_filename: str, file_hash: str) -> str:
        """Generate stable document ID."""
        # Combine filename and hash for uniqueness
        combined = f"{original_filename}_{file_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic file metadata."""
        stat = file_path.stat()
        return {
            'size': stat.st_size,
            'creation_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modification_time': datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    
    def _check_duplicate(self, file_hash: str) -> Optional[str]:
        """Check if document is a duplicate and return existing document ID."""
        if not self.enable_deduplication:
            return None
        
        return self.hash_index.get(file_hash)
    
    def _update_version(self, existing_doc_id: str, new_metadata: DocumentMetadata) -> DocumentMetadata:
        """Update document version."""
        if not self.enable_versioning:
            return new_metadata
        
        existing_doc = self.document_registry.get(existing_doc_id, {})
        current_version = existing_doc.get('version', 1)
        
        # Create backup of current version
        self._backup_document(existing_doc_id)
        
        # Update version
        new_metadata.version = current_version + 1
        new_metadata.processing_status = "updated"
        
        return new_metadata
    
    def _backup_document(self, document_id: str):
        """Backup current version of document."""
        try:
            backup_dir = self.metadata_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            # Backup markdown file
            current_md = self.markdown_dir / f"{document_id}.md"
            if current_md.exists():
                backup_md = backup_dir / f"{document_id}_v{self.document_registry[document_id]['version']}.md"
                current_md.rename(backup_md)
            
            # Backup metadata
            backup_meta = backup_dir / f"{document_id}_v{self.document_registry[document_id]['version']}_meta.json"
            with open(backup_meta, 'w', encoding='utf-8') as f:
                json.dump(self.document_registry[document_id], f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to backup document {document_id}: {e}")
    
    def add_document(self, file_path: Path, metadata: Optional[Dict] = None, 
                    force_reprocess: bool = False) -> Tuple[bool, str, Optional[DocumentMetadata]]:
        """Add a single document with full production features."""
        try:
            if not file_path.exists():
                return False, f"File not found: {file_path}", None
            
            # Calculate file hash for deduplication
            file_hash = self._calculate_file_hash(file_path)
            if not file_hash:
                return False, f"Failed to calculate hash for {file_path}", None
            
            # Check for duplicates
            existing_doc_id = self._check_duplicate(file_hash)
            if existing_doc_id and not force_reprocess:
                logger.info(f"Duplicate document found: {file_path} (ID: {existing_doc_id})")
                return True, f"Duplicate skipped (ID: {existing_doc_id})", None
            
            # Generate document ID
            doc_id = self._generate_document_id(file_path.name, file_hash)
            
            # Create metadata
            file_meta = self._get_file_metadata(file_path)
            doc_metadata = DocumentMetadata(
                document_id=doc_id,
                original_filename=file_path.name,
                file_path=str(file_path),
                file_size=file_meta['size'],
                file_hash=file_hash,
                document_type=DocumentTypeDetector.detect_format(file_path) or 'unknown',
                creation_date=file_meta['creation_time'],
                modification_date=file_meta['modification_time'],
                **(metadata or {})
            )
            
            # Handle versioning
            if existing_doc_id and force_reprocess:
                doc_metadata = self._update_version(existing_doc_id, doc_metadata)
                doc_id = existing_doc_id  # Keep same ID for versioning
            
            # Update hash index
            self.hash_index[file_hash] = doc_id
            self._save_hash_index()
            
            # Process document
            success, processed_path = self.ingestion_pipeline.ingest_file(file_path)
            if not success or not processed_path:
                doc_metadata.processing_status = "failed"
                doc_metadata.error_message = "Failed to process document"
                self._save_document_metadata(doc_metadata)
                return False, "Failed to process document", doc_metadata
            
            # Process markdown with RAG system
            md_path = Path(processed_path)
            parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_path)
            
            if not child_chunks:
                doc_metadata.processing_status = "failed"
                doc_metadata.error_message = "No content extracted"
                self._save_document_metadata(doc_metadata)
                return False, "No content extracted", doc_metadata
            
            # Add to vector database
            collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
            
            # Add document ID to chunk metadata
            for chunk in child_chunks:
                chunk.metadata['document_id'] = doc_id
                chunk.metadata['version'] = doc_metadata.version
            
            collection.add_documents(child_chunks)
            self.rag_system.parent_store.save_many(parent_chunks)
            
            # Save metadata
            self._save_document_metadata(doc_metadata)
            
            logger.info(f"Successfully added document: {file_path} (ID: {doc_id}, Version: {doc_metadata.version})")
            return True, f"Successfully added (ID: {doc_id}, Version: {doc_metadata.version})", doc_metadata
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return False, str(e), None
    
    def _save_document_metadata(self, metadata: DocumentMetadata):
        """Save document metadata."""
        try:
            # Save to registry
            self.document_registry[metadata.document_id] = asdict(metadata)
            self._save_document_registry()
            
            # Save individual metadata file
            meta_file = self.metadata_dir / f"{metadata.document_id}_meta.json"
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metadata for {metadata.document_id}: {e}")
    
    def add_web_page(self, url: str, metadata: Optional[Dict] = None) -> Tuple[bool, str, Optional[DocumentMetadata]]:
        """Add a web page with full production features."""
        try:
            # Use URL as unique identifier for deduplication
            url_hash = hashlib.md5(url.encode()).hexdigest()
            
            # Check for duplicates
            existing_doc_id = self._check_duplicate(url_hash)
            if existing_doc_id:
                logger.info(f"Duplicate web page found: {url} (ID: {existing_doc_id})")
                return True, f"Duplicate skipped (ID: {existing_doc_id})", None
            
            # Generate document ID
            doc_id = self._generate_document_id(f"web_{url}", url_hash)
            
            # Create metadata
            doc_metadata = DocumentMetadata(
                document_id=doc_id,
                original_filename=f"web_{urlparse(url).netloc}",
                file_path=url,
                file_size=0,  # Web pages don't have file size
                file_hash=url_hash,
                document_type='html',
                source_url=url,
                **(metadata or {})
            )
            
            # Update hash index
            self.hash_index[url_hash] = doc_id
            self._save_hash_index()
            
            # Process web page
            success, processed_path = self.ingestion_pipeline.ingest_web_page(url, metadata)
            if not success or not processed_path:
                doc_metadata.processing_status = "failed"
                doc_metadata.error_message = "Failed to fetch web page"
                self._save_document_metadata(doc_metadata)
                return False, "Failed to fetch web page", doc_metadata
            
            # Process markdown with RAG system
            md_path = Path(processed_path)
            parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_path)
            
            if not child_chunks:
                doc_metadata.processing_status = "failed"
                doc_metadata.error_message = "No content extracted"
                self._save_document_metadata(doc_metadata)
                return False, "No content extracted", doc_metadata
            
            # Add to vector database
            collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
            
            # Add document ID to chunk metadata
            for chunk in child_chunks:
                chunk.metadata['document_id'] = doc_id
                chunk.metadata['version'] = doc_metadata.version
            
            collection.add_documents(child_chunks)
            self.rag_system.parent_store.save_many(parent_chunks)
            
            # Save metadata
            self._save_document_metadata(doc_metadata)
            
            logger.info(f"Successfully added web page: {url} (ID: {doc_id})")
            return True, f"Successfully added (ID: {doc_id})", doc_metadata
            
        except Exception as e:
            logger.error(f"Error adding web page {url}: {e}")
            return False, str(e), None
    
    def add_directory(self, directory_path: Path, recursive: bool = True, 
                     progress_callback=None) -> Dict[str, Any]:
        """Add all documents from a directory with production features."""
        try:
            directory_path = Path(directory_path)
            if not directory_path.exists():
                return {'error': f"Directory not found: {directory_path}"}
            
            # Get all supported files
            if recursive:
                files = [f for f in directory_path.rglob('*') if f.is_file() and DocumentTypeDetector.is_supported(f)]
            else:
                files = [f for f in directory_path.iterdir() if f.is_file() and DocumentTypeDetector.is_supported(f)]
            
            total_files = len(files)
            results = {
                'total_files': total_files,
                'processed': 0,
                'added': 0,
                'skipped': 0,
                'failed': 0,
                'duplicates': 0,
                'details': []
            }
            
            for i, file_path in enumerate(files):
                if progress_callback:
                    progress = (i + 1) / total_files
                    progress_callback(progress, f"Processing {file_path.name}")
                
                success, message, metadata = self.add_document(file_path)
                results['processed'] += 1
                
                if success:
                    if "Duplicate skipped" in message:
                        results['duplicates'] += 1
                    else:
                        results['added'] += 1
                else:
                    results['failed'] += 1
                
                results['details'].append({
                    'file': str(file_path),
                    'success': success,
                    'message': message,
                    'document_id': metadata.document_id if metadata else None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error adding directory {directory_path}: {e}")
            return {'error': str(e)}
    
    def get_document_info(self, document_id: str) -> Optional[Dict]:
        """Get information about a specific document."""
        return self.document_registry.get(document_id)
    
    def list_documents(self, document_type: Optional[str] = None, 
                      tags: Optional[List[str]] = None) -> List[Dict]:
        """List documents with optional filtering."""
        documents = list(self.document_registry.values())
        
        if document_type:
            documents = [d for d in documents if d.get('document_type') == document_type]
        
        if tags:
            documents = [d for d in documents if any(tag in d.get('tags', []) for tag in tags)]
        
        return sorted(documents, key=lambda x: x.get('ingestion_date', ''), reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get document statistics."""
        total_docs = len(self.document_registry)
        by_type = {}
        by_status = {}
        
        for doc in self.document_registry.values():
            doc_type = doc.get('document_type', 'unknown')
            status = doc.get('processing_status', 'unknown')
            
            by_type[doc_type] = by_type.get(doc_type, 0) + 1
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            'total_documents': total_docs,
            'by_type': by_type,
            'by_status': by_status,
            'total_duplicates_checked': len(self.hash_index),
            'metadata_files': len(list(self.metadata_dir.glob('*_meta.json')))
        }
    
    def clear_all(self):
        """Clear all documents and metadata."""
        try:
            # Clear markdown files
            if self.markdown_dir.exists():
                import shutil
                shutil.rmtree(self.markdown_dir)
                self.markdown_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear metadata
            if self.metadata_dir.exists():
                import shutil
                shutil.rmtree(self.metadata_dir)
                self.metadata_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear RAG system data
            self.rag_system.parent_store.clear_store()
            self.rag_system.vector_db.delete_collection(self.rag_system.collection_name)
            self.rag_system.vector_db.create_collection(self.rag_system.collection_name)
            
            # Reset indexes
            self.hash_index = {}
            self.document_registry = {}
            
            logger.info("Cleared all documents and metadata")
            
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")


# Convenience functions for easy use
def create_enhanced_document_manager(rag_system, **kwargs):
    """Create an enhanced document manager."""
    return EnhancedDocumentManager(rag_system, **kwargs)


if __name__ == "__main__":
    
    # rag_system = RAGSystem()  # Assume RAGSystem is imported
    # doc_manager = EnhancedDocumentManager(rag_system)
    
    # Add a document
    # success, message, metadata = doc_manager.add_document(Path("path/to/document.pdf"))
    
    # Add a web page
    # success, message, metadata = doc_manager.add_web_page("https://example.com")
    
    # Get statistics
    # stats = doc_manager.get_statistics()
    pass