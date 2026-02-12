"""
Deduplication Integration Module

Integrates the advanced deduplication engine with the existing EnhancedDocumentManager
to provide seamless deduplication capabilities during document ingestion.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import asdict

from config import MARKDOWN_DIR
from enhanced_document_manager import EnhancedDocumentManager, DocumentMetadata
from deduplication_engine import DeduplicationEngine, DeduplicationConfig, DuplicateRecord
from multi_format_ingestion import MultiFormatIngestionPipeline, DocumentTypeDetector

logger = logging.getLogger(__name__)

class EnhancedDocumentManagerWithDeduplication(EnhancedDocumentManager):
    """Enhanced document manager with integrated deduplication capabilities."""
    
    def __init__(self, rag_system, enable_deduplication=True, enable_versioning=True, 
                 deduplication_config: Optional[DeduplicationConfig] = None):
        """
        Initialize enhanced document manager with deduplication.
        
        Args:
            rag_system: The RAG system instance
            enable_deduplication: Whether to enable deduplication
            enable_versioning: Whether to enable versioning
            deduplication_config: Deduplication configuration
        """
        # Initialize parent class
        super().__init__(rag_system, enable_deduplication, enable_versioning)
        
        # Deduplication settings
        self.enable_deduplication = enable_deduplication
        self.deduplication_config = deduplication_config or DeduplicationConfig()
        
        # Initialize deduplication engine
        self.deduplication_engine = None
        self.is_deduplication_initialized = False
        
        # Deduplication callbacks
        self.deduplication_callbacks = []
    
    async def initialize_deduplication(self):
        """Initialize the deduplication engine."""
        if not self.enable_deduplication:
            logger.info("Deduplication is disabled")
            return
        
        try:
            self.deduplication_engine = DeduplicationEngine(
                self.deduplication_config, 
                self
            )
            await self.deduplication_engine.initialize()
            self.is_deduplication_initialized = True
            
            # Add resolution callback
            self.deduplication_engine.add_resolution_callback(self._on_duplicate_resolved)
            
            logger.info("Deduplication engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize deduplication engine: {e}")
            self.is_deduplication_initialized = False
    
    def add_deduplication_callback(self, callback: callable):
        """Add a callback for deduplication events."""
        self.deduplication_callbacks.append(callback)
    
    def _on_duplicate_resolved(self, duplicate: DuplicateRecord):
        """Handle duplicate resolution events."""
        for callback in self.deduplication_callbacks:
            try:
                callback(duplicate)
            except Exception as e:
                logger.error(f"Error in deduplication callback: {e}")
    
    async def add_document(self, file_path: Path, metadata: Optional[Dict] = None, 
                          force_reprocess: bool = False) -> Tuple[bool, str, Optional[DocumentMetadata]]:
        """
        Add a document with integrated deduplication checking.
        
        This method overrides the parent method to add deduplication checking
        before and after document processing.
        """
        try:
            # Step 1: Process document (same as parent method)
            success, message, doc_metadata = await super().add_document(
                file_path, metadata, force_reprocess
            )
            
            if not success or not doc_metadata:
                return success, message, doc_metadata
            
            # Step 2: Check for duplicates if deduplication is enabled
            if self.enable_deduplication and self.is_deduplication_initialized:
                # Get document content for deduplication checking
                md_path = self.markdown_dir / f"{doc_metadata.document_id}.md"
                if md_path.exists():
                    with open(md_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for duplicates
                    duplicates = await self.deduplication_engine.check_duplicates(
                        doc_metadata.document_id, 
                        content, 
                        doc_metadata
                    )
                    
                    if duplicates:
                        # Notify callbacks about duplicates found
                        for callback in self.deduplication_callbacks:
                            try:
                                callback({
                                    'type': 'duplicates_found',
                                    'document_id': doc_metadata.document_id,
                                    'duplicates': [asdict(d) for d in duplicates],
                                    'count': len(duplicates)
                                })
                            except Exception as e:
                                logger.error(f"Error in duplicates found callback: {e}")
                        
                        # Update message to include duplicate information
                        message += f" | Found {len(duplicates)} duplicates"
            
            return success, message, doc_metadata
            
        except Exception as e:
            logger.error(f"Error adding document with deduplication: {e}")
            return False, str(e), None
    
    async def add_web_page(self, url: str, metadata: Optional[Dict] = None) -> Tuple[bool, str, Optional[DocumentMetadata]]:
        """
        Add a web page with integrated deduplication checking.
        """
        try:
            # Step 1: Process web page (same as parent method)
            success, message, doc_metadata = await super().add_web_page(url, metadata)
            
            if not success or not doc_metadata:
                return success, message, doc_metadata
            
            # Step 2: Check for duplicates if deduplication is enabled
            if self.enable_deduplication and self.is_deduplication_initialized:
                # Get document content for deduplication checking
                md_path = self.markdown_dir / f"{doc_metadata.document_id}.md"
                if md_path.exists():
                    with open(md_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for duplicates
                    duplicates = await self.deduplication_engine.check_duplicates(
                        doc_metadata.document_id, 
                        content, 
                        doc_metadata
                    )
                    
                    if duplicates:
                        # Notify callbacks about duplicates found
                        for callback in self.deduplication_callbacks:
                            try:
                                callback({
                                    'type': 'duplicates_found',
                                    'document_id': doc_metadata.document_id,
                                    'duplicates': [asdict(d) for d in duplicates],
                                    'count': len(duplicates)
                                })
                            except Exception as e:
                                logger.error(f"Error in duplicates found callback: {e}")
                        
                        # Update message to include duplicate information
                        message += f" | Found {len(duplicates)} duplicates"
            
            return success, message, doc_metadata
            
        except Exception as e:
            logger.error(f"Error adding web page with deduplication: {e}")
            return False, str(e), None
    
    def get_deduplication_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        if not self.is_deduplication_initialized or not self.deduplication_engine:
            return {'error': 'Deduplication engine not initialized'}
        
        try:
            return self.deduplication_engine.get_duplicate_stats()
        except Exception as e:
            logger.error(f"Error getting deduplication stats: {e}")
            return {'error': str(e)}
    
    def get_duplicate_details(self, document_id: str) -> List[Dict[str, Any]]:
        """Get details about duplicates for a specific document."""
        if not self.is_deduplication_initialized or not self.deduplication_engine:
            return []
        
        try:
            cursor = self.deduplication_engine.similarity_index.conn.cursor()
            
            # Get duplicates for this document
            cursor.execute('''
                SELECT * FROM duplicate_records 
                WHERE document_id = ? OR duplicate_of = ?
                ORDER BY detected_at DESC
            ''', (document_id, document_id))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'document_id': row[1],
                    'duplicate_of': row[2],
                    'similarity_score': row[3],
                    'detection_method': row[4],
                    'detected_at': row[5],
                    'resolved': bool(row[6]),
                    'resolution_method': row[7],
                    'resolution_details': row[8]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting duplicate details: {e}")
            return []
    
    async def resolve_duplicate_manually(self, duplicate_id: int, resolution_method: str, 
                                       resolution_details: Optional[Dict] = None) -> bool:
        """Manually resolve a duplicate record."""
        if not self.is_deduplication_initialized or not self.deduplication_engine:
            return False
        
        try:
            cursor = self.deduplication_engine.similarity_index.conn.cursor()
            
            # Update duplicate record
            resolution_details_json = None
            if resolution_details:
                import json
                resolution_details_json = json.dumps(resolution_details)
            
            cursor.execute('''
                UPDATE duplicate_records 
                SET resolved = 1, resolution_method = ?, resolution_details = ?
                WHERE id = ?
            ''', (resolution_method, resolution_details_json, duplicate_id))
            
            self.deduplication_engine.similarity_index.conn.commit()
            
            # Get the updated record for callback
            cursor.execute('SELECT * FROM duplicate_records WHERE id = ?', (duplicate_id,))
            row = cursor.fetchone()
            
            if row:
                duplicate = DuplicateRecord(
                    document_id=row[1],
                    duplicate_of=row[2],
                    similarity_score=row[3],
                    detection_method=row[4],
                    detected_at=row[5],
                    resolved=bool(row[6]),
                    resolution_method=row[7],
                    resolution_details=resolution_details
                )
                
                # Notify callbacks
                for callback in self.deduplication_callbacks:
                    try:
                        callback({
                            'type': 'duplicate_resolved',
                            'duplicate': asdict(duplicate)
                        })
                    except Exception as e:
                        logger.error(f"Error in duplicate resolved callback: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error resolving duplicate manually: {e}")
            return False
    
    async def shutdown_deduplication(self):
        """Shutdown the deduplication engine."""
        if self.is_deduplication_initialized and self.deduplication_engine:
            try:
                await self.deduplication_engine.shutdown()
                self.is_deduplication_initialized = False
                logger.info("Deduplication engine shutdown complete")
            except Exception as e:
                logger.error(f"Error during deduplication engine shutdown: {e}")
    
    async def shutdown(self):
        """Shutdown the enhanced document manager with deduplication."""
        # Shutdown deduplication first
        await self.shutdown_deduplication()
        
        # Then shutdown parent
        await super().shutdown()


class DeduplicationBatchProcessor:
    """Batch processor for deduplication operations."""
    
    def __init__(self, enhanced_doc_manager: EnhancedDocumentManagerWithDeduplication):
        self.doc_manager = enhanced_doc_manager
        self.processing_lock = asyncio.Lock()
    
    async def process_batch_deduplication(self, document_ids: List[str], 
                                        detection_methods: List[str] = None) -> Dict[str, Any]:
        """
        Process deduplication for a batch of documents.
        
        Args:
            document_ids: List of document IDs to check for duplicates
            detection_methods: List of detection methods to use ('exact', 'semantic', 'fuzzy')
        
        Returns:
            Dictionary with batch processing results
        """
        if not self.doc_manager.is_deduplication_initialized:
            return {'error': 'Deduplication engine not initialized'}
        
        detection_methods = detection_methods or ['exact', 'semantic', 'fuzzy']
        
        results = {
            'total_documents': len(document_ids),
            'processed': 0,
            'duplicates_found': 0,
            'duplicates_by_method': defaultdict(int),
            'detailed_results': []
        }
        
        async with self.processing_lock:
            for document_id in document_ids:
                try:
                    # Get document content
                    md_path = self.doc_manager.markdown_dir / f"{document_id}.md"
                    if not md_path.exists():
                        continue
                    
                    with open(md_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for duplicates
                    duplicates = await self.doc_manager.deduplication_engine.check_duplicates(
                        document_id, content
                    )
                    
                    results['processed'] += 1
                    
                    if duplicates:
                        results['duplicates_found'] += len(duplicates)
                        
                        # Count by detection method
                        for duplicate in duplicates:
                            results['duplicates_by_method'][duplicate.detection_method] += 1
                        
                        # Add detailed results
                        results['detailed_results'].append({
                            'document_id': document_id,
                            'duplicates': [asdict(d) for d in duplicates],
                            'count': len(duplicates)
                        })
                
                except Exception as e:
                    logger.error(f"Error processing document {document_id}: {e}")
        
        return results
    
    async def find_all_duplicates(self, detection_methods: List[str] = None) -> Dict[str, Any]:
        """Find all duplicates in the document collection."""
        if not self.doc_manager.is_deduplication_initialized:
            return {'error': 'Deduplication engine not initialized'}
        
        try:
            # Get all document IDs
            documents = self.doc_manager.list_documents()
            document_ids = [doc['document_id'] for doc in documents]
            
            # Process batch
            results = await self.process_batch_deduplication(document_ids, detection_methods)
            
            # Add additional statistics
            results['total_unique_documents'] = len(document_ids)
            results['duplicate_rate'] = results['duplicates_found'] / max(results['processed'], 1)
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding all duplicates: {e}")
            return {'error': str(e)}
    
    async def cleanup_resolved_duplicates(self, dry_run: bool = True) -> Dict[str, Any]:
        """Clean up resolved duplicate documents."""
        if not self.doc_manager.is_deduplication_initialized:
            return {'error': 'Deduplication engine not initialized'}
        
        try:
            cursor = self.doc_manager.deduplication_engine.similarity_index.conn.cursor()
            
            # Get resolved duplicates
            cursor.execute('''
                SELECT document_id, duplicate_of, resolution_method 
                FROM duplicate_records 
                WHERE resolved = 1 AND resolution_method IN ('keep_newest', 'keep_oldest', 'keep_largest')
            ''')
            
            resolved_duplicates = cursor.fetchall()
            
            cleanup_results = {
                'total_resolved': len(resolved_duplicates),
                'would_remove': [],
                'removed': [],
                'errors': []
            }
            
            for document_id, duplicate_of, resolution_method in resolved_duplicates:
                try:
                    if resolution_method == 'keep_newest':
                        # Remove the older document
                        document_to_remove = duplicate_of
                    elif resolution_method == 'keep_oldest':
                        # Remove the newer document
                        document_to_remove = document_id
                    elif resolution_method == 'keep_largest':
                        # This would require checking file sizes
                        document_to_remove = duplicate_of  # Simplified for now
                    else:
                        continue
                    
                    if dry_run:
                        cleanup_results['would_remove'].append({
                            'document_id': document_id,
                            'duplicate_of': duplicate_of,
                            'resolution_method': resolution_method,
                            'would_remove': document_to_remove
                        })
                    else:
                        # Actually remove the document
                        # Note: This would need to be implemented based on document storage
                        cleanup_results['removed'].append({
                            'document_id': document_id,
                            'duplicate_of': duplicate_of,
                            'resolution_method': resolution_method,
                            'removed': document_to_remove
                        })
                
                except Exception as e:
                    cleanup_results['errors'].append({
                        'document_id': document_id,
                        'error': str(e)
                    })
            
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Error cleaning up resolved duplicates: {e}")
            return {'error': str(e)}


# Convenience functions for easy integration
def create_enhanced_doc_manager_with_deduplication(rag_system, **kwargs) -> EnhancedDocumentManagerWithDeduplication:
    """Create an enhanced document manager with deduplication capabilities."""
    return EnhancedDocumentManagerWithDeduplication(rag_system, **kwargs)


def create_deduplication_batch_processor(enhanced_doc_manager: EnhancedDocumentManagerWithDeduplication) -> DeduplicationBatchProcessor:
    """Create a batch processor for deduplication operations."""
    return DeduplicationBatchProcessor(enhanced_doc_manager)


if __name__ == "__main__":
    
    # rag_system = RAGSystem()  # Assume RAGSystem is imported
    
    # Create enhanced document manager with deduplication
    # config = DeduplicationConfig(
    #     enable_exact_dedup=True,
    #     enable_semantic_dedup=True,
    #     semantic_threshold=0.85,
    #     auto_resolve_duplicates=True,
    #     resolution_strategy='keep_newest'
    # )
    
    # doc_manager = EnhancedDocumentManagerWithDeduplication(
    #     rag_system, 
    #     enable_deduplication=True,
    #     deduplication_config=config
    # )
    
    # await doc_manager.initialize_deduplication()
    
    # Add a document with deduplication
    # success, message, metadata = await doc_manager.add_document(Path("document.pdf"))
    
    # Get deduplication statistics
    # stats = doc_manager.get_deduplication_stats()
    # print(f"Deduplication stats: {stats}")
    
    pass