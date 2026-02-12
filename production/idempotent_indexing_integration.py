"""
Idempotent Indexing Integration Module

Integrates the advanced idempotent indexing system with the existing EnhancedDocumentManager
to provide seamless idempotent indexing capabilities during document processing.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import asdict

from config import MARKDOWN_DIR
from enhanced_document_manager import EnhancedDocumentManager, DocumentMetadata
from idempotent_indexing import (
    IdempotentIndexingSystem, IndexType, IndexKey, IndexEntry, 
    IndexConsistencyLevel, IndexStatus, IndexOperation
)
from metadata_manager import MetadataManager, MetadataSchema, MetadataField, FieldConstraint, MetadataFieldType
from versioning_system import VersioningSystem, SemanticVersion
from rag_system import RAGSystem

logger = logging.getLogger(__name__)

class EnhancedDocumentManagerWithIdempotentIndexing(EnhancedDocumentManager):
    """Enhanced document manager with integrated idempotent indexing capabilities."""
    
    def __init__(self, rag_system, enable_idempotent_indexing=True, 
                 indexing_config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced document manager with idempotent indexing.
        
        Args:
            rag_system: The RAG system instance
            enable_idempotent_indexing: Whether to enable idempotent indexing
            indexing_config: Idempotent indexing configuration
        """
        # Initialize parent class
        super().__init__(rag_system, enable_deduplication=True, enable_versioning=True)
        
        # Idempotent indexing settings
        self.enable_idempotent_indexing = enable_idempotent_indexing
        self.indexing_config = indexing_config or self._get_default_indexing_config()
        
        # Initialize idempotent indexing system
        self.idempotent_indexing_system = None
        self.is_indexing_initialized = False
        
        # Indexing settings
        self.default_index_type = IndexType.DENSE_VECTOR
        self.default_consistency_level = IndexConsistencyLevel.STRONG
        
    def _get_default_indexing_config(self) -> Dict[str, Any]:
        """Get default idempotent indexing configuration."""
        return {
            'indexing_db_path': 'idempotent_indexing.db',
            'enable_deterministic_indexing': True,
            'enable_index_versioning': True,
            'enable_index_consistency': True,
            'enable_index_optimization': True,
            'enable_distributed_indexing': False
        }
    
    async def initialize_idempotent_indexing(self):
        """Initialize the idempotent indexing system."""
        if not self.enable_idempotent_indexing:
            logger.info("Idempotent indexing is disabled")
            return
        
        try:
            self.idempotent_indexing_system = IdempotentIndexingSystem(self.indexing_config)
            self.is_indexing_initialized = True
            logger.info("Idempotent indexing system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize idempotent indexing system: {e}")
            self.is_indexing_initialized = False
    
    async def add_document(self, file_path: Path, metadata: Optional[Dict] = None, 
                          force_reprocess: bool = False) -> Tuple[bool, str, Optional[DocumentMetadata]]:
        """
        Add a document with integrated idempotent indexing.
        
        This method overrides the parent method to add idempotent indexing
        during document processing.
        """
        try:
            # Step 1: Process document (same as parent method)
            success, message, doc_metadata = await super().add_document(
                file_path, metadata, force_reprocess
            )
            
            if not success or not doc_metadata:
                return success, message, doc_metadata
            
            # Step 2: Create idempotent index if indexing is enabled
            if self.enable_idempotent_indexing and self.is_indexing_initialized:
                # Get document content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    logger.warning(f"Could not read file content for indexing: {e}")
                    content = ""
                
                # Get document vector (simplified - would use actual embedding)
                vector = self._generate_mock_vector(content)
                
                # Create idempotent index
                index_key = await self.idempotent_indexing_system.index_document(
                    document_id=doc_metadata.document_id,
                    content=content,
                    vector=vector,
                    metadata=doc_metadata.metadata,
                    index_type=self.default_index_type,
                    consistency_level=self.default_consistency_level
                )
                
                if index_key:
                    message += f" | Indexed with key {index_key}"
                else:
                    message += " | Indexing failed"
            
            return success, message, doc_metadata
            
        except Exception as e:
            logger.error(f"Error adding document with idempotent indexing: {e}")
            return False, str(e), None
    
    def _generate_mock_vector(self, content: str) -> List[float]:
        """Generate a mock vector for demonstration purposes."""
        # In practice, this would use actual embedding models
        import hashlib
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        # Convert hash to vector
        vector = [float(int(content_hash[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
        return vector + [0.0] * (1536 - len(vector))  # Pad to standard size
    
    async def index_document(self, document_id: str, content: str, 
                           vector: List[float], metadata: Dict[str, Any],
                           index_type: IndexType = IndexType.DENSE_VECTOR,
                           consistency_level: IndexConsistencyLevel = IndexConsistencyLevel.STRONG,
                           version: Optional[str] = None) -> Optional[str]:
        """Index a document with idempotent guarantees."""
        if not self.enable_idempotent_indexing or not self.is_indexing_initialized:
            return None
        
        try:
            return await self.idempotent_indexing_system.index_document(
                document_id, content, vector, metadata, index_type, consistency_level, version
            )
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            return None
    
    async def update_index(self, index_key: str, content: Optional[str] = None,
                          vector: Optional[List[float]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing index entry."""
        if not self.enable_idempotent_indexing or not self.is_indexing_initialized:
            return False
        
        try:
            return await self.idempotent_indexing_system.update_index(index_key, content, vector, metadata)
        except Exception as e:
            logger.error(f"Error updating index: {e}")
            return False
    
    async def delete_index(self, index_key: str) -> bool:
        """Delete an index entry."""
        if not self.enable_idempotent_indexing or not self.is_indexing_initialized:
            return False
        
        try:
            return await self.idempotent_indexing_system.delete_index(index_key)
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False
    
    async def validate_index(self, index_key: str) -> bool:
        """Validate an index entry."""
        if not self.enable_idempotent_indexing or not self.is_indexing_initialized:
            return False
        
        try:
            return await self.idempotent_indexing_system.validate_index(index_key)
        except Exception as e:
            logger.error(f"Error validating index: {e}")
            return False
    
    async def optimize_index(self, index_type: IndexType) -> bool:
        """Optimize an index."""
        if not self.enable_idempotent_indexing or not self.is_indexing_initialized:
            return False
        
        try:
            return await self.idempotent_indexing_system.optimize_index(index_type)
        except Exception as e:
            logger.error(f"Error optimizing index: {e}")
            return False
    
    async def rollback_index(self, index_key: str, target_version: str) -> bool:
        """Rollback an index to a previous version."""
        if not self.enable_idempotent_indexing or not self.is_indexing_initialized:
            return False
        
        try:
            return await self.idempotent_indexing_system.rollback_index(index_key, target_version)
        except Exception as e:
            logger.error(f"Error rolling back index: {e}")
            return False
    
    async def get_index_health(self, index_type: IndexType) -> Optional[Dict[str, Any]]:
        """Get index health information."""
        if not self.enable_idempotent_indexing or not self.is_indexing_initialized:
            return None
        
        try:
            health = await self.idempotent_indexing_system.get_index_health(index_type)
            if health:
                return {
                    'index_type': health.index_type.value,
                    'status': health.status.value,
                    'health_score': health.health_score,
                    'last_check_at': health.last_check_at.isoformat(),
                    'issues': health.issues,
                    'performance_metrics': health.performance_metrics
                }
            return None
        except Exception as e:
            logger.error(f"Error getting index health: {e}")
            return None
    
    async def get_index_statistics(self) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        if not self.enable_idempotent_indexing or not self.is_indexing_initialized:
            return {'error': 'Idempotent indexing system not initialized'}
        
        try:
            return await self.idempotent_indexing_system.get_index_statistics()
        except Exception as e:
            logger.error(f"Error getting index statistics: {e}")
            return {'error': str(e)}
    
    async def get_index_entries(self, index_type: IndexType) -> List[Dict[str, Any]]:
        """Get all index entries of a specific type."""
        if not self.enable_idempotent_indexing or not self.is_indexing_initialized:
            return []
        
        try:
            if not self.idempotent_indexing_system.index_storage:
                return []
            
            entries = await self.idempotent_indexing_system.index_storage.list_index_entries(index_type)
            return [
                {
                    'index_key': str(entry.index_key),
                    'document_id': entry.document_id,
                    'content_length': len(entry.content),
                    'vector_size': len(entry.vector),
                    'metadata': entry.metadata,
                    'created_at': entry.created_at.isoformat(),
                    'updated_at': entry.updated_at.isoformat(),
                    'version': entry.version,
                    'consistency_level': entry.consistency_level.value,
                    'partition_id': entry.partition_id,
                    'shard_id': entry.shard_id
                }
                for entry in entries
            ]
        except Exception as e:
            logger.error(f"Error getting index entries: {e}")
            return []
    
    async def batch_index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch index multiple documents."""
        if not self.enable_idempotent_indexing or not self.is_indexing_initialized:
            return {'error': 'Idempotent indexing system not initialized'}
        
        results = {
            'total_documents': len(documents),
            'indexed': 0,
            'failed': 0,
            'errors': []
        }
        
        for doc in documents:
            try:
                index_key = await self.index_document(
                    doc['document_id'],
                    doc['content'],
                    doc['vector'],
                    doc.get('metadata', {}),
                    doc.get('index_type', self.default_index_type),
                    doc.get('consistency_level', self.default_consistency_level)
                )
                
                if index_key:
                    results['indexed'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Failed to index document {doc['document_id']}")
            
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Error indexing {doc['document_id']}: {str(e)}")
        
        return results
    
    async def batch_validate_indexes(self, index_keys: List[str]) -> Dict[str, Any]:
        """Batch validate multiple index entries."""
        if not self.enable_idempotent_indexing or not self.is_indexing_initialized:
            return {'error': 'Idempotent indexing system not initialized'}
        
        results = {
            'total_indexes': len(index_keys),
            'valid': 0,
            'invalid': 0,
            'errors': []
        }
        
        for index_key in index_keys:
            try:
                is_valid = await self.validate_index(index_key)
                
                if is_valid:
                    results['valid'] += 1
                else:
                    results['invalid'] += 1
                    results['errors'].append(f"Invalid index: {index_key}")
            
            except Exception as e:
                results['invalid'] += 1
                results['errors'].append(f"Error validating {index_key}: {str(e)}")
        
        return results
    
    async def shutdown_idempotent_indexing(self):
        """Shutdown the idempotent indexing system."""
        if self.is_indexing_initialized and self.idempotent_indexing_system:
            try:
                await self.idempotent_indexing_system.shutdown()
                self.is_indexing_initialized = False
                logger.info("Idempotent indexing system shutdown complete")
            except Exception as e:
                logger.error(f"Error during idempotent indexing system shutdown: {e}")
    
    async def shutdown(self):
        """Shutdown the enhanced document manager with idempotent indexing."""
        # Shutdown idempotent indexing first
        await self.shutdown_idempotent_indexing()
        
        # Then shutdown parent
        await super().shutdown()


class IdempotentIndexingBatchProcessor:
    """Batch processor for idempotent indexing operations."""
    
    def __init__(self, enhanced_doc_manager: EnhancedDocumentManagerWithIdempotentIndexing):
        self.doc_manager = enhanced_doc_manager
        self.processing_lock = asyncio.Lock()
    
    async def process_batch_indexing(self, documents: List[Dict[str, Any]], 
                                   index_type: IndexType = IndexType.DENSE_VECTOR) -> Dict[str, Any]:
        """Process batch indexing operations."""
        if not self.doc_manager.is_indexing_initialized:
            return {'error': 'Idempotent indexing system not initialized'}
        
        results = {
            'total_documents': len(documents),
            'indexed': 0,
            'failed': 0,
            'errors': []
        }
        
        async with self.processing_lock:
            for doc in documents:
                try:
                    index_key = await self.doc_manager.index_document(
                        doc['document_id'],
                        doc['content'],
                        doc['vector'],
                        doc.get('metadata', {}),
                        index_type
                    )
                    
                    if index_key:
                        results['indexed'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Failed to index document {doc['document_id']}")
                
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Error indexing {doc['document_id']}: {str(e)}")
        
        return results
    
    async def process_batch_validation(self, index_keys: List[str]) -> Dict[str, Any]:
        """Process batch validation operations."""
        if not self.doc_manager.is_indexing_initialized:
            return {'error': 'Idempotent indexing system not initialized'}
        
        results = {
            'total_indexes': len(index_keys),
            'valid': 0,
            'invalid': 0,
            'errors': []
        }
        
        async with self.processing_lock:
            for index_key in index_keys:
                try:
                    is_valid = await self.doc_manager.validate_index(index_key)
                    
                    if is_valid:
                        results['valid'] += 1
                    else:
                        results['invalid'] += 1
                        results['errors'].append(f"Invalid index: {index_key}")
                
                except Exception as e:
                    results['invalid'] += 1
                    results['errors'].append(f"Error validating {index_key}: {str(e)}")
        
        return results
    
    async def process_batch_optimization(self, index_types: List[IndexType]) -> Dict[str, Any]:
        """Process batch optimization operations."""
        if not self.doc_manager.is_indexing_initialized:
            return {'error': 'Idempotent indexing system not initialized'}
        
        results = {
            'total_indexes': len(index_types),
            'optimized': 0,
            'failed': 0,
            'errors': []
        }
        
        async with self.processing_lock:
            for index_type in index_types:
                try:
                    success = await self.doc_manager.optimize_index(index_type)
                    
                    if success:
                        results['optimized'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Failed to optimize index: {index_type.value}")
                
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Error optimizing {index_type.value}: {str(e)}")
        
        return results


class IdempotentIndexingWorkflowManager:
    """Manager for idempotent indexing workflows and optimization processes."""
    
    def __init__(self, enhanced_doc_manager: EnhancedDocumentManagerWithIdempotentIndexing):
        self.doc_manager = enhanced_doc_manager
        self.workflows = {}  # workflow_id -> workflow_state
    
    async def start_indexing_workflow(self, workflow_id: str, documents: List[Dict[str, Any]],
                                    index_type: IndexType = IndexType.DENSE_VECTOR,
                                    consistency_level: IndexConsistencyLevel = IndexConsistencyLevel.STRONG) -> bool:
        """Start an indexing workflow."""
        if not self.doc_manager.is_indexing_initialized:
            return False
        
        try:
            # Store workflow state
            self.workflows[workflow_id] = {
                'status': 'running',
                'documents': documents,
                'index_type': index_type,
                'consistency_level': consistency_level,
                'progress': 0,
                'total': len(documents),
                'completed': 0,
                'failed': 0,
                'errors': []
            }
            
            # Process documents
            results = await self.doc_manager.batch_index_documents(documents)
            
            # Update workflow state
            workflow = self.workflows[workflow_id]
            workflow['completed'] = results['indexed']
            workflow['failed'] = results['failed']
            workflow['errors'] = results['errors']
            workflow['progress'] = 100
            workflow['status'] = 'completed'
            
            logger.info(f"Indexing workflow {workflow_id} completed: {results}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting indexing workflow: {e}")
            return False
    
    async def start_optimization_workflow(self, workflow_id: str, index_types: List[IndexType]) -> bool:
        """Start an optimization workflow."""
        if not self.doc_manager.is_indexing_initialized:
            return False
        
        try:
            # Store workflow state
            self.workflows[workflow_id] = {
                'status': 'running',
                'index_types': index_types,
                'progress': 0,
                'total': len(index_types),
                'completed': 0,
                'failed': 0,
                'errors': []
            }
            
            # Process optimizations
            results = await self.doc_manager.batch_optimize_indexes(index_types)
            
            # Update workflow state
            workflow = self.workflows[workflow_id]
            workflow['completed'] = results['optimized']
            workflow['failed'] = results['failed']
            workflow['errors'] = results['errors']
            workflow['progress'] = 100
            workflow['status'] = 'completed'
            
            logger.info(f"Optimization workflow {workflow_id} completed: {results}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting optimization workflow: {e}")
            return False
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow."""
        return self.workflows.get(workflow_id)
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id in self.workflows:
            self.workflows[workflow_id]['status'] = 'cancelled'
            logger.info(f"Cancelled workflow {workflow_id}")
            return True
        return False
    
    async def cleanup_completed_workflows(self) -> bool:
        """Clean up completed workflows."""
        try:
            completed_workflows = [
                workflow_id for workflow_id, workflow in self.workflows.items()
                if workflow['status'] in ['completed', 'cancelled', 'failed']
            ]
            
            for workflow_id in completed_workflows:
                del self.workflows[workflow_id]
            
            logger.info(f"Cleaned up {len(completed_workflows)} completed workflows")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up workflows: {e}")
            return False


# Convenience functions for easy integration
def create_enhanced_doc_manager_with_idempotent_indexing(rag_system, **kwargs) -> EnhancedDocumentManagerWithIdempotentIndexing:
    """Create an enhanced document manager with idempotent indexing capabilities."""
    return EnhancedDocumentManagerWithIdempotentIndexing(rag_system, **kwargs)


def create_idempotent_indexing_batch_processor(enhanced_doc_manager: EnhancedDocumentManagerWithIdempotentIndexing) -> IdempotentIndexingBatchProcessor:
    """Create a batch processor for idempotent indexing operations."""
    return IdempotentIndexingBatchProcessor(enhanced_doc_manager)


def create_idempotent_indexing_workflow_manager(enhanced_doc_manager: EnhancedDocumentManagerWithIdempotentIndexing) -> IdempotentIndexingWorkflowManager:
    """Create a workflow manager for idempotent indexing processes."""
    return IdempotentIndexingWorkflowManager(enhanced_doc_manager)


if __name__ == "__main__":
    
    # rag_system = RAGSystem()  # Assume RAGSystem is imported
    
    # Create enhanced document manager with idempotent indexing
    # indexing_config = {
    #     'indexing_db_path': 'idempotent_indexing.db',
    #     'enable_deterministic_indexing': True,
    #     'enable_index_versioning': True,
    #     'enable_index_consistency': True,
    #     'enable_index_optimization': True,
    #     'enable_distributed_indexing': False
    # }
    
    # doc_manager = EnhancedDocumentManagerWithIdempotentIndexing(
    #     rag_system, 
    #     enable_idempotent_indexing=True,
    #     indexing_config=indexing_config
    # )
    
    # await doc_manager.initialize_idempotent_indexing()
    
    # Add a document with idempotent indexing
    # success, message, metadata = await doc_manager.add_document(Path("document.pdf"))
    
    # Index document manually
    # index_key = await doc_manager.index_document(
    #     metadata.document_id, "Document content", [0.1, 0.2, 0.3], {"author": "John"}
    # )
    
    # Validate index
    # is_valid = await doc_manager.validate_index(index_key)
    
    # Get index health
    # health = await doc_manager.get_index_health(IndexType.DENSE_VECTOR)
    
    # Batch operations
    # documents = [
    #     {'document_id': 'doc_1', 'content': 'Content 1', 'vector': [0.1, 0.2], 'metadata': {}},
    #     {'document_id': 'doc_2', 'content': 'Content 2', 'vector': [0.3, 0.4], 'metadata': {}}
    # ]
    # batch_results = await doc_manager.batch_index_documents(documents)
    
    pass