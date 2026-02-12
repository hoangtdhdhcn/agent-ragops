"""
Metadata Example: Complete Production RAG System with Advanced Metadata Management

This example demonstrates how to integrate the advanced metadata management system
into a complete production RAG system.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

from config import MARKDOWN_DIR, PARENT_STORE_PATH, QDRANT_DB_PATH
from enhanced_document_manager import EnhancedDocumentManager
from metadata_manager import MetadataManager, MetadataSchema, MetadataField, FieldConstraint, MetadataFieldType
from metadata_integration import (
    EnhancedDocumentManagerWithMetadata,
    create_enhanced_doc_manager_with_metadata,
    create_metadata_batch_processor
)
from deduplication_engine import DeduplicationEngine, DeduplicationConfig
from deduplication_integration import create_enhanced_doc_manager_with_deduplication
from delta_processor import DeltaProcessor, create_delta_processor
from monitoring_system import setup_monitoring_system
from rag_system import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionRAGSystemWithMetadata:
    """Complete production-ready RAG system with advanced metadata management."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize production RAG system with metadata management.
        
        Args:
            config: Configuration dictionary with system settings
        """
        self.config = config
        self.rag_system = None
        self.doc_manager = None
        self.metadata_manager = None
        self.metadata_batch_processor = None
        self.deduplication_engine = None
        self.delta_processor = None
        self.monitoring_system = None
        
        # System state
        self.is_initialized = False
        self.is_monitoring = False
        
    async def initialize(self):
        """Initialize the complete production system with metadata management."""
        try:
            logger.info("Initializing Production RAG System with Metadata Management...")
            
            # Initialize core RAG system
            self.rag_system = RAGSystem()
            await self.rag_system.initialize()
            
            # Initialize enhanced document manager with metadata
            metadata_config = self.config.get('metadata_config', {})
            self.doc_manager = create_enhanced_doc_manager_with_metadata(
                self.rag_system,
                enable_metadata=self.config.get('enable_metadata', True),
                enable_versioning=self.config.get('enable_versioning', True),
                metadata_config=metadata_config
            )
            
            # Initialize metadata management
            await self.doc_manager.initialize_metadata()
            
            # Initialize metadata batch processor
            self.metadata_batch_processor = create_metadata_batch_processor(self.doc_manager)
            
            # Initialize deduplication (if enabled)
            if self.config.get('enable_deduplication', False):
                dedup_config = DeduplicationConfig(
                    enable_exact_dedup=self.config.get('enable_exact_dedup', True),
                    enable_semantic_dedup=self.config.get('enable_semantic_dedup', True),
                    enable_fuzzy_dedup=self.config.get('enable_fuzzy_dedup', True),
                    semantic_threshold=self.config.get('semantic_threshold', 0.85),
                    fuzzy_threshold=self.config.get('fuzzy_threshold', 0.90),
                    auto_resolve_duplicates=self.config.get('auto_resolve_duplicates', True),
                    resolution_strategy=self.config.get('resolution_strategy', 'keep_newest')
                )
                
                # Create enhanced document manager with deduplication
                dedup_doc_manager = create_enhanced_doc_manager_with_deduplication(
                    self.rag_system,
                    enable_deduplication=True,
                    deduplication_config=dedup_config
                )
                await dedup_doc_manager.initialize_deduplication()
            
            # Initialize delta processor
            self.delta_processor = create_delta_processor(
                self.rag_system,
                self.doc_manager,
                **self.config.get('delta_processor', {})
            )
            
            # Initialize monitoring system
            self.monitoring_system = setup_monitoring_system(
                self.rag_system,
                self.delta_processor,
                prometheus_port=self.config.get('prometheus_port', 8000)
            )
            
            self.is_initialized = True
            logger.info("Production RAG System with Metadata Management initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def start_monitoring(self, directories: List[Path]):
        """Start monitoring directories for document changes."""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        try:
            await self.delta_processor.start_monitoring(
                directories,
                interval=self.config.get('monitoring_interval', 60)
            )
            
            self.is_monitoring = True
            logger.info(f"Started monitoring {len(directories)} directories")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            raise
    
    async def stop_monitoring(self):
        """Stop monitoring directories."""
        if not self.is_monitoring:
            return
        
        try:
            await self.delta_processor.stop_monitoring()
            self.is_monitoring = False
            logger.info("Stopped monitoring")
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
    
    async def ingest_documents(self, paths: List[Path], 
                             recursive: bool = True,
                             metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Ingest documents with full production features including metadata management."""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        try:
            # Use enhanced document manager with metadata for ingestion
            results = self.doc_manager.add_directory(
                paths[0] if len(paths) == 1 else paths,
                recursive=recursive,
                metadata=metadata
            )
            
            logger.info(f"Ingestion completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to ingest documents: {e}")
            return {'error': str(e)}
    
    async def ingest_web_pages(self, urls: List[str], 
                             metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Ingest web pages with full production features including metadata management."""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        results = []
        for url in urls:
            try:
                success, message, doc_metadata = self.doc_manager.add_web_page(url, metadata)
                results.append({
                    'url': url,
                    'success': success,
                    'message': message,
                    'document_id': doc_metadata.document_id if doc_metadata else None
                })
            except Exception as e:
                results.append({
                    'url': url,
                    'success': False,
                    'error': str(e)
                })
        
        logger.info(f"Web page ingestion completed: {len(results)} pages processed")
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including metadata management metrics."""
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        try:
            # Get document statistics
            doc_stats = self.doc_manager.get_statistics()
            
            # Get metadata statistics
            metadata_stats = self.doc_manager.get_metadata_statistics()
            
            # Get processing status
            processing_status = self.delta_processor.get_queue_status()
            
            # Get monitoring status
            system_health = self.monitoring_system['metrics_collector'].get_system_health()
            processing_health = self.monitoring_system['metrics_collector'].get_processing_health()
            alert_summary = self.monitoring_system['alert_manager'].get_alert_summary()
            
            return {
                'status': 'running',
                'document_stats': doc_stats,
                'metadata_stats': metadata_stats,
                'processing_status': processing_status,
                'system_health': system_health,
                'processing_health': processing_health,
                'alerts': alert_summary,
                'monitoring_active': self.is_monitoring
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document."""
        if not self.is_initialized:
            return None
        
        try:
            return self.doc_manager.get_document_metadata(document_id)
        except Exception as e:
            logger.error(f"Error getting document metadata: {e}")
            return None
    
    def update_document_metadata(self, document_id: str, metadata_updates: Dict[str, Any], 
                               user_id: Optional[str] = None, 
                               reason: Optional[str] = None) -> bool:
        """Update metadata for a specific document."""
        if not self.is_initialized:
            return False
        
        try:
            return self.doc_manager.update_document_metadata(
                document_id, metadata_updates, user_id, reason
            )
        except Exception as e:
            logger.error(f"Error updating document metadata: {e}")
            return False
    
    def search_documents_by_metadata(self, query: Dict[str, Any], 
                                   schema_id: Optional[str] = None) -> List[str]:
        """Search documents by metadata."""
        if not self.is_initialized:
            return []
        
        try:
            return self.doc_manager.search_documents_by_metadata(query, schema_id)
        except Exception as e:
            logger.error(f"Error searching documents by metadata: {e}")
            return []
    
    def get_document_audit_trail(self, document_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit trail for a document."""
        if not self.is_initialized:
            return []
        
        try:
            return self.doc_manager.get_document_audit_trail(document_id, limit)
        except Exception as e:
            logger.error(f"Error getting document audit trail: {e}")
            return []
    
    def create_document_relationship(self, source_document_id: str, target_document_id: str, 
                                   relationship_type: str, relationship_data: Optional[Dict[str, Any]] = None,
                                   user_id: Optional[str] = None) -> bool:
        """Create a relationship between documents."""
        if not self.is_initialized:
            return False
        
        try:
            return self.doc_manager.create_document_relationship(
                source_document_id, target_document_id, relationship_type, 
                relationship_data, user_id
            )
        except Exception as e:
            logger.error(f"Error creating document relationship: {e}")
            return False
    
    def get_document_relationships(self, document_id: str, 
                                 relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get relationships for a document."""
        if not self.is_initialized:
            return []
        
        try:
            return self.doc_manager.get_document_relationships(document_id, relationship_type)
        except Exception as e:
            logger.error(f"Error getting document relationships: {e}")
            return []
    
    async def run_batch_metadata_operations(self, document_ids: List[str], 
                                          operation: str, 
                                          operation_data: Optional[Dict[str, Any]] = None,
                                          user_id: Optional[str] = None) -> Dict[str, Any]:
        """Run batch metadata operations."""
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            if operation == 'update':
                results = await self.metadata_batch_processor.process_batch_metadata_update(
                    document_ids, operation_data or {}, user_id
                )
            elif operation == 'export':
                results = await self.metadata_batch_processor.export_batch_metadata(
                    document_ids, operation_data.get('format', 'json') if operation_data else 'json'
                )
            elif operation == 'import':
                results = await self.metadata_batch_processor.import_batch_metadata(
                    operation_data or {}, user_id
                )
            else:
                return {'error': f'Unknown operation: {operation}'}
            
            logger.info(f"Batch metadata operation completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to run batch metadata operations: {e}")
            return {'error': str(e)}
    
    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive system report including metadata management metrics."""
        try:
            # Get base report
            report = self.monitoring_system['dashboard'].generate_report(hours)
            
            # Add metadata management metrics
            metadata_stats = self.doc_manager.get_metadata_statistics()
            report['metadata_management'] = metadata_stats
            
            # Add metadata configuration
            report['metadata_config'] = {
                'enabled': self.config.get('enable_metadata', True),
                'validation_enabled': self.config.get('metadata_config', {}).get('enable_validation', True),
                'auditing_enabled': self.config.get('metadata_config', {}).get('enable_auditing', True),
                'relationships_enabled': self.config.get('metadata_config', {}).get('enable_relationships', True),
                'id_strategy': self.config.get('metadata_config', {}).get('id_strategy', 'hierarchical'),
                'default_schema': self.doc_manager.default_schema_id if hasattr(self.doc_manager, 'default_schema_id') else 'document_v1'
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Shutdown the production system with metadata management gracefully."""
        try:
            # Stop monitoring
            await self.stop_monitoring()
            
            # Shutdown metadata management
            await self.doc_manager.shutdown_metadata()
            
            # Stop monitoring system
            if self.monitoring_system:
                self.monitoring_system['metrics_collector'].stop_collection()
                self.monitoring_system['dashboard'].stop_dashboard()
            
            # Clear all data if configured
            if self.config.get('clear_on_shutdown', False):
                self.doc_manager.clear_all()
            
            self.is_initialized = False
            logger.info("Production RAG System with Metadata Management shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def main():
    """Example usage of the production RAG system with metadata management."""
    
    # Configuration
    config = {
        'enable_metadata': True,
        'enable_versioning': True,
        'enable_deduplication': True,
        'enable_exact_dedup': True,
        'enable_semantic_dedup': True,
        'enable_fuzzy_dedup': True,
        'semantic_threshold': 0.85,
        'fuzzy_threshold': 0.90,
        'auto_resolve_duplicates': True,
        'resolution_strategy': 'keep_newest',
        'metadata_config': {
            'metadata_db_path': 'metadata_manager.db',
            'enable_validation': True,
            'enable_auditing': True,
            'enable_relationships': True,
            'id_strategy': 'hierarchical',
            'namespace': 'enterprise',
            'document_type': 'document'
        },
        'monitoring_interval': 60,  # seconds
        'prometheus_port': 8000,
        'delta_processor': {
            'batch_size': 10,
            'max_workers': 4
        },
        'clear_on_shutdown': False
    }
    
    # Initialize system
    system = ProductionRAGSystemWithMetadata(config)
    await system.initialize()
    
    # Example 1: Ingest documents with metadata
    document_dirs = [
        Path("documents/"),
        Path("reports/"),
        Path("manuals/")
    ]
    
    # Custom metadata for ingestion
    custom_metadata = {
        'department': 'Engineering',
        'classification': 'internal',
        'tags': ['technical', 'documentation'],
        'project': 'RAG System',
        'version': '1.0'
    }
    
    for directory in document_dirs:
        if directory.exists():
            results = await system.ingest_documents([directory], recursive=True, metadata=custom_metadata)
            print(f"Ingested {directory}: {results}")
    
    # Example 2: Ingest web pages with metadata
    web_pages = [
        "https://docs.python.org/3/",
        "https://langchain.com/",
        "https://qdrant.tech/"
    ]
    
    web_metadata = {
        'source_type': 'web',
        'crawl_date': '2024-01-01',
        'tags': ['reference', 'documentation']
    }
    
    web_results = await system.ingest_web_pages(web_pages, metadata=web_metadata)
    print(f"Web page ingestion results: {web_results}")
    
    # Example 3: Start monitoring with metadata management
    await system.start_monitoring(document_dirs)
    
    # Example 4: Get system status with metadata metrics
    status = system.get_system_status()
    print(f"System status: {status}")
    
    # Example 5: Get document metadata
    if status.get('document_stats', {}).get('total_documents', 0) > 0:
        # Get first document ID
        documents = system.doc_manager.list_documents()
        if documents:
            first_doc_id = documents[0]['document_id']
            
            # Get document metadata
            doc_metadata = system.get_document_metadata(first_doc_id)
            print(f"Document metadata: {doc_metadata}")
            
            # Update document metadata
            update_success = system.update_document_metadata(
                first_doc_id, 
                {'tags': ['updated', 'important'], 'classification': 'confidential'},
                user_id='admin',
                reason='Updated document classification'
            )
            print(f"Metadata update success: {update_success}")
            
            # Get audit trail
            audit_trail = system.get_document_audit_trail(first_doc_id)
            print(f"Audit trail entries: {len(audit_trail)}")
    
    # Example 6: Search documents by metadata
    search_results = system.search_documents_by_metadata({'classification': 'internal'})
    print(f"Documents with internal classification: {len(search_results)}")
    
    # Example 7: Create document relationships
    if len(search_results) >= 2:
        relationship_success = system.create_document_relationship(
            search_results[0], search_results[1], 
            'related_to', 
            {'relationship_strength': 0.8, 'relationship_type': 'content_similarity'},
            user_id='admin'
        )
        print(f"Relationship creation success: {relationship_success}")
        
        # Get document relationships
        relationships = system.get_document_relationships(search_results[0])
        print(f"Document relationships: {len(relationships)}")
    
    # Example 8: Batch metadata operations
    batch_results = await system.run_batch_metadata_operations(
        search_results[:5], 
        'update', 
        {'department': 'Updated Department'},
        user_id='admin'
    )
    print(f"Batch metadata update results: {batch_results}")
    
    # Example 9: Generate comprehensive report
    await asyncio.sleep(120)  # Wait 2 minutes
    
    report = system.generate_report(hours=1)
    print(f"System report: {report}")
    
    # Shutdown
    await system.shutdown()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())