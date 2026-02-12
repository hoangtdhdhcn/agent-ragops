"""
Metadata Integration Module

Integrates the advanced metadata management system with the existing EnhancedDocumentManager
to provide seamless metadata capabilities during document ingestion and management.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import asdict

from config import MARKDOWN_DIR
from enhanced_document_manager import EnhancedDocumentManager, DocumentMetadata
from metadata_manager import (
    MetadataManager, MetadataSchema, MetadataField, FieldConstraint, 
    MetadataFieldType, IDGenerationStrategy, MetadataLifecycleStage
)
from multi_format_ingestion import MultiFormatIngestionPipeline, DocumentTypeDetector

logger = logging.getLogger(__name__)

class EnhancedDocumentManagerWithMetadata(EnhancedDocumentManager):
    """Enhanced document manager with integrated metadata management capabilities."""
    
    def __init__(self, rag_system, enable_metadata=True, enable_versioning=True, 
                 metadata_config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced document manager with metadata management.
        
        Args:
            rag_system: The RAG system instance
            enable_metadata: Whether to enable metadata management
            enable_versioning: Whether to enable versioning
            metadata_config: Metadata management configuration
        """
        # Initialize parent class
        super().__init__(rag_system, enable_deduplication=True, enable_versioning=enable_versioning)
        
        # Metadata settings
        self.enable_metadata = enable_metadata
        self.metadata_config = metadata_config or self._get_default_metadata_config()
        
        # Initialize metadata manager
        self.metadata_manager = None
        self.is_metadata_initialized = False
        
        # Default metadata schema
        self.default_schema_id = "document_v1"
        
    def _get_default_metadata_config(self) -> Dict[str, Any]:
        """Get default metadata configuration."""
        return {
            'metadata_db_path': 'metadata_manager.db',
            'enable_validation': True,
            'enable_auditing': True,
            'enable_relationships': True,
            'id_strategy': IDGenerationStrategy.HIERARCHICAL,
            'namespace': 'enterprise',
            'document_type': 'document'
        }
    
    async def initialize_metadata(self):
        """Initialize the metadata management system."""
        if not self.enable_metadata:
            logger.info("Metadata management is disabled")
            return
        
        try:
            self.metadata_manager = MetadataManager(self.metadata_config)
            
            # Create default schema if it doesn't exist
            await self._create_default_schema()
            
            self.is_metadata_initialized = True
            logger.info("Metadata management system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize metadata management: {e}")
            self.is_metadata_initialized = False
    
    async def _create_default_schema(self):
        """Create the default document metadata schema."""
        if not self.metadata_manager:
            return
        
        # Define default schema
        default_schema = MetadataSchema(
            schema_id=self.default_schema_id,
            name="Document Metadata Schema",
            version="1.0",
            description="Default schema for document metadata",
            fields=[
                MetadataField(
                    name="title",
                    field_type=MetadataFieldType.STRING,
                    description="Document title",
                    constraints=FieldConstraint(required=True, max_length=500)
                ),
                MetadataField(
                    name="author",
                    field_type=MetadataFieldType.STRING,
                    description="Document author",
                    constraints=FieldConstraint(max_length=200)
                ),
                MetadataField(
                    name="document_type",
                    field_type=MetadataFieldType.STRING,
                    description="Type of document",
                    constraints=FieldConstraint(
                        allowed_values=['pdf', 'docx', 'xlsx', 'csv', 'html', 'txt', 'image', 'web']
                    )
                ),
                MetadataField(
                    name="file_size",
                    field_type=MetadataFieldType.INTEGER,
                    description="File size in bytes"
                ),
                MetadataField(
                    name="file_hash",
                    field_type=MetadataFieldType.STRING,
                    description="SHA-256 hash of original file"
                ),
                MetadataField(
                    name="creation_date",
                    field_type=MetadataFieldType.DATETIME,
                    description="Document creation date"
                ),
                MetadataField(
                    name="modification_date",
                    field_type=MetadataFieldType.DATETIME,
                    description="Document modification date"
                ),
                MetadataField(
                    name="tags",
                    field_type=MetadataFieldType.LIST,
                    description="Document tags",
                    constraints=FieldConstraint(max_length=50)
                ),
                MetadataField(
                    name="version",
                    field_type=MetadataFieldType.STRING,
                    description="Document version"
                ),
                MetadataField(
                    name="source",
                    field_type=MetadataFieldType.STRING,
                    description="Document source or origin"
                ),
                MetadataField(
                    name="department",
                    field_type=MetadataFieldType.STRING,
                    description="Department or organization"
                ),
                MetadataField(
                    name="classification",
                    field_type=MetadataFieldType.STRING,
                    description="Document classification level",
                    constraints=FieldConstraint(
                        allowed_values=['public', 'internal', 'confidential', 'restricted']
                    )
                ),
                MetadataField(
                    name="language",
                    field_type=MetadataFieldType.STRING,
                    description="Document language",
                    constraints=FieldConstraint(max_length=10)
                ),
                MetadataField(
                    name="page_count",
                    field_type=MetadataFieldType.INTEGER,
                    description="Number of pages in document"
                ),
                MetadataField(
                    name="word_count",
                    field_type=MetadataFieldType.INTEGER,
                    description="Approximate word count"
                ),
                MetadataField(
                    name="custom_fields",
                    field_type=MetadataFieldType.DICT,
                    description="Custom metadata fields"
                )
            ],
            tags=['document', 'enterprise', 'default']
        )
        
        # Create schema
        success = self.metadata_manager.create_schema(default_schema)
        if success:
            logger.info("Created default metadata schema")
        else:
            logger.warning("Default metadata schema already exists or creation failed")
    
    async def add_document(self, file_path: Path, metadata: Optional[Dict] = None, 
                          force_reprocess: bool = False) -> Tuple[bool, str, Optional[DocumentMetadata]]:
        """
        Add a document with integrated metadata management.
        
        This method overrides the parent method to add metadata management
        during document processing.
        """
        try:
            # Step 1: Generate document ID using metadata manager
            document_id = None
            if self.enable_metadata and self.is_metadata_initialized:
                # Get document content for ID generation
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    document_id = self.metadata_manager.generate_document_id(content.decode('utf-8', errors='ignore'))
                except Exception as e:
                    logger.warning(f"Could not read file content for ID generation: {e}")
                    document_id = self.metadata_manager.generate_document_id()
            
            # Step 2: Process document (same as parent method)
            success, message, doc_metadata = await super().add_document(
                file_path, metadata, force_reprocess
            )
            
            if not success or not doc_metadata:
                return success, message, doc_metadata
            
            # Step 3: Create metadata record if metadata management is enabled
            if self.enable_metadata and self.is_metadata_initialized:
                # Generate comprehensive metadata
                document_metadata = await self._generate_document_metadata(
                    file_path, doc_metadata, metadata
                )
                
                # Create metadata record
                metadata_success = self.metadata_manager.create_metadata_record(
                    document_id=doc_metadata.document_id,
                    schema_id=self.default_schema_id,
                    metadata=document_metadata,
                    user_id="system"
                )
                
                if metadata_success:
                    message += " | Metadata recorded"
                else:
                    message += " | Metadata recording failed"
            
            return success, message, doc_metadata
            
        except Exception as e:
            logger.error(f"Error adding document with metadata: {e}")
            return False, str(e), None
    
    async def _generate_document_metadata(self, file_path: Path, 
                                        doc_metadata: DocumentMetadata,
                                        user_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate comprehensive metadata for a document."""
        try:
            # Get file information
            stat = file_path.stat()
            
            # Base metadata from document metadata
            metadata = {
                'title': user_metadata.get('title', doc_metadata.original_filename) if user_metadata else doc_metadata.original_filename,
                'author': user_metadata.get('author', 'Unknown') if user_metadata else 'Unknown',
                'document_type': doc_metadata.document_type,
                'file_size': stat.st_size,
                'file_hash': doc_metadata.file_hash,
                'creation_date': doc_metadata.creation_date,
                'modification_date': doc_metadata.modification_date,
                'tags': user_metadata.get('tags', []) if user_metadata else [],
                'version': doc_metadata.version,
                'source': str(file_path),
                'department': user_metadata.get('department', 'General') if user_metadata else 'General',
                'classification': user_metadata.get('classification', 'internal') if user_metadata else 'internal',
                'language': user_metadata.get('language', 'en') if user_metadata else 'en',
                'page_count': doc_metadata.page_count,
                'word_count': doc_metadata.word_count,
                'custom_fields': user_metadata.get('custom_fields', {}) if user_metadata else {}
            }
            
            # Add document-specific metadata
            if doc_metadata.document_type == 'pdf':
                metadata.update({
                    'pdf_version': doc_metadata.metadata.get('pdf_version', ''),
                    'pdf_encrypted': doc_metadata.metadata.get('encrypted', False),
                    'pdf_pages': doc_metadata.metadata.get('pages', 0)
                })
            elif doc_metadata.document_type == 'docx':
                metadata.update({
                    'docx_author': doc_metadata.metadata.get('author', ''),
                    'docx_company': doc_metadata.metadata.get('company', ''),
                    'docx_revision': doc_metadata.metadata.get('revision', '')
                })
            elif doc_metadata.document_type == 'image':
                metadata.update({
                    'image_width': doc_metadata.metadata.get('width', 0),
                    'image_height': doc_metadata.metadata.get('height', 0),
                    'image_format': doc_metadata.metadata.get('format', ''),
                    'ocr_confidence': doc_metadata.metadata.get('ocr_confidence', 0.0)
                })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating document metadata: {e}")
            return {}
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a document."""
        if not self.enable_metadata or not self.is_metadata_initialized:
            return None
        
        try:
            record = self.metadata_manager.get_metadata_record(document_id)
            if record:
                return {
                    'document_id': record.document_id,
                    'schema_id': record.schema_id,
                    'metadata': record.metadata,
                    'version': record.version,
                    'lifecycle_stage': record.lifecycle_stage.value,
                    'created_at': record.created_at.isoformat(),
                    'updated_at': record.updated_at.isoformat(),
                    'validation_errors': record.validation_errors
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting document metadata: {e}")
            return None
    
    def update_document_metadata(self, document_id: str, metadata_updates: Dict[str, Any], 
                               user_id: Optional[str] = None, 
                               reason: Optional[str] = None) -> bool:
        """Update metadata for a document."""
        if not self.enable_metadata or not self.is_metadata_initialized:
            return False
        
        try:
            success = self.metadata_manager.update_metadata_record(
                document_id, metadata_updates, user_id, reason
            )
            if success:
                logger.info(f"Updated metadata for document: {document_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error updating document metadata: {e}")
            return False
    
    def get_document_audit_trail(self, document_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit trail for a document."""
        if not self.enable_metadata or not self.is_metadata_initialized:
            return []
        
        try:
            audit_entries = self.metadata_manager.get_audit_trail(document_id, limit)
            return [
                {
                    'field_name': entry.field_name,
                    'old_value': entry.old_value,
                    'new_value': entry.new_value,
                    'operation': entry.operation,
                    'timestamp': entry.timestamp.isoformat(),
                    'user_id': entry.user_id,
                    'reason': entry.reason
                }
                for entry in audit_entries
            ]
            
        except Exception as e:
            logger.error(f"Error getting document audit trail: {e}")
            return []
    
    def search_documents_by_metadata(self, query: Dict[str, Any], 
                                   schema_id: Optional[str] = None) -> List[str]:
        """Search documents by metadata."""
        if not self.enable_metadata or not self.is_metadata_initialized:
            return []
        
        try:
            return self.metadata_manager.search_metadata(query, schema_id)
            
        except Exception as e:
            logger.error(f"Error searching documents by metadata: {e}")
            return []
    
    def create_document_relationship(self, source_document_id: str, target_document_id: str, 
                                   relationship_type: str, relationship_data: Optional[Dict[str, Any]] = None,
                                   user_id: Optional[str] = None) -> bool:
        """Create a relationship between documents."""
        if not self.enable_metadata or not self.is_metadata_initialized:
            return False
        
        try:
            success = self.metadata_manager.create_relationship(
                source_document_id, target_document_id, relationship_type, 
                relationship_data, user_id
            )
            if success:
                logger.info(f"Created relationship: {source_document_id} -> {target_document_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error creating document relationship: {e}")
            return False
    
    def get_document_relationships(self, document_id: str, 
                                 relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get relationships for a document."""
        if not self.enable_metadata or not self.is_metadata_initialized:
            return []
        
        try:
            relationships = self.metadata_manager.get_relationships(document_id, relationship_type)
            return [
                {
                    'source_document_id': rel.source_document_id,
                    'target_document_id': rel.target_document_id,
                    'relationship_type': rel.relationship_type,
                    'relationship_data': rel.relationship_data,
                    'created_at': rel.created_at.isoformat(),
                    'created_by': rel.created_by
                }
                for rel in relationships
            ]
            
        except Exception as e:
            logger.error(f"Error getting document relationships: {e}")
            return []
    
    def get_metadata_statistics(self) -> Dict[str, Any]:
        """Get metadata management statistics."""
        if not self.enable_metadata or not self.is_metadata_initialized:
            return {'error': 'Metadata management not initialized'}
        
        try:
            return self.metadata_manager.get_statistics()
            
        except Exception as e:
            logger.error(f"Error getting metadata statistics: {e}")
            return {'error': str(e)}
    
    async def shutdown_metadata(self):
        """Shutdown the metadata management system."""
        if self.is_metadata_initialized and self.metadata_manager:
            try:
                # Metadata manager doesn't need explicit shutdown in current implementation
                self.is_metadata_initialized = False
                logger.info("Metadata management system shutdown complete")
            except Exception as e:
                logger.error(f"Error during metadata management shutdown: {e}")
    
    async def shutdown(self):
        """Shutdown the enhanced document manager with metadata."""
        # Shutdown metadata first
        await self.shutdown_metadata()
        
        # Then shutdown parent
        await super().shutdown()


class MetadataBatchProcessor:
    """Batch processor for metadata operations."""
    
    def __init__(self, enhanced_doc_manager: EnhancedDocumentManagerWithMetadata):
        self.doc_manager = enhanced_doc_manager
        self.processing_lock = asyncio.Lock()
    
    async def process_batch_metadata_update(self, document_ids: List[str], 
                                          metadata_updates: Dict[str, Any],
                                          user_id: Optional[str] = None,
                                          reason: Optional[str] = None) -> Dict[str, Any]:
        """Process metadata updates for a batch of documents."""
        if not self.doc_manager.is_metadata_initialized:
            return {'error': 'Metadata management not initialized'}
        
        results = {
            'total_documents': len(document_ids),
            'updated': 0,
            'failed': 0,
            'errors': []
        }
        
        async with self.processing_lock:
            for document_id in document_ids:
                try:
                    success = self.doc_manager.update_document_metadata(
                        document_id, metadata_updates, user_id, reason
                    )
                    
                    if success:
                        results['updated'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Failed to update metadata for {document_id}")
                
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Error updating {document_id}: {str(e)}")
        
        return results
    
    async def export_batch_metadata(self, document_ids: List[str], 
                                  format: str = 'json') -> Dict[str, Any]:
        """Export metadata for a batch of documents."""
        if not self.doc_manager.is_metadata_initialized:
            return {'error': 'Metadata management not initialized'}
        
        try:
            export_data = self.doc_manager.metadata_manager.export_metadata(document_ids, format)
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting batch metadata: {e}")
            return {'error': str(e)}
    
    async def import_batch_metadata(self, export_data: Dict[str, Any], 
                                  user_id: Optional[str] = None) -> Dict[str, Any]:
        """Import metadata for a batch of documents."""
        if not self.doc_manager.is_metadata_initialized:
            return {'error': 'Metadata management not initialized'}
        
        try:
            success = self.doc_manager.metadata_manager.import_metadata(export_data, user_id)
            return {
                'success': success,
                'imported_count': len(export_data.get('documents', [])) if success else 0
            }
            
        except Exception as e:
            logger.error(f"Error importing batch metadata: {e}")
            return {'error': str(e)}


# Convenience functions for easy integration
def create_enhanced_doc_manager_with_metadata(rag_system, **kwargs) -> EnhancedDocumentManagerWithMetadata:
    """Create an enhanced document manager with metadata management capabilities."""
    return EnhancedDocumentManagerWithMetadata(rag_system, **kwargs)


def create_metadata_batch_processor(enhanced_doc_manager: EnhancedDocumentManagerWithMetadata) -> MetadataBatchProcessor:
    """Create a batch processor for metadata operations."""
    return MetadataBatchProcessor(enhanced_doc_manager)


if __name__ == "__main__":
    
    # rag_system = RAGSystem()  # Assume RAGSystem is imported
    
    # Create enhanced document manager with metadata
    # metadata_config = {
    #     'metadata_db_path': 'metadata_manager.db',
    #     'enable_validation': True,
    #     'enable_auditing': True,
    #     'enable_relationships': True,
    #     'id_strategy': IDGenerationStrategy.HIERARCHICAL,
    #     'namespace': 'enterprise',
    #     'document_type': 'document'
    # }
    
    # doc_manager = EnhancedDocumentManagerWithMetadata(
    #     rag_system, 
    #     enable_metadata=True,
    #     metadata_config=metadata_config
    # )
    
    # await doc_manager.initialize_metadata()
    
    # Add a document with metadata
    # success, message, metadata = await doc_manager.add_document(Path("document.pdf"))
    
    # Get document metadata
    # doc_metadata = doc_manager.get_document_metadata(metadata.document_id)
    
    # Update document metadata
    # doc_manager.update_document_metadata(metadata.document_id, {'tags': ['important', 'finance']})
    
    # Search documents by metadata
    # results = doc_manager.search_documents_by_metadata({'classification': 'confidential'})
    
    pass