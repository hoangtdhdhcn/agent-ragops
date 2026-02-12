"""
Versioning Example: Complete Production RAG System with Advanced Versioning

This example demonstrates how to integrate the advanced versioning system
into a complete production RAG system.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

from config import MARKDOWN_DIR, PARENT_STORE_PATH, QDRANT_DB_PATH
from enhanced_document_manager import EnhancedDocumentManager
from versioning_system import VersioningSystem, SemanticVersion, VersionBranch, BranchType, VersionStatus
from versioning_integration import (
    EnhancedDocumentManagerWithVersioning,
    create_enhanced_doc_manager_with_versioning,
    create_versioning_batch_processor,
    create_versioning_workflow_manager
)
from metadata_manager import MetadataManager, MetadataSchema, MetadataField, FieldConstraint, MetadataFieldType
from metadata_integration import create_enhanced_doc_manager_with_metadata
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

class ProductionRAGSystemWithVersioning:
    """Complete production-ready RAG system with advanced versioning capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize production RAG system with versioning.
        
        Args:
            config: Configuration dictionary with system settings
        """
        self.config = config
        self.rag_system = None
        self.doc_manager = None
        self.versioning_system = None
        self.versioning_batch_processor = None
        self.versioning_workflow_manager = None
        self.deduplication_engine = None
        self.delta_processor = None
        self.monitoring_system = None
        
        # System state
        self.is_initialized = False
        self.is_monitoring = False
        
    async def initialize(self):
        """Initialize the complete production system with versioning."""
        try:
            logger.info("Initializing Production RAG System with Versioning...")
            
            # Initialize core RAG system
            self.rag_system = RAGSystem()
            await self.rag_system.initialize()
            
            # Initialize enhanced document manager with versioning
            versioning_config = self.config.get('versioning_config', {})
            self.doc_manager = create_enhanced_doc_manager_with_versioning(
                self.rag_system,
                enable_versioning=self.config.get('enable_versioning', True),
                versioning_config=versioning_config
            )
            
            # Initialize versioning system
            await self.doc_manager.initialize_versioning()
            
            # Initialize versioning batch processor
            self.versioning_batch_processor = create_versioning_batch_processor(self.doc_manager)
            
            # Initialize versioning workflow manager
            self.versioning_workflow_manager = create_versioning_workflow_manager(self.doc_manager)
            
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
            logger.info("Production RAG System with Versioning initialized successfully")
            
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
        """Ingest documents with full production features including versioning."""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        try:
            # Use enhanced document manager with versioning for ingestion
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
        """Ingest web pages with full production features including versioning."""
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
        """Get comprehensive system status including versioning metrics."""
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        try:
            # Get document statistics
            doc_stats = self.doc_manager.get_statistics()
            
            # Get versioning statistics
            versioning_stats = self.doc_manager.get_versioning_statistics()
            
            # Get processing status
            processing_status = self.delta_processor.get_queue_status()
            
            # Get monitoring status
            system_health = self.monitoring_system['metrics_collector'].get_system_health()
            processing_health = self.monitoring_system['metrics_collector'].get_processing_health()
            alert_summary = self.monitoring_system['alert_manager'].get_alert_summary()
            
            return {
                'status': 'running',
                'document_stats': doc_stats,
                'versioning_stats': versioning_stats,
                'processing_status': processing_status,
                'system_health': system_health,
                'processing_health': processing_health,
                'alerts': alert_summary,
                'monitoring_active': self.is_monitoring
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_document_versions(self, document_id: str, branch: Optional[str] = None) -> List[SemanticVersion]:
        """Get all versions for a document."""
        if not self.is_initialized:
            return []
        
        try:
            return self.doc_manager.list_document_versions(document_id, branch)
        except Exception as e:
            logger.error(f"Error getting document versions: {e}")
            return []
    
    def compare_document_versions(self, document_id: str, version1: SemanticVersion, 
                                version2: SemanticVersion) -> Optional[Dict[str, Any]]:
        """Compare two versions of a document."""
        if not self.is_initialized:
            return None
        
        try:
            diff = self.doc_manager.compare_document_versions(document_id, version1, version2)
            if diff:
                return {
                    'from_version': diff.from_version,
                    'to_version': diff.to_version,
                    'diff_type': diff.diff_type,
                    'changes_count': len(diff.changes),
                    'summary': diff.diff_summary,
                    'created_at': diff.created_at.isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Error comparing document versions: {e}")
            return None
    
    def update_document_version(self, document_id: str, new_content: str, 
                              message: str, author: Optional[str] = None,
                              branch: Optional[str] = None) -> Optional[str]:
        """Update a document and create a new version."""
        if not self.is_initialized:
            return None
        
        try:
            return self.doc_manager.update_document(document_id, new_content, message, author, branch)
        except Exception as e:
            logger.error(f"Error updating document version: {e}")
            return None
    
    def create_document_branch(self, branch_name: str, branch_type: BranchType,
                             base_version: SemanticVersion, document_id: str,
                             description: str = "", created_by: Optional[str] = None) -> bool:
        """Create a new branch for a document."""
        if not self.is_initialized:
            return False
        
        try:
            return self.doc_manager.create_document_branch(
                branch_name, branch_type, base_version, document_id, description, created_by
            )
        except Exception as e:
            logger.error(f"Error creating document branch: {e}")
            return False
    
    def merge_document_branches(self, source_branch: str, target_branch: str,
                              document_id: str, message: str, 
                              author: Optional[str] = None) -> Optional[str]:
        """Merge changes from source branch to target branch."""
        if not self.is_initialized:
            return None
        
        try:
            return self.doc_manager.merge_document_branches(
                source_branch, target_branch, document_id, message, author
            )
        except Exception as e:
            logger.error(f"Error merging document branches: {e}")
            return None
    
    def create_document_tag(self, tag_name: str, version: SemanticVersion,
                          commit_id: str, document_id: str, description: str = "",
                          created_by: Optional[str] = None, is_release: bool = False) -> bool:
        """Create a tag for a document version."""
        if not self.is_initialized:
            return False
        
        try:
            return self.doc_manager.create_document_tag(
                tag_name, version, commit_id, document_id, description, created_by, is_release
            )
        except Exception as e:
            logger.error(f"Error creating document tag: {e}")
            return False
    
    def revert_document_to_version(self, document_id: str, target_version: SemanticVersion,
                                 message: str, author: Optional[str] = None) -> Optional[str]:
        """Revert document to a previous version."""
        if not self.is_initialized:
            return None
        
        try:
            return self.doc_manager.revert_document_to_version(
                document_id, target_version, message, author
            )
        except Exception as e:
            logger.error(f"Error reverting document: {e}")
            return None
    
    def get_document_commit_history(self, document_id: str, branch: Optional[str] = None,
                                  limit: int = 100) -> List[Dict[str, Any]]:
        """Get commit history for a document."""
        if not self.is_initialized:
            return []
        
        try:
            commits = self.doc_manager.get_document_commit_history(document_id, branch, limit)
            return [
                {
                    'commit_id': commit.commit_id,
                    'version': str(commit.version),
                    'branch': commit.branch,
                    'message': commit.message,
                    'author': commit.author,
                    'created_at': commit.created_at.isoformat(),
                    'status': commit.status.value,
                    'change_summary': commit.change_summary
                }
                for commit in commits
            ]
        except Exception as e:
            logger.error(f"Error getting document commit history: {e}")
            return []
    
    def get_document_branches(self, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all branches for a document."""
        if not self.is_initialized:
            return []
        
        try:
            branches = self.doc_manager.get_document_branches(document_id)
            return [
                {
                    'branch_name': branch.branch_name,
                    'branch_type': branch.branch_type.value,
                    'base_version': str(branch.base_version),
                    'created_at': branch.created_at.isoformat(),
                    'created_by': branch.created_by,
                    'description': branch.description,
                    'is_active': branch.is_active,
                    'parent_branch': branch.parent_branch
                }
                for branch in branches
            ]
        except Exception as e:
            logger.error(f"Error getting document branches: {e}")
            return []
    
    def get_document_tags(self, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all tags for a document."""
        if not self.is_initialized:
            return []
        
        try:
            tags = self.doc_manager.get_document_tags(document_id)
            return [
                {
                    'tag_name': tag.tag_name,
                    'version': str(tag.version),
                    'commit_id': tag.commit_id,
                    'created_at': tag.created_at.isoformat(),
                    'created_by': tag.created_by,
                    'description': tag.description,
                    'is_release': tag.is_release
                }
                for tag in tags
            ]
        except Exception as e:
            logger.error(f"Error getting document tags: {e}")
            return []
    
    def set_document_version_status(self, document_id: str, version: SemanticVersion,
                                  status: VersionStatus) -> bool:
        """Set status for a document version."""
        if not self.is_initialized:
            return False
        
        try:
            return self.doc_manager.set_document_version_status(document_id, version, status)
        except Exception as e:
            logger.error(f"Error setting document version status: {e}")
            return False
    
    def cleanup_document_versions(self, document_id: str, keep_count: int = 10) -> bool:
        """Clean up old versions for a document."""
        if not self.is_initialized:
            return False
        
        try:
            return self.doc_manager.cleanup_document_versions(document_id, keep_count)
        except Exception as e:
            logger.error(f"Error cleaning up document versions: {e}")
            return False
    
    async def start_review_workflow(self, document_id: str, version: SemanticVersion,
                                  reviewers: List[str], deadline: Optional[str] = None) -> bool:
        """Start a review workflow for a document version."""
        if not self.is_initialized:
            return False
        
        try:
            deadline_dt = None
            if deadline:
                deadline_dt = datetime.fromisoformat(deadline)
            
            return await self.versioning_workflow_manager.start_review_workflow(
                document_id, version, reviewers, deadline_dt
            )
        except Exception as e:
            logger.error(f"Error starting review workflow: {e}")
            return False
    
    async def submit_review(self, document_id: str, reviewer: str, 
                          approval: bool, comments: str = "") -> bool:
        """Submit a review for a document version."""
        if not self.is_initialized:
            return False
        
        try:
            return await self.versioning_workflow_manager.submit_review(
                document_id, reviewer, approval, comments
            )
        except Exception as e:
            logger.error(f"Error submitting review: {e}")
            return False
    
    async def get_workflow_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow."""
        if not self.is_initialized:
            return None
        
        try:
            return await self.versioning_workflow_manager.get_workflow_status(document_id)
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return None
    
    async def complete_workflow(self, document_id: str) -> bool:
        """Complete a workflow and make the version active."""
        if not self.is_initialized:
            return False
        
        try:
            return await self.versioning_workflow_manager.complete_workflow(document_id)
        except Exception as e:
            logger.error(f"Error completing workflow: {e}")
            return False
    
    async def run_batch_version_operations(self, document_ids: List[str], 
                                         operation: str, 
                                         operation_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run batch version operations."""
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            if operation == 'update':
                results = await self.versioning_batch_processor.process_batch_version_updates(
                    document_ids, 
                    operation_data.get('content', ''), 
                    operation_data.get('message', 'Batch update'),
                    operation_data.get('author'),
                    operation_data.get('branch')
                )
            elif operation == 'compare':
                results = await self.versioning_batch_processor.process_batch_version_comparisons(
                    document_ids,
                    SemanticVersion.parse(operation_data.get('version1', '1.0.0')),
                    SemanticVersion.parse(operation_data.get('version2', '1.1.0'))
                )
            elif operation == 'cleanup':
                results = await self.versioning_batch_processor.process_batch_version_cleanup(
                    document_ids,
                    operation_data.get('keep_count', 10)
                )
            else:
                return {'error': f'Unknown operation: {operation}'}
            
            logger.info(f"Batch version operation completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to run batch version operations: {e}")
            return {'error': str(e)}
    
    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive system report including versioning metrics."""
        try:
            # Get base report
            report = self.monitoring_system['dashboard'].generate_report(hours)
            
            # Add versioning management metrics
            versioning_stats = self.doc_manager.get_versioning_statistics()
            report['versioning_management'] = versioning_stats
            
            # Add versioning configuration
            report['versioning_config'] = {
                'enabled': self.config.get('enable_versioning', True),
                'semantic_versioning_enabled': self.config.get('versioning_config', {}).get('enable_semantic_versioning', True),
                'branching_enabled': self.config.get('versioning_config', {}).get('enable_branching', True),
                'conflict_resolution_enabled': self.config.get('versioning_config', {}).get('enable_conflict_resolution', True),
                'permissions_enabled': self.config.get('versioning_config', {}).get('enable_permissions', True),
                'storage_path': self.config.get('versioning_config', {}).get('storage_path', 'version_storage'),
                'default_branch': 'main'
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Shutdown the production system with versioning gracefully."""
        try:
            # Stop monitoring
            await self.stop_monitoring()
            
            # Shutdown versioning system
            await self.doc_manager.shutdown_versioning()
            
            # Stop monitoring system
            if self.monitoring_system:
                self.monitoring_system['metrics_collector'].stop_collection()
                self.monitoring_system['dashboard'].stop_dashboard()
            
            # Clear all data if configured
            if self.config.get('clear_on_shutdown', False):
                self.doc_manager.clear_all()
            
            self.is_initialized = False
            logger.info("Production RAG System with Versioning shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def main():
    """Example usage of the production RAG system with versioning."""
    
    # Configuration
    config = {
        'enable_versioning': True,
        'enable_deduplication': True,
        'enable_exact_dedup': True,
        'enable_semantic_dedup': True,
        'enable_fuzzy_dedup': True,
        'semantic_threshold': 0.85,
        'fuzzy_threshold': 0.90,
        'auto_resolve_duplicates': True,
        'resolution_strategy': 'keep_newest',
        'versioning_config': {
            'versioning_db_path': 'versioning_system.db',
            'storage_path': 'version_storage',
            'enable_semantic_versioning': True,
            'enable_branching': True,
            'enable_conflict_resolution': True,
            'enable_permissions': True
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
    system = ProductionRAGSystemWithVersioning(config)
    await system.initialize()
    
    # Example 1: Ingest documents with versioning
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
    
    # Example 2: Ingest web pages with versioning
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
    
    # Example 3: Start monitoring with versioning
    await system.start_monitoring(document_dirs)
    
    # Example 4: Get system status with versioning metrics
    status = system.get_system_status()
    print(f"System status: {status}")
    
    # Example 5: Get document versions
    if status.get('document_stats', {}).get('total_documents', 0) > 0:
        # Get first document ID
        documents = system.doc_manager.list_documents()
        if documents:
            first_doc_id = documents[0]['document_id']
            
            # Get document versions
            versions = system.get_document_versions(first_doc_id)
            print(f"Document versions: {[str(v) for v in versions]}")
            
            # Update document version
            update_commit = system.update_document_version(
                first_doc_id, 
                "Updated document content with new information",
                "Updated content for testing",
                author='admin'
            )
            print(f"Version update commit: {update_commit}")
            
            # Compare versions
            if len(versions) >= 2:
                diff = system.compare_document_versions(first_doc_id, versions[0], versions[1])
                print(f"Version comparison: {diff}")
    
    # Example 6: Create document branches
    if status.get('document_stats', {}).get('total_documents', 0) > 0:
        documents = system.doc_manager.list_documents()
        if documents:
            doc_id = documents[0]['document_id']
            
            branch_success = system.create_document_branch(
                "feature_1", BranchType.FEATURE, 
                SemanticVersion(1, 0, 0), doc_id,
                description="Feature branch for new functionality"
            )
            print(f"Branch creation success: {branch_success}")
    
    # Example 7: Create document tags
    if status.get('document_stats', {}).get('total_documents', 0) > 0:
        documents = system.doc_manager.list_documents()
        if documents:
            doc_id = documents[0]['document_id']
            
            tag_success = system.create_document_tag(
                "v1.0.0", SemanticVersion(1, 0, 0), 
                "commit_12345", doc_id,
                description="Initial release",
                is_release=True
            )
            print(f"Tag creation success: {tag_success}")
    
    # Example 8: Start review workflow
    if status.get('document_stats', {}).get('total_documents', 0) > 0:
        documents = system.doc_manager.list_documents()
        if documents:
            doc_id = documents[0]['document_id']
            
            workflow_success = await system.start_review_workflow(
                doc_id, SemanticVersion(1, 1, 0), 
                ["reviewer1", "reviewer2", "reviewer3"],
                deadline="2024-01-15T17:00:00"
            )
            print(f"Workflow start success: {workflow_success}")
    
    # Example 9: Batch version operations
    if status.get('document_stats', {}).get('total_documents', 0) > 0:
        documents = system.doc_manager.list_documents()
        if len(documents) >= 3:
            doc_ids = [doc['document_id'] for doc in documents[:3]]
            
            batch_results = await system.run_batch_version_operations(
                doc_ids, 
                'update', 
                {
                    'content': 'Batch updated content',
                    'message': 'Batch version update',
                    'author': 'admin'
                }
            )
            print(f"Batch version update results: {batch_results}")
    
    # Example 10: Generate comprehensive report
    await asyncio.sleep(120)  # Wait 2 minutes
    
    report = system.generate_report(hours=1)
    print(f"System report: {report}")
    
    # Shutdown
    await system.shutdown()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())