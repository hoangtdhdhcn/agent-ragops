"""
Versioning Integration Module

Integrates the advanced versioning system with the existing EnhancedDocumentManager
to provide seamless version control capabilities during document ingestion and management.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import asdict

from config import MARKDOWN_DIR
from enhanced_document_manager import EnhancedDocumentManager, DocumentMetadata
from versioning_system import (
    VersioningSystem, SemanticVersion, VersionBranch, VersionType, BranchType,
    VersionCommit, VersionDiff, VersionTag, VersionStatus, VersionPermission
)
from metadata_manager import MetadataManager, MetadataSchema, MetadataField, FieldConstraint, MetadataFieldType

logger = logging.getLogger(__name__)

class EnhancedDocumentManagerWithVersioning(EnhancedDocumentManager):
    """Enhanced document manager with integrated versioning capabilities."""
    
    def __init__(self, rag_system, enable_versioning=True, 
                 versioning_config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced document manager with versioning.
        
        Args:
            rag_system: The RAG system instance
            enable_versioning: Whether to enable versioning
            versioning_config: Versioning configuration
        """
        # Initialize parent class
        super().__init__(rag_system, enable_deduplication=True, enable_versioning=enable_versioning)
        
        # Versioning settings
        self.enable_versioning = enable_versioning
        self.versioning_config = versioning_config or self._get_default_versioning_config()
        
        # Initialize versioning system
        self.versioning_system = None
        self.is_versioning_initialized = False
        
        # Default branch
        self.default_branch = "main"
        
    def _get_default_versioning_config(self) -> Dict[str, Any]:
        """Get default versioning configuration."""
        return {
            'versioning_db_path': 'versioning_system.db',
            'storage_path': 'version_storage',
            'enable_semantic_versioning': True,
            'enable_branching': True,
            'enable_conflict_resolution': True,
            'enable_permissions': True
        }
    
    async def initialize_versioning(self):
        """Initialize the versioning system."""
        if not self.enable_versioning:
            logger.info("Versioning is disabled")
            return
        
        try:
            self.versioning_system = VersioningSystem(self.versioning_config)
            self.is_versioning_initialized = True
            logger.info("Versioning system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize versioning system: {e}")
            self.is_versioning_initialized = False
    
    async def add_document(self, file_path: Path, metadata: Optional[Dict] = None, 
                          force_reprocess: bool = False) -> Tuple[bool, str, Optional[DocumentMetadata]]:
        """
        Add a document with integrated versioning.
        
        This method overrides the parent method to add versioning
        during document processing.
        """
        try:
            # Step 1: Process document (same as parent method)
            success, message, doc_metadata = await super().add_document(
                file_path, metadata, force_reprocess
            )
            
            if not success or not doc_metadata:
                return success, message, doc_metadata
            
            # Step 2: Create version commit if versioning is enabled
            if self.enable_versioning and self.is_versioning_initialized:
                # Get document content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    logger.warning(f"Could not read file content for versioning: {e}")
                    content = ""
                
                # Create version commit
                commit_id = await self.versioning_system.create_commit(
                    document_id=doc_metadata.document_id,
                    content=content,
                    message=f"Initial commit for {doc_metadata.original_filename}",
                    author="system",
                    branch=self.default_branch,
                    metadata=doc_metadata.metadata
                )
                
                if commit_id:
                    message += f" | Versioned as commit {commit_id}"
                else:
                    message += " | Versioning failed"
            
            return success, message, doc_metadata
            
        except Exception as e:
            logger.error(f"Error adding document with versioning: {e}")
            return False, str(e), None
    
    async def update_document(self, document_id: str, new_content: str, 
                            message: str, author: Optional[str] = None,
                            branch: Optional[str] = None) -> Optional[str]:
        """Update a document and create a new version."""
        if not self.enable_versioning or not self.is_versioning_initialized:
            return None
        
        try:
            # Create version commit
            commit_id = await self.versioning_system.create_commit(
                document_id=document_id,
                content=new_content,
                message=message,
                author=author,
                branch=branch or self.default_branch
            )
            
            if commit_id:
                logger.info(f"Updated document {document_id} with commit {commit_id}")
            
            return commit_id
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return None
    
    async def get_document_version(self, document_id: str, version: SemanticVersion) -> Optional[str]:
        """Get content for a specific document version."""
        if not self.enable_versioning or not self.is_versioning_initialized:
            return None
        
        try:
            return await self.versioning_system.get_version_content(document_id, version)
        except Exception as e:
            logger.error(f"Error getting document version: {e}")
            return None
    
    async def list_document_versions(self, document_id: str, branch: Optional[str] = None) -> List[SemanticVersion]:
        """List all versions for a document."""
        if not self.enable_versioning or not self.is_versioning_initialized:
            return []
        
        try:
            return await self.versioning_system.list_versions(document_id, branch)
        except Exception as e:
            logger.error(f"Error listing document versions: {e}")
            return []
    
    async def compare_document_versions(self, document_id: str, version1: SemanticVersion, 
                                      version2: SemanticVersion) -> Optional[VersionDiff]:
        """Compare two versions of a document."""
        if not self.enable_versioning or not self.is_versioning_initialized:
            return None
        
        try:
            return await self.versioning_system.compare_versions(document_id, version1, version2)
        except Exception as e:
            logger.error(f"Error comparing document versions: {e}")
            return None
    
    async def create_document_branch(self, branch_name: str, branch_type: BranchType,
                                   base_version: SemanticVersion, document_id: str,
                                   description: str = "", created_by: Optional[str] = None) -> bool:
        """Create a new branch for a document."""
        if not self.enable_versioning or not self.is_versioning_initialized:
            return False
        
        try:
            return await self.versioning_system.create_branch(
                branch_name, branch_type, base_version, document_id, description, created_by
            )
        except Exception as e:
            logger.error(f"Error creating document branch: {e}")
            return False
    
    async def merge_document_branches(self, source_branch: str, target_branch: str,
                                    document_id: str, message: str, 
                                    author: Optional[str] = None) -> Optional[str]:
        """Merge changes from source branch to target branch."""
        if not self.enable_versioning or not self.is_versioning_initialized:
            return None
        
        try:
            return await self.versioning_system.merge_branches(
                source_branch, target_branch, document_id, message, author
            )
        except Exception as e:
            logger.error(f"Error merging document branches: {e}")
            return None
    
    async def create_document_tag(self, tag_name: str, version: SemanticVersion,
                                commit_id: str, document_id: str, description: str = "",
                                created_by: Optional[str] = None, is_release: bool = False) -> bool:
        """Create a tag for a document version."""
        if not self.enable_versioning or not self.is_versioning_initialized:
            return False
        
        try:
            return await self.versioning_system.create_tag(
                tag_name, version, commit_id, document_id, description, created_by, is_release
            )
        except Exception as e:
            logger.error(f"Error creating document tag: {e}")
            return False
    
    async def revert_document_to_version(self, document_id: str, target_version: SemanticVersion,
                                       message: str, author: Optional[str] = None) -> Optional[str]:
        """Revert document to a previous version."""
        if not self.enable_versioning or not self.is_versioning_initialized:
            return None
        
        try:
            return await self.versioning_system.revert_to_version(
                document_id, target_version, message, author
            )
        except Exception as e:
            logger.error(f"Error reverting document: {e}")
            return None
    
    async def get_document_commit_history(self, document_id: str, branch: Optional[str] = None,
                                        limit: int = 100) -> List[VersionCommit]:
        """Get commit history for a document."""
        if not self.enable_versioning or not self.is_versioning_initialized:
            return []
        
        try:
            return await self.versioning_system.get_commit_history(document_id, branch, limit)
        except Exception as e:
            logger.error(f"Error getting document commit history: {e}")
            return []
    
    async def get_document_branches(self, document_id: Optional[str] = None) -> List[VersionBranch]:
        """Get all branches for a document."""
        if not self.enable_versioning or not self.is_versioning_initialized:
            return []
        
        try:
            return await self.versioning_system.get_branches(document_id)
        except Exception as e:
            logger.error(f"Error getting document branches: {e}")
            return []
    
    async def get_document_tags(self, document_id: Optional[str] = None) -> List[VersionTag]:
        """Get all tags for a document."""
        if not self.enable_versioning or not self.is_versioning_initialized:
            return []
        
        try:
            return await self.versioning_system.get_tags(document_id)
        except Exception as e:
            logger.error(f"Error getting document tags: {e}")
            return []
    
    async def set_document_version_status(self, document_id: str, version: SemanticVersion,
                                        status: VersionStatus) -> bool:
        """Set status for a document version."""
        if not self.enable_versioning or not self.is_versioning_initialized:
            return False
        
        try:
            return await self.versioning_system.set_version_status(document_id, version, status)
        except Exception as e:
            logger.error(f"Error setting document version status: {e}")
            return False
    
    async def cleanup_document_versions(self, document_id: str, keep_count: int = 10) -> bool:
        """Clean up old versions for a document."""
        if not self.enable_versioning or not self.is_versioning_initialized:
            return False
        
        try:
            return await self.versioning_system.cleanup_old_versions(document_id, keep_count)
        except Exception as e:
            logger.error(f"Error cleaning up document versions: {e}")
            return False
    
    def get_versioning_statistics(self) -> Dict[str, Any]:
        """Get versioning system statistics."""
        if not self.enable_versioning or not self.is_versioning_initialized:
            return {'error': 'Versioning system not initialized'}
        
        try:
            return self.versioning_system.get_statistics()
        except Exception as e:
            logger.error(f"Error getting versioning statistics: {e}")
            return {'error': str(e)}
    
    async def shutdown_versioning(self):
        """Shutdown the versioning system."""
        if self.is_versioning_initialized and self.versioning_system:
            try:
                # Versioning system doesn't need explicit shutdown in current implementation
                self.is_versioning_initialized = False
                logger.info("Versioning system shutdown complete")
            except Exception as e:
                logger.error(f"Error during versioning system shutdown: {e}")
    
    async def shutdown(self):
        """Shutdown the enhanced document manager with versioning."""
        # Shutdown versioning first
        await self.shutdown_versioning()
        
        # Then shutdown parent
        await super().shutdown()


class VersioningBatchProcessor:
    """Batch processor for versioning operations."""
    
    def __init__(self, enhanced_doc_manager: EnhancedDocumentManagerWithVersioning):
        self.doc_manager = enhanced_doc_manager
        self.processing_lock = asyncio.Lock()
    
    async def process_batch_version_updates(self, document_ids: List[str], 
                                          new_content: str, message: str,
                                          author: Optional[str] = None,
                                          branch: Optional[str] = None) -> Dict[str, Any]:
        """Process version updates for a batch of documents."""
        if not self.doc_manager.is_versioning_initialized:
            return {'error': 'Versioning system not initialized'}
        
        results = {
            'total_documents': len(document_ids),
            'updated': 0,
            'failed': 0,
            'errors': []
        }
        
        async with self.processing_lock:
            for document_id in document_ids:
                try:
                    commit_id = await self.doc_manager.update_document(
                        document_id, new_content, message, author, branch
                    )
                    
                    if commit_id:
                        results['updated'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Failed to update version for {document_id}")
                
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Error updating {document_id}: {str(e)}")
        
        return results
    
    async def process_batch_version_comparisons(self, document_ids: List[str], 
                                              version1: SemanticVersion,
                                              version2: SemanticVersion) -> Dict[str, Any]:
        """Process version comparisons for a batch of documents."""
        if not self.doc_manager.is_versioning_initialized:
            return {'error': 'Versioning system not initialized'}
        
        results = {
            'total_documents': len(document_ids),
            'compared': 0,
            'failed': 0,
            'differences': {}
        }
        
        async with self.processing_lock:
            for document_id in document_ids:
                try:
                    diff = await self.doc_manager.compare_document_versions(
                        document_id, version1, version2
                    )
                    
                    if diff:
                        results['compared'] += 1
                        results['differences'][document_id] = {
                            'diff_type': diff.diff_type,
                            'changes_count': len(diff.changes),
                            'summary': diff.diff_summary
                        }
                    else:
                        results['failed'] += 1
                
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Error comparing {document_id}: {str(e)}")
        
        return results
    
    async def process_batch_version_cleanup(self, document_ids: List[str], 
                                          keep_count: int = 10) -> Dict[str, Any]:
        """Process version cleanup for a batch of documents."""
        if not self.doc_manager.is_versioning_initialized:
            return {'error': 'Versioning system not initialized'}
        
        results = {
            'total_documents': len(document_ids),
            'cleaned': 0,
            'failed': 0,
            'errors': []
        }
        
        async with self.processing_lock:
            for document_id in document_ids:
                try:
                    success = await self.doc_manager.cleanup_document_versions(document_id, keep_count)
                    
                    if success:
                        results['cleaned'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Failed to cleanup versions for {document_id}")
                
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Error cleaning up {document_id}: {str(e)}")
        
        return results


class VersioningWorkflowManager:
    """Manager for versioning workflows and approval processes."""
    
    def __init__(self, enhanced_doc_manager: EnhancedDocumentManagerWithVersioning):
        self.doc_manager = enhanced_doc_manager
        self.workflows = {}  # document_id -> workflow_state
    
    async def start_review_workflow(self, document_id: str, version: SemanticVersion,
                                  reviewers: List[str], deadline: Optional[datetime] = None) -> bool:
        """Start a review workflow for a document version."""
        if not self.doc_manager.is_versioning_initialized:
            return False
        
        try:
            # Set version status to review
            success = await self.doc_manager.set_document_version_status(
                document_id, version, VersionStatus.REVIEW
            )
            
            if success:
                # Store workflow state
                self.workflows[document_id] = {
                    'version': version,
                    'status': 'review',
                    'reviewers': reviewers,
                    'deadline': deadline,
                    'reviews': {},
                    'approved': False
                }
                
                logger.info(f"Started review workflow for {document_id} version {version}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting review workflow: {e}")
            return False
    
    async def submit_review(self, document_id: str, reviewer: str, 
                          approval: bool, comments: str = "") -> bool:
        """Submit a review for a document version."""
        if document_id not in self.workflows:
            return False
        
        try:
            workflow = self.workflows[document_id]
            workflow['reviews'][reviewer] = {
                'approval': approval,
                'comments': comments,
                'timestamp': datetime.now()
            }
            
            # Check if all reviews are complete
            all_reviews = workflow['reviews']
            if len(all_reviews) == len(workflow['reviewers']):
                approvals = sum(1 for r in all_reviews.values() if r['approval'])
                total_reviews = len(all_reviews)
                
                if approvals >= total_reviews // 2 + 1:  # Simple majority approval
                    # Set version status to approved
                    success = await self.doc_manager.set_document_version_status(
                        document_id, workflow['version'], VersionStatus.APPROVED
                    )
                    
                    if success:
                        workflow['approved'] = True
                        logger.info(f"Document {document_id} version {workflow['version']} approved")
                
            return True
            
        except Exception as e:
            logger.error(f"Error submitting review: {e}")
            return False
    
    async def get_workflow_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow."""
        return self.workflows.get(document_id)
    
    async def complete_workflow(self, document_id: str) -> bool:
        """Complete a workflow and make the version active."""
        if document_id not in self.workflows:
            return False
        
        try:
            workflow = self.workflows[document_id]
            if workflow['approved']:
                # Set version status to active
                success = await self.doc_manager.set_document_version_status(
                    document_id, workflow['version'], VersionStatus.APPROVED
                )
                
                if success:
                    del self.workflows[document_id]
                    logger.info(f"Completed workflow for {document_id}")
                
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Error completing workflow: {e}")
            return False


# Convenience functions for easy integration
def create_enhanced_doc_manager_with_versioning(rag_system, **kwargs) -> EnhancedDocumentManagerWithVersioning:
    """Create an enhanced document manager with versioning capabilities."""
    return EnhancedDocumentManagerWithVersioning(rag_system, **kwargs)


def create_versioning_batch_processor(enhanced_doc_manager: EnhancedDocumentManagerWithVersioning) -> VersioningBatchProcessor:
    """Create a batch processor for versioning operations."""
    return VersioningBatchProcessor(enhanced_doc_manager)


def create_versioning_workflow_manager(enhanced_doc_manager: EnhancedDocumentManagerWithVersioning) -> VersioningWorkflowManager:
    """Create a workflow manager for versioning processes."""
    return VersioningWorkflowManager(enhanced_doc_manager)


if __name__ == "__main__":
    
    # rag_system = RAGSystem()  # Assume RAGSystem is imported
    
    # Create enhanced document manager with versioning
    # versioning_config = {
    #     'versioning_db_path': 'versioning_system.db',
    #     'storage_path': 'version_storage',
    #     'enable_semantic_versioning': True,
    #     'enable_branching': True,
    #     'enable_conflict_resolution': True,
    #     'enable_permissions': True
    # }
    
    # doc_manager = EnhancedDocumentManagerWithVersioning(
    #     rag_system, 
    #     enable_versioning=True,
    #     versioning_config=versioning_config
    # )
    
    # await doc_manager.initialize_versioning()
    
    # Add a document with versioning
    # success, message, metadata = await doc_manager.add_document(Path("document.pdf"))
    
    # Update document version
    # commit_id = await doc_manager.update_document(metadata.document_id, "New content", "Updated content")
    
    # Compare versions
    # diff = await doc_manager.compare_document_versions(metadata.document_id, 
    #                                                  SemanticVersion(1, 0, 0), 
    #                                                  SemanticVersion(1, 1, 0))
    
    # Create branch
    # await doc_manager.create_document_branch("feature_1", BranchType.FEATURE, 
    #                                        SemanticVersion(1, 0, 0), metadata.document_id)
    
    # Start review workflow
    # workflow_manager = create_versioning_workflow_manager(doc_manager)
    # await workflow_manager.start_review_workflow(metadata.document_id, SemanticVersion(1, 1, 0), 
    #                                            ["reviewer1", "reviewer2"])
    
    pass