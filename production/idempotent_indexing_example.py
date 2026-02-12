"""
Complete Idempotent Indexing Example

Demonstrates how to use the advanced idempotent indexing system with the enhanced
document manager for production RAG operations.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import json

from config import MARKDOWN_DIR
from core.rag_system import RAGSystem
from enhanced_document_manager import EnhancedDocumentManager
from idempotent_indexing import (
    IdempotentIndexingSystem, IndexType, IndexKey, IndexEntry, 
    IndexConsistencyLevel, IndexStatus, IndexOperation
)
from idempotent_indexing_integration import (
    EnhancedDocumentManagerWithIdempotentIndexing,
    IdempotentIndexingBatchProcessor,
    IdempotentIndexingWorkflowManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demonstrate_basic_idempotent_indexing():
    """Demonstrate basic idempotent indexing operations."""
    logger.info("=== Basic Idempotent Indexing Demo ===")
    
    # Initialize idempotent indexing system
    config = {
        'indexing_db_path': 'demo_idempotent_indexing.db',
        'enable_deterministic_indexing': True,
        'enable_index_versioning': True,
        'enable_index_consistency': True,
        'enable_index_optimization': True,
        'enable_distributed_indexing': False
    }
    
    indexing_system = IdempotentIndexingSystem(config)
    
    # Sample documents
    documents = [
        {
            'document_id': 'doc_001',
            'content': 'This is the first document about machine learning and AI.',
            'vector': [0.1, 0.2, 0.3, 0.4, 0.5] * 307,  # 1536 dimensions
            'metadata': {'author': 'John Doe', 'category': 'AI', 'tags': ['machine learning', 'AI']}
        },
        {
            'document_id': 'doc_002', 
            'content': 'This document discusses data science and analytics.',
            'vector': [0.2, 0.3, 0.4, 0.5, 0.6] * 307,  # 1536 dimensions
            'metadata': {'author': 'Jane Smith', 'category': 'Data Science', 'tags': ['data', 'analytics']}
        },
        {
            'document_id': 'doc_003',
            'content': 'Natural language processing and text analysis techniques.',
            'vector': [0.3, 0.4, 0.5, 0.6, 0.7] * 307,  # 1536 dimensions
            'metadata': {'author': 'Bob Wilson', 'category': 'NLP', 'tags': ['NLP', 'text analysis']}
        }
    ]
    
    # Index documents
    logger.info("Indexing documents...")
    index_keys = []
    
    for doc in documents:
        index_key = await indexing_system.index_document(
            document_id=doc['document_id'],
            content=doc['content'],
            vector=doc['vector'],
            metadata=doc['metadata'],
            index_type=IndexType.DENSE_VECTOR,
            consistency_level=IndexConsistencyLevel.STRONG
        )
        
        if index_key:
            index_keys.append(index_key)
            logger.info(f"Indexed {doc['document_id']} with key: {index_key}")
        else:
            logger.error(f"Failed to index {doc['document_id']}")
    
    # Validate indexes
    logger.info("Validating indexes...")
    for index_key in index_keys:
        is_valid = await indexing_system.validate_index(index_key)
        logger.info(f"Index {index_key} is {'valid' if is_valid else 'invalid'}")
    
    # Get index statistics
    stats = await indexing_system.get_index_statistics()
    logger.info(f"Index statistics: {json.dumps(stats, indent=2)}")
    
    # Get index health
    health = await indexing_system.get_index_health(IndexType.DENSE_VECTOR)
    if health:
        logger.info(f"Index health: {health}")
    
    # Shutdown
    await indexing_system.shutdown()
    
    return index_keys

async def demonstrate_enhanced_document_manager_with_indexing():
    """Demonstrate enhanced document manager with integrated idempotent indexing."""
    logger.info("=== Enhanced Document Manager with Idempotent Indexing Demo ===")
    
    # Initialize RAG system
    rag_system = RAGSystem()
    
    # Initialize enhanced document manager with idempotent indexing
    indexing_config = {
        'indexing_db_path': 'demo_enhanced_indexing.db',
        'enable_deterministic_indexing': True,
        'enable_index_versioning': True,
        'enable_index_consistency': True,
        'enable_index_optimization': True,
        'enable_distributed_indexing': False
    }
    
    doc_manager = EnhancedDocumentManagerWithIdempotentIndexing(
        rag_system,
        enable_idempotent_indexing=True,
        indexing_config=indexing_config
    )
    
    await doc_manager.initialize_idempotent_indexing()
    
    # Create sample documents
    sample_docs = [
        {
            'document_id': 'sample_001',
            'content': 'Sample document content about artificial intelligence and machine learning.',
            'vector': [0.1, 0.15, 0.2, 0.25, 0.3] * 307,
            'metadata': {'author': 'AI Expert', 'category': 'Technology'}
        },
        {
            'document_id': 'sample_002',
            'content': 'Sample document content about data analysis and visualization.',
            'vector': [0.2, 0.25, 0.3, 0.35, 0.4] * 307,
            'metadata': {'author': 'Data Analyst', 'category': 'Analytics'}
        }
    ]
    
    # Add documents with integrated indexing
    logger.info("Adding documents with integrated idempotent indexing...")
    
    for doc in sample_docs:
        # Create a temporary file for the document
        temp_file = Path(f"temp_{doc['document_id']}.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(doc['content'])
        
        try:
            success, message, metadata = await doc_manager.add_document(temp_file)
            logger.info(f"Added document: {message}")
            
            if metadata:
                logger.info(f"Document metadata: {metadata.metadata}")
        
        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
    
    # Manual indexing operations
    logger.info("Performing manual indexing operations...")
    
    for doc in sample_docs:
        index_key = await doc_manager.index_document(
            doc['document_id'],
            doc['content'],
            doc['vector'],
            doc['metadata']
        )
        
        if index_key:
            logger.info(f"Manually indexed {doc['document_id']} with key: {index_key}")
            
            # Validate the index
            is_valid = await doc_manager.validate_index(index_key)
            logger.info(f"Index validation for {doc['document_id']}: {'valid' if is_valid else 'invalid'}")
    
    # Get index statistics
    stats = await doc_manager.get_index_statistics()
    logger.info(f"Enhanced document manager index statistics: {json.dumps(stats, indent=2)}")
    
    # Get index entries
    entries = await doc_manager.get_index_entries(IndexType.DENSE_VECTOR)
    logger.info(f"Found {len(entries)} index entries")
    
    # Shutdown
    await doc_manager.shutdown()
    
    return True

async def demonstrate_batch_indexing_operations():
    """Demonstrate batch indexing operations."""
    logger.info("=== Batch Indexing Operations Demo ===")
    
    # Initialize RAG system and enhanced document manager
    rag_system = RAGSystem()
    
    indexing_config = {
        'indexing_db_path': 'demo_batch_indexing.db',
        'enable_deterministic_indexing': True,
        'enable_index_versioning': True,
        'enable_index_consistency': True,
        'enable_index_optimization': True,
        'enable_distributed_indexing': False
    }
    
    doc_manager = EnhancedDocumentManagerWithIdempotentIndexing(
        rag_system,
        enable_idempotent_indexing=True,
        indexing_config=indexing_config
    )
    
    await doc_manager.initialize_idempotent_indexing()
    
    # Create batch processor
    batch_processor = IdempotentIndexingBatchProcessor(doc_manager)
    
    # Create large batch of documents
    batch_documents = []
    for i in range(10):
        doc = {
            'document_id': f'batch_doc_{i:03d}',
            'content': f'This is batch document {i} with content about topic {i}.',
            'vector': [float(i)/10 + j/100 for j in range(1536)],
            'metadata': {
                'author': f'Author {i}',
                'category': f'Category {i % 3}',
                'batch_id': 'demo_batch_001',
                'created_at': time.time()
            }
        }
        batch_documents.append(doc)
    
    logger.info(f"Processing batch of {len(batch_documents)} documents...")
    
    # Process batch indexing
    start_time = time.time()
    batch_results = await batch_processor.process_batch_indexing(
        batch_documents, 
        IndexType.DENSE_VECTOR
    )
    end_time = time.time()
    
    logger.info(f"Batch indexing completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Batch results: {batch_results}")
    
    # Validate batch indexes
    index_keys = [f"batch_doc_{i:03d}_v{int(time.time())}" for i in range(10)]
    validation_results = await batch_processor.process_batch_validation(index_keys)
    
    logger.info(f"Batch validation results: {validation_results}")
    
    # Process batch optimization
    optimization_results = await batch_processor.process_batch_optimization([IndexType.DENSE_VECTOR])
    logger.info(f"Batch optimization results: {optimization_results}")
    
    # Shutdown
    await doc_manager.shutdown()
    
    return batch_results

async def demonstrate_indexing_workflows():
    """Demonstrate indexing workflows and management."""
    logger.info("=== Indexing Workflows Demo ===")
    
    # Initialize RAG system and enhanced document manager
    rag_system = RAGSystem()
    
    indexing_config = {
        'indexing_db_path': 'demo_workflows_indexing.db',
        'enable_deterministic_indexing': True,
        'enable_index_versioning': True,
        'enable_index_consistency': True,
        'enable_index_optimization': True,
        'enable_distributed_indexing': False
    }
    
    doc_manager = EnhancedDocumentManagerWithIdempotentIndexing(
        rag_system,
        enable_idempotent_indexing=True,
        indexing_config=indexing_config
    )
    
    await doc_manager.initialize_idempotent_indexing()
    
    # Create workflow manager
    workflow_manager = IdempotentIndexingWorkflowManager(doc_manager)
    
    # Create indexing workflow
    workflow_id = f"workflow_{int(time.time())}"
    documents = [
        {
            'document_id': f'workflow_doc_{i}',
            'content': f'Workflow document {i} content.',
            'vector': [float(i)/10 + j/100 for j in range(1536)],
            'metadata': {'workflow_id': workflow_id, 'step': i}
        }
        for i in range(5)
    ]
    
    logger.info(f"Starting indexing workflow: {workflow_id}")
    
    # Start indexing workflow
    workflow_started = await workflow_manager.start_indexing_workflow(
        workflow_id, documents, IndexType.DENSE_VECTOR
    )
    
    if workflow_started:
        # Check workflow status
        status = await workflow_manager.get_workflow_status(workflow_id)
        logger.info(f"Workflow status: {status}")
        
        # Wait a moment and check again
        await asyncio.sleep(1)
        status = await workflow_manager.get_workflow_status(workflow_id)
        logger.info(f"Workflow status after processing: {status}")
    
    # Create optimization workflow
    optimization_workflow_id = f"opt_workflow_{int(time.time())}"
    index_types = [IndexType.DENSE_VECTOR, IndexType.SPARSE_VECTOR]
    
    logger.info(f"Starting optimization workflow: {optimization_workflow_id}")
    
    # Start optimization workflow
    opt_workflow_started = await workflow_manager.start_optimization_workflow(
        optimization_workflow_id, index_types
    )
    
    if opt_workflow_started:
        # Check optimization workflow status
        opt_status = await workflow_manager.get_workflow_status(optimization_workflow_id)
        logger.info(f"Optimization workflow status: {opt_status}")
    
    # List all workflows
    logger.info("All workflows:")
    for workflow_id, workflow in workflow_manager.workflows.items():
        logger.info(f"  {workflow_id}: {workflow['status']}")
    
    # Cleanup completed workflows
    cleanup_success = await workflow_manager.cleanup_completed_workflows()
    logger.info(f"Workflow cleanup successful: {cleanup_success}")
    
    # Shutdown
    await doc_manager.shutdown()
    
    return True

async def demonstrate_indexing_security_and_compliance():
    """Demonstrate indexing security and compliance features."""
    logger.info("=== Indexing Security and Compliance Demo ===")
    
    # Initialize idempotent indexing system with security features
    config = {
        'indexing_db_path': 'demo_security_indexing.db',
        'enable_deterministic_indexing': True,
        'enable_index_versioning': True,
        'enable_index_consistency': True,
        'enable_index_optimization': True,
        'enable_distributed_indexing': False
    }
    
    indexing_system = IdempotentIndexingSystem(config)
    
    # Create secure documents with sensitive metadata
    secure_documents = [
        {
            'document_id': 'secure_doc_001',
            'content': 'This document contains sensitive information that requires secure indexing.',
            'vector': [0.1, 0.2, 0.3] * 512,  # 1536 dimensions
            'metadata': {
                'classification': 'CONFIDENTIAL',
                'access_level': 'RESTRICTED',
                'owner': 'Security Team',
                'retention_policy': '7 years',
                'audit_trail': True
            }
        },
        {
            'document_id': 'secure_doc_002',
            'content': 'Another secure document with compliance requirements.',
            'vector': [0.2, 0.3, 0.4] * 512,  # 1536 dimensions
            'metadata': {
                'classification': 'SECRET',
                'access_level': 'TOP_SECRET',
                'owner': 'Intelligence Unit',
                'retention_policy': '10 years',
                'audit_trail': True
            }
        }
    ]
    
    # Index secure documents
    logger.info("Indexing secure documents...")
    secure_index_keys = []
    
    for doc in secure_documents:
        index_key = await indexing_system.index_document(
            document_id=doc['document_id'],
            content=doc['content'],
            vector=doc['vector'],
            metadata=doc['metadata'],
            index_type=IndexType.DENSE_VECTOR,
            consistency_level=IndexConsistencyLevel.LINEARIZABLE  # Highest consistency
        )
        
        if index_key:
            secure_index_keys.append(index_key)
            logger.info(f"Securely indexed {doc['document_id']} with key: {index_key}")
    
    # Validate secure indexes
    logger.info("Validating secure indexes...")
    for index_key in secure_index_keys:
        is_valid = await indexing_system.validate_index(index_key)
        logger.info(f"Secure index {index_key} validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Get comprehensive statistics for compliance reporting
    stats = await indexing_system.get_index_statistics()
    logger.info("Compliance statistics:")
    logger.info(f"  Total entries: {stats.get('total_entries', 0)}")
    logger.info(f"  Entries by type: {stats.get('entries_by_type', {})}")
    logger.info(f"  Deterministic indexing enabled: {stats.get('deterministic_indexing_enabled', False)}")
    logger.info(f"  Index versioning enabled: {stats.get('index_versioning_enabled', False)}")
    logger.info(f"  Index consistency enabled: {stats.get('index_consistency_enabled', False)}")
    
    # Get index health for security monitoring
    health = await indexing_system.get_index_health(IndexType.DENSE_VECTOR)
    if health:
        logger.info(f"Index health score: {health.get('health_score', 0):.2f}")
        logger.info(f"Index status: {health.get('status', 'UNKNOWN')}")
        logger.info(f"Issues: {health.get('issues', [])}")
    
    # Shutdown
    await indexing_system.shutdown()
    
    return secure_index_keys

async def demonstrate_production_indexing_patterns():
    """Demonstrate production-ready indexing patterns."""
    logger.info("=== Production Indexing Patterns Demo ===")
    
    # Pattern 1: Incremental Indexing
    logger.info("Pattern 1: Incremental Indexing")
    
    config = {
        'indexing_db_path': 'demo_incremental_indexing.db',
        'enable_deterministic_indexing': True,
        'enable_index_versioning': True,
        'enable_index_consistency': True,
        'enable_index_optimization': True,
        'enable_distributed_indexing': False
    }
    
    indexing_system = IdempotentIndexingSystem(config)
    
    # Simulate incremental indexing
    batch_1 = [
        {'document_id': f'inc_doc_{i}', 'content': f'Incremental document {i}', 
         'vector': [float(i)/10] * 1536, 'metadata': {'batch': '1'}}
        for i in range(5)
    ]
    
    batch_2 = [
        {'document_id': f'inc_doc_{i+5}', 'content': f'Incremental document {i+5}', 
         'vector': [float(i+5)/10] * 1536, 'metadata': {'batch': '2'}}
        for i in range(5)
    ]
    
    # Index first batch
    logger.info("Indexing first batch...")
    for doc in batch_1:
        index_key = await indexing_system.index_document(
            doc['document_id'], doc['content'], doc['vector'], doc['metadata']
        )
        logger.info(f"Indexed: {doc['document_id']}")
    
    # Index second batch (incremental)
    logger.info("Indexing second batch (incremental)...")
    for doc in batch_2:
        index_key = await indexing_system.index_document(
            doc['document_id'], doc['content'], doc['vector'], doc['metadata']
        )
        logger.info(f"Indexed: {doc['document_id']}")
    
    # Pattern 2: Index Versioning and Rollback
    logger.info("Pattern 2: Index Versioning and Rollback")
    
    # Create a document with multiple versions
    test_doc = {
        'document_id': 'version_test_doc',
        'content': 'Original content',
        'vector': [0.1] * 1536,
        'metadata': {'version': 'v1.0'}
    }
    
    # Index original version
    original_key = await indexing_system.index_document(
        test_doc['document_id'], test_doc['content'], test_doc['vector'], test_doc['metadata']
    )
    
    # Update content (new version)
    test_doc['content'] = 'Updated content with improvements'
    test_doc['metadata']['version'] = 'v2.0'
    test_doc['vector'] = [0.2] * 1536
    
    updated_key = await indexing_system.index_document(
        test_doc['document_id'], test_doc['content'], test_doc['vector'], test_doc['metadata']
    )
    
    logger.info(f"Original version key: {original_key}")
    logger.info(f"Updated version key: {updated_key}")
    
    # Pattern 3: Index Optimization and Maintenance
    logger.info("Pattern 3: Index Optimization and Maintenance")
    
    # Get index statistics before optimization
    stats_before = await indexing_system.get_index_statistics()
    logger.info(f"Statistics before optimization: {stats_before}")
    
    # Perform optimization
    optimization_success = await indexing_system.optimize_index(IndexType.DENSE_VECTOR)
    logger.info(f"Optimization successful: {optimization_success}")
    
    # Get index statistics after optimization
    stats_after = await indexing_system.get_index_statistics()
    logger.info(f"Statistics after optimization: {stats_after}")
    
    # Pattern 4: Index Monitoring and Health Checks
    logger.info("Pattern 4: Index Monitoring and Health Checks")
    
    health = await indexing_system.get_index_health(IndexType.DENSE_VECTOR)
    if health:
        logger.info(f"Index health: {health}")
        
        # Simulate health monitoring over time
        for i in range(3):
            await asyncio.sleep(1)  # Simulate time passing
            current_health = await indexing_system.get_index_health(IndexType.DENSE_VECTOR)
            logger.info(f"Health check {i+1}: {current_health}")
    
    # Shutdown
    await indexing_system.shutdown()
    
    return True

async def main():
    """Main demonstration function."""
    logger.info("Starting Idempotent Indexing System Demonstration")
    
    try:
        # Run all demonstrations
        demonstrations = [
            demonstrate_basic_idempotent_indexing,
            demonstrate_enhanced_document_manager_with_indexing,
            demonstrate_batch_indexing_operations,
            demonstrate_indexing_workflows,
            demonstrate_indexing_security_and_compliance,
            demonstrate_production_indexing_patterns
        ]
        
        results = {}
        
        for demo in demonstrations:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running: {demo.__name__}")
                logger.info(f"{'='*60}")
                
                result = await demo()
                results[demo.__name__] = {'status': 'success', 'result': result}
                
                logger.info(f"✓ {demo.__name__} completed successfully")
                
            except Exception as e:
                logger.error(f"✗ {demo.__name__} failed: {e}")
                results[demo.__name__] = {'status': 'failed', 'error': str(e)}
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("DEMONSTRATION SUMMARY")
        logger.info(f"{'='*60}")
        
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        total = len(results)
        
        logger.info(f"Successful demonstrations: {successful}/{total}")
        
        for demo_name, result in results.items():
            status = "✓" if result['status'] == 'success' else "✗"
            logger.info(f"  {status} {demo_name}: {result['status']}")
        
        logger.info(f"\nIdempotent Indexing System demonstration completed!")
        
    except Exception as e:
        logger.error(f"Demostration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())