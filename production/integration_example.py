"""
Integration Example: Complete Production RAG System

This example demonstrates how to integrate all the production features
into a complete, production-ready RAG system.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

from config import MARKDOWN_DIR, PARENT_STORE_PATH, QDRANT_DB_PATH
from enhanced_document_manager import EnhancedDocumentManager
from delta_processor import DeltaProcessor, create_delta_processor
from monitoring_system import setup_monitoring_system
from rag_system import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionRAGSystem:
    """Complete production-ready RAG system with all enterprise features."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize production RAG system.
        
        Args:
            config: Configuration dictionary with system settings
        """
        self.config = config
        self.rag_system = None
        self.doc_manager = None
        self.delta_processor = None
        self.monitoring_system = None
        
        # System state
        self.is_initialized = False
        self.is_monitoring = False
        
    async def initialize(self):
        """Initialize the complete production system."""
        try:
            logger.info("Initializing Production RAG System...")
            
            # Initialize core RAG system
            self.rag_system = RAGSystem()
            await self.rag_system.initialize()
            
            # Initialize enhanced document manager
            self.doc_manager = EnhancedDocumentManager(
                self.rag_system,
                enable_deduplication=self.config.get('enable_deduplication', True),
                enable_versioning=self.config.get('enable_versioning', True)
            )
            
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
            logger.info("Production RAG System initialized successfully")
            
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
                             recursive: bool = True) -> Dict[str, Any]:
        """Ingest documents with full production features."""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        try:
            # Use enhanced document manager for ingestion
            results = self.doc_manager.add_directory(
                paths[0] if len(paths) == 1 else paths,
                recursive=recursive
            )
            
            logger.info(f"Ingestion completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to ingest documents: {e}")
            return {'error': str(e)}
    
    async def ingest_web_pages(self, urls: List[str]) -> Dict[str, Any]:
        """Ingest web pages with full production features."""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        results = []
        for url in urls:
            try:
                success, message, metadata = self.doc_manager.add_web_page(url)
                results.append({
                    'url': url,
                    'success': success,
                    'message': message,
                    'document_id': metadata.document_id if metadata else None
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
        """Get comprehensive system status."""
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        try:
            # Get document statistics
            doc_stats = self.doc_manager.get_statistics()
            
            # Get processing status
            processing_status = self.delta_processor.get_queue_status()
            
            # Get monitoring status
            system_health = self.monitoring_system['metrics_collector'].get_system_health()
            processing_health = self.monitoring_system['metrics_collector'].get_processing_health()
            alert_summary = self.monitoring_system['alert_manager'].get_alert_summary()
            
            return {
                'status': 'running',
                'document_stats': doc_stats,
                'processing_status': processing_status,
                'system_health': system_health,
                'processing_health': processing_health,
                'alerts': alert_summary,
                'monitoring_active': self.is_monitoring
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive system report."""
        try:
            return self.monitoring_system['dashboard'].generate_report(hours)
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Shutdown the production system gracefully."""
        try:
            # Stop monitoring
            await self.stop_monitoring()
            
            # Stop monitoring system
            if self.monitoring_system:
                self.monitoring_system['metrics_collector'].stop_collection()
                self.monitoring_system['dashboard'].stop_dashboard()
            
            # Clear all data if configured
            if self.config.get('clear_on_shutdown', False):
                self.doc_manager.clear_all()
            
            self.is_initialized = False
            logger.info("Production RAG System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def main():
    """Example usage of the production RAG system."""
    
    # Configuration
    config = {
        'enable_deduplication': True,
        'enable_versioning': True,
        'monitoring_interval': 60,  # seconds
        'prometheus_port': 8000,
        'delta_processor': {
            'batch_size': 10,
            'max_workers': 4
        },
        'clear_on_shutdown': False
    }
    
    # Initialize system
    system = ProductionRAGSystem(config)
    await system.initialize()
    
    # Example 1: Ingest documents from directories
    document_dirs = [
        Path("documents/"),
        Path("reports/"),
        Path("manuals/")
    ]
    
    for directory in document_dirs:
        if directory.exists():
            results = await system.ingest_documents([directory], recursive=True)
            print(f"Ingested {directory}: {results}")
    
    # Example 2: Ingest web pages
    web_pages = [
        "https://docs.python.org/3/",
        "https://langchain.com/",
        "https://qdrant.tech/"
    ]
    
    web_results = await system.ingest_web_pages(web_pages)
    print(f"Web page ingestion results: {web_results}")
    
    # Example 3: Start monitoring
    await system.start_monitoring(document_dirs)
    
    # Example 4: Get system status
    status = system.get_system_status()
    print(f"System status: {status}")
    
    # Example 5: Generate report after some time
    await asyncio.sleep(120)  # Wait 2 minutes
    
    report = system.generate_report(hours=1)
    print(f"System report: {report}")
    
    # Example 6: Export metrics
    system.monitoring_system['dashboard'].export_metrics_to_file('production_metrics.json')
    
    # Shutdown
    await system.shutdown()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())