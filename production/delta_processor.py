"""
Delta Processing Module for Production RAG System

Provides smart incremental ingestion, delta processing, and background task management
for efficient document updates and changes detection.
"""

import asyncio
import hashlib
import json
import logging
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiofiles
import aiofiles.os
from queue import Queue, Empty
import pickle

from config import MARKDOWN_DIR, PARENT_STORE_PATH, QDRANT_DB_PATH
from enhanced_document_manager import EnhancedDocumentManager, DocumentMetadata
from multi_format_ingestion import DocumentTypeDetector

logger = logging.getLogger(__name__)

@dataclass
class DeltaTask:
    """Represents a delta processing task."""
    task_id: str
    file_path: Path
    task_type: str  # 'new', 'update', 'delete', 'check'
    priority: int = 1
    metadata: Optional[Dict] = None
    created_at: datetime = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ProcessingMetrics:
    """Metrics for delta processing performance."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    processing_time: float = 0.0
    avg_processing_time: float = 0.0
    throughput: float = 0.0  # tasks per second
    
    def update_completion(self, processing_time: float):
        """Update metrics for completed task."""
        self.completed_tasks += 1
        self.processing_time += processing_time
        self.avg_processing_time = self.processing_time / self.completed_tasks
        self.throughput = self.completed_tasks / max(self.processing_time, 0.001)

class SmartChangeDetector:
    """Advanced change detection with multiple levels of analysis."""
    
    def __init__(self, sensitivity: str = 'medium'):
        """
        Initialize change detector.
        
        Args:
            sensitivity: 'low', 'medium', 'high' - detection sensitivity
        """
        self.sensitivity = sensitivity
        self.similarity_threshold = self._get_threshold(sensitivity)
        self.file_cache = {}  # Cache for file metadata
        
    def _get_threshold(self, sensitivity: str) -> float:
        """Get similarity threshold based on sensitivity."""
        thresholds = {
            'low': 0.85,    # Only major changes
            'medium': 0.95, # Moderate changes
            'high': 0.99    # Minor changes
        }
        return thresholds.get(sensitivity, 0.95)
    
    async def detect_changes(self, file_path: Path, 
                           existing_metadata: Optional[DocumentMetadata] = None) -> Tuple[str, Dict]:
        """
        Detect changes in a document with multiple levels of analysis.
        
        Returns:
            Tuple of (change_type, detection_details)
            change_type: 'new', 'unchanged', 'minor_change', 'major_change'
        """
        try:
            # Level 1: Quick metadata check
            metadata_change = await self._check_metadata_change(file_path, existing_metadata)
            if metadata_change == 'unchanged':
                return 'unchanged', {'level': 1, 'reason': 'metadata_unchanged'}
            
            # Level 2: Hash comparison
            hash_change = await self._check_hash_change(file_path, existing_metadata)
            if hash_change == 'unchanged':
                return 'unchanged', {'level': 2, 'reason': 'hash_unchanged'}
            
            # Level 3: Content similarity (for major documents)
            if existing_metadata and self._should_check_similarity(file_path, existing_metadata):
                similarity_change = await self._check_content_similarity(file_path, existing_metadata)
                if similarity_change == 'minor_change':
                    return 'minor_change', {'level': 3, 'reason': 'minor_content_change'}
            
            return 'major_change', {'level': 2, 'reason': 'hash_changed'}
            
        except Exception as e:
            logger.error(f"Error detecting changes for {file_path}: {e}")
            return 'major_change', {'level': 0, 'reason': 'error', 'error': str(e)}
    
    async def _check_metadata_change(self, file_path: Path, 
                                   existing_metadata: Optional[DocumentMetadata]) -> str:
        """Quick metadata comparison."""
        if not existing_metadata:
            return 'new'
        
        try:
            stat = await aiofiles.os.stat(file_path)
            current_size = stat.st_size
            current_mtime = datetime.fromtimestamp(stat.st_mtime)
            
            # Check size change
            if current_size != existing_metadata.file_size:
                return 'changed'
            
            # Check modification time (with tolerance)
            existing_mtime = datetime.fromisoformat(existing_metadata.modification_date)
            time_diff = abs((current_mtime - existing_mtime).total_seconds())
            
            # If file is older than existing metadata, it might be unchanged
            if time_diff < 1:  # 1 second tolerance
                return 'unchanged'
            
            return 'changed'
            
        except Exception as e:
            logger.warning(f"Metadata check failed for {file_path}: {e}")
            return 'changed'
    
    async def _check_hash_change(self, file_path: Path, 
                               existing_metadata: Optional[DocumentMetadata]) -> str:
        """Hash-based change detection."""
        if not existing_metadata:
            return 'new'
        
        try:
            current_hash = await self._calculate_file_hash(file_path)
            if current_hash == existing_metadata.file_hash:
                return 'unchanged'
            return 'changed'
            
        except Exception as e:
            logger.warning(f"Hash check failed for {file_path}: {e}")
            return 'changed'
    
    async def _check_content_similarity(self, file_path: Path, 
                                      existing_metadata: DocumentMetadata) -> str:
        """Content similarity analysis for fine-grained change detection."""
        try:
            # For now, we'll use a simple approach
            # In production, this could use semantic similarity models
            
            # Get file sizes for quick comparison
            current_stat = await aiofiles.os.stat(file_path)
            current_size = current_stat.st_size
            existing_size = existing_metadata.file_size
            
            size_diff = abs(current_size - existing_size) / max(existing_size, 1)
            
            if size_diff < 0.01:  # Less than 1% size difference
                return 'minor_change'
            elif size_diff < 0.1:  # Less than 10% size difference
                return 'minor_change'
            else:
                return 'major_change'
                
        except Exception as e:
            logger.warning(f"Content similarity check failed for {file_path}: {e}")
            return 'major_change'
    
    def _should_check_similarity(self, file_path: Path, existing_metadata: DocumentMetadata) -> bool:
        """Determine if content similarity check is needed."""
        # Skip for very small files or images
        if existing_metadata.file_size < 1024:  # Less than 1KB
            return False
        
        # Skip for image files (OCR results may vary)
        if existing_metadata.document_type == 'image':
            return False
        
        return True
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        hash_sha256 = hashlib.sha256()
        try:
            async with aiofiles.open(file_path, "rb") as f:
                while chunk := await f.read(4096):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""

class BackgroundTaskQueue:
    """Background task queue with priority management and retry logic."""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 10):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.task_queue = Queue()
        self.priority_queue = Queue()
        self.completed_tasks = set()
        self.failed_tasks = set()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = False
        self.worker_threads = []
        
        # Metrics
        self.metrics = ProcessingMetrics()
        self.metrics_lock = threading.Lock()
        
        # Event for graceful shutdown
        self.shutdown_event = threading.Event()
    
    async def start(self):
        """Start background processing."""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info(f"Starting background task queue with {self.max_workers} workers")
        
        # Start worker threads
        for i in range(self.max_workers):
            thread = threading.Thread(target=self._worker_loop, args=(i,))
            thread.daemon = True
            thread.start()
            self.worker_threads.append(thread)
    
    async def stop(self):
        """Stop background processing gracefully."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Wait for all workers to finish
        for thread in self.worker_threads:
            thread.join(timeout=30)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Background task queue stopped")
    
    def add_task(self, task: DeltaTask):
        """Add a task to the queue."""
        if task.priority > 1:
            self.priority_queue.put(task)
        else:
            self.task_queue.put(task)
        
        logger.debug(f"Added task {task.task_id} to queue (priority: {task.priority})")
    
    def add_tasks(self, tasks: List[DeltaTask]):
        """Add multiple tasks to the queue."""
        for task in tasks:
            self.add_task(task)
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop."""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Try priority queue first, then regular queue
                try:
                    task = self.priority_queue.get_nowait()
                except Empty:
                    try:
                        task = self.task_queue.get_nowait()
                    except Empty:
                        # No tasks available, sleep briefly
                        time.sleep(0.1)
                        continue
                
                # Process the task
                self._process_task(task, worker_id)
                
                # Mark task as completed
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                time.sleep(1)  # Brief delay on error
        
        logger.info(f"Worker {worker_id} stopped")
    
    def _process_task(self, task: DeltaTask, worker_id: int):
        """Process a single task."""
        start_time = time.time()
        
        try:
            logger.info(f"Worker {worker_id} processing task {task.task_id}")
            
            # Execute the task
            success = self._execute_task(task)
            
            if success:
                with self.metrics_lock:
                    self.metrics.update_completion(time.time() - start_time)
                    self.completed_tasks.add(task.task_id)
                logger.info(f"Task {task.task_id} completed successfully")
            else:
                self._handle_task_failure(task)
                
        except Exception as e:
            logger.error(f"Task {task.task_id} failed with exception: {e}")
            self._handle_task_failure(task)
    
    def _execute_task(self, task: DeltaTask) -> bool:
        """Execute a specific task."""
        # This will be implemented by the DeltaProcessor
        # For now, return True to simulate success
        return True
    
    def _handle_task_failure(self, task: DeltaTask):
        """Handle task failure with retry logic."""
        task.retry_count += 1
        
        if task.retry_count < task.max_retries:
            # Re-queue with higher priority
            task.priority = min(task.priority + 1, 5)
            self.add_task(task)
            logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
        else:
            # Max retries exceeded
            self.failed_tasks.add(task.task_id)
            logger.error(f"Task {task.task_id} failed after {task.max_retries} retries")
    
    def get_status(self) -> Dict:
        """Get queue status and metrics."""
        with self.metrics_lock:
            return {
                'queue_size': self.task_queue.qsize(),
                'priority_queue_size': self.priority_queue.qsize(),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'metrics': asdict(self.metrics)
            }

class DeltaProcessor:
    """Main delta processing orchestrator."""
    
    def __init__(self, rag_system, enhanced_doc_manager: EnhancedDocumentManager):
        self.rag_system = rag_system
        self.doc_manager = enhanced_doc_manager
        self.change_detector = SmartChangeDetector(sensitivity='medium')
        self.task_queue = BackgroundTaskQueue(max_workers=4)
        
        # Configuration
        self.batch_size = 10
        self.check_interval = 60  # seconds
        self.enable_monitoring = True
        
        # State
        self.is_monitoring = False
        self.monitoring_thread = None
        self.last_scan_time = None
        
        # Monitoring callbacks
        self.progress_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
    
    async def start_monitoring(self, directories: List[Path], 
                             interval: Optional[int] = None):
        """Start monitoring directories for changes."""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        if interval:
            self.check_interval = interval
        
        # Start background queue
        await self.task_queue.start()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(directories,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Started monitoring {len(directories)} directories with {self.check_interval}s interval")
    
    async def stop_monitoring(self):
        """Stop monitoring directories."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        # Stop background queue
        await self.task_queue.stop()
        
        # Wait for monitoring thread
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=30)
        
        logger.info("Stopped monitoring")
    
    def _monitoring_loop(self, directories: List[Path]):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # Scan directories for changes
                tasks = self._scan_directories(directories)
                
                # Add tasks to queue
                if tasks:
                    self.task_queue.add_tasks(tasks)
                    logger.info(f"Found {len(tasks)} changes to process")
                
                # Report status
                if self.status_callback:
                    status = self.task_queue.get_status()
                    self.status_callback(status)
                
                # Sleep until next check
                elapsed = time.time() - start_time
                sleep_time = max(0, self.check_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief delay on error
    
    def _scan_directories(self, directories: List[Path]) -> List[DeltaTask]:
        """Scan directories for document changes."""
        tasks = []
        
        for directory in directories:
            if not directory.exists():
                continue
            
            # Get all supported files
            supported_files = [
                f for f in directory.rglob('*') 
                if f.is_file() and DocumentTypeDetector.is_supported(f)
            ]
            
            # Check each file for changes
            for file_path in supported_files:
                task = self._create_delta_task(file_path)
                if task:
                    tasks.append(task)
        
        return tasks
    
    def _create_delta_task(self, file_path: Path) -> Optional[DeltaTask]:
        """Create a delta task for a file."""
        try:
            # Get existing metadata if available
            existing_metadata = self._get_existing_metadata(file_path)
            
            # Detect changes
            change_type, details = asyncio.run(
                self.change_detector.detect_changes(file_path, existing_metadata)
            )
            
            if change_type == 'unchanged':
                return None
            
            # Create task
            task_id = f"{file_path}_{change_type}_{int(time.time())}"
            task = DeltaTask(
                task_id=task_id,
                file_path=file_path,
                task_type=change_type,
                priority=2 if change_type == 'major_change' else 1,
                metadata={'change_details': details, 'existing_metadata': existing_metadata}
            )
            
            return task
            
        except Exception as e:
            logger.error(f"Error creating delta task for {file_path}: {e}")
            return None
    
    def _get_existing_metadata(self, file_path: Path) -> Optional[DocumentMetadata]:
        """Get existing metadata for a file."""
        try:
            # Calculate hash to find existing document
            file_hash = self._calculate_file_hash_sync(file_path)
            
            # Check hash index
            hash_index = self.doc_manager.hash_index
            if file_hash in hash_index:
                doc_id = hash_index[file_hash]
                doc_info = self.doc_manager.get_document_info(doc_id)
                if doc_info:
                    return DocumentMetadata(**doc_info)
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not get existing metadata for {file_path}: {e}")
            return None
    
    def _calculate_file_hash_sync(self, file_path: Path) -> str:
        """Calculate file hash synchronously."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    async def process_directory_delta(self, directory: Path, 
                                    recursive: bool = True) -> Dict[str, Any]:
        """Process delta changes in a directory."""
        try:
            # Get all supported files
            if recursive:
                files = [f for f in directory.rglob('*') if f.is_file() and DocumentTypeDetector.is_supported(f)]
            else:
                files = [f for f in directory.iterdir() if f.is_file() and DocumentTypeDetector.is_supported(f)]
            
            # Create delta tasks
            tasks = []
            for file_path in files:
                task = self._create_delta_task(file_path)
                if task:
                    tasks.append(task)
            
            # Process tasks
            results = {
                'total_files': len(files),
                'tasks_created': len(tasks),
                'processed': 0,
                'added': 0,
                'updated': 0,
                'skipped': 0,
                'failed': 0,
                'details': []
            }
            
            # Process in batches
            for i in range(0, len(tasks), self.batch_size):
                batch = tasks[i:i + self.batch_size]
                batch_results = await self._process_batch(batch)
                
                # Update results
                results['processed'] += batch_results['processed']
                results['added'] += batch_results['added']
                results['updated'] += batch_results['updated']
                results['skipped'] += batch_results['skipped']
                results['failed'] += batch_results['failed']
                results['details'].extend(batch_results['details'])
                
                # Progress callback
                if self.progress_callback:
                    progress = (i + len(batch)) / len(tasks)
                    self.progress_callback(progress, f"Processed batch {i//self.batch_size + 1}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing directory delta {directory}: {e}")
            return {'error': str(e)}
    
    async def _process_batch(self, tasks: List[DeltaTask]) -> Dict[str, Any]:
        """Process a batch of delta tasks."""
        results = {
            'processed': 0,
            'added': 0,
            'updated': 0,
            'skipped': 0,
            'failed': 0,
            'details': []
        }
        
        for task in tasks:
            try:
                result = await self._process_delta_task(task)
                results['processed'] += 1
                results['details'].append(result)
                
                if result['success']:
                    if task.task_type == 'new':
                        results['added'] += 1
                    elif task.task_type in ['minor_change', 'major_change']:
                        results['updated'] += 1
                else:
                    results['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing task {task.task_id}: {e}")
                results['failed'] += 1
                results['details'].append({
                    'task_id': task.task_id,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def _process_delta_task(self, task: DeltaTask) -> Dict[str, Any]:
        """Process a single delta task."""
        try:
            if task.task_type == 'new':
                success, message, metadata = self.doc_manager.add_document(
                    task.file_path, 
                    metadata=task.metadata
                )
                return {
                    'task_id': task.task_id,
                    'file': str(task.file_path),
                    'success': success,
                    'message': message,
                    'document_id': metadata.document_id if metadata else None
                }
            
            elif task.task_type in ['minor_change', 'major_change']:
                # Force reprocess the document
                success, message, metadata = self.doc_manager.add_document(
                    task.file_path, 
                    metadata=task.metadata,
                    force_reprocess=True
                )
                return {
                    'task_id': task.task_id,
                    'file': str(task.file_path),
                    'success': success,
                    'message': message,
                    'document_id': metadata.document_id if metadata else None,
                    'version': metadata.version if metadata else None
                }
            
            else:
                return {
                    'task_id': task.task_id,
                    'file': str(task.file_path),
                    'success': False,
                    'message': f'Unknown task type: {task.task_type}'
                }
                
        except Exception as e:
            return {
                'task_id': task.task_id,
                'file': str(task.file_path),
                'success': False,
                'error': str(e)
            }
    
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set progress callback function."""
        self.progress_callback = callback
    
    def set_status_callback(self, callback: Callable[[Dict], None]):
        """Set status callback function."""
        self.status_callback = callback
    
    def get_queue_status(self) -> Dict:
        """Get current queue status."""
        return self.task_queue.get_status()
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics."""
        return self.task_queue.metrics


# Convenience functions for easy use
def create_delta_processor(rag_system, enhanced_doc_manager: EnhancedDocumentManager, **kwargs):
    """Create a delta processor instance."""
    return DeltaProcessor(rag_system, enhanced_doc_manager, **kwargs)


if __name__ == "__main__":
    
    # rag_system = RAGSystem()  # Assume RAGSystem is imported
    # doc_manager = EnhancedDocumentManager(rag_system)
    # delta_processor = DeltaProcessor(rag_system, doc_manager)
    
    # Start monitoring
    # await delta_processor.start_monitoring([Path("documents/")])
    
    # Process directory delta
    # results = await delta_processor.process_directory_delta(Path("documents/"))
    # print(f"Processed: {results['processed']}, Added: {results['added']}")
    pass