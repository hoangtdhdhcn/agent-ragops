"""
Latency Measurement Module for RAG System

This module provides comprehensive latency measurement capabilities for the RAG system,
tracking performance across different stages of the retrieval-augmented generation pipeline.

The latency measurement covers:
1. Document ingestion and chunking
2. Vector database operations (insertion, retrieval)
3. Parent store operations (metadata management)
4. Agent graph execution (query processing)
5. End-to-end response generation
6. Evaluation pipeline performance

Usage:
    from eval.latency import LatencyTracker
    tracker = LatencyTracker()
    
    # Measure a complete RAG query
    with tracker.measure_query("user question"):
        response = rag_system.query("user question")
    
    # Get detailed latency breakdown
    stats = tracker.get_latency_stats()
"""

import time
import json
import logging
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from collections import defaultdict
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class LatencyMetric:
    """Represents a single latency measurement."""
    operation: str
    duration: float
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation': self.operation,
            'duration': self.duration,
            'timestamp': self.timestamp,
            'metadata': self.metadata or {}
        }

@dataclass
class LatencyStats:
    """Aggregated statistics for an operation."""
    operation: str
    count: int
    total_time: float
    min_time: float
    max_time: float
    avg_time: float
    median_time: float
    p95_time: float
    p99_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

class LatencyTracker:
    """
    Comprehensive latency tracking for RAG system operations.
    
    This class provides fine-grained timing measurements across all components
    of the RAG pipeline, enabling performance analysis and optimization.
    """
    
    def __init__(self, enable_detailed_logging: bool = True):
        """
        Initialize the latency tracker.
        
        Args:
            enable_detailed_logging: Whether to log individual measurements
        """
        self.enable_detailed_logging = enable_detailed_logging
        self._metrics: List[LatencyMetric] = []
        self._operation_times: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Define RAG pipeline stages
        self.pipeline_stages = [
            'document_ingestion',
            'document_chunking', 
            'vector_insertion',
            'vector_retrieval',
            'parent_store_operations',
            'agent_graph_execution',
            'llm_generation',
            'evaluation_pipeline',
            'end_to_end_query'
        ]
        
        logger.info("LatencyTracker initialized")
    
    @contextmanager
    def measure_operation(self, operation_name: str, metadata: Dict[str, Any] = None):
        """
        Context manager for measuring operation latency.
        
        Args:
            operation_name: Name of the operation to measure
            metadata: Additional metadata to store with the measurement
            
        Yields:
            None
        """
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            metric = LatencyMetric(
                operation=operation_name,
                duration=duration,
                timestamp=time.time(),
                metadata=metadata
            )
            
            self._record_metric(metric)
    
    @contextmanager  
    def measure_query(self, query: str, expected_retrieval_count: int = 3):
        """
        Context manager for measuring complete RAG query latency.
        
        Args:
            query: The user query being processed
            expected_retrieval_count: Expected number of documents to retrieve
            
        Yields:
            None
        """
        metadata = {
            'query': query,
            'expected_retrieval_count': expected_retrieval_count,
            'query_length': len(query)
        }
        
        with self.measure_operation('end_to_end_query', metadata):
            yield
    
    def measure_function(self, operation_name: str) -> Callable:
        """
        Decorator for measuring function latency.
        
        Args:
            operation_name: Name of the operation to measure
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                metadata = {
                    'function_name': func.__name__,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }
                
                with self.measure_operation(operation_name, metadata):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def _record_metric(self, metric: LatencyMetric):
        """Record a latency metric with thread safety."""
        with self._lock:
            self._metrics.append(metric)
            self._operation_times[metric.operation].append(metric.duration)
            
            if self.enable_detailed_logging:
                logger.info(
                    f"Latency: {metric.operation} = {metric.duration:.4f}s "
                    f"(metadata: {metric.metadata})"
                )
    
    def get_latency_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get aggregated latency statistics.
        
        Args:
            operation: Specific operation to get stats for. If None, returns all operations.
            
        Returns:
            Dictionary containing latency statistics
        """
        with self._lock:
            if operation:
                if operation not in self._operation_times:
                    return {}
                
                times = self._operation_times[operation]
                if not times:
                    return {}
                
                stats = self._calculate_stats(operation, times)
                return stats.to_dict()
            else:
                all_stats = {}
                for op, times in self._operation_times.items():
                    if times:
                        stats = self._calculate_stats(op, times)
                        all_stats[op] = stats.to_dict()
                return all_stats
    
    def _calculate_stats(self, operation: str, times: List[float]) -> LatencyStats:
        """Calculate statistical metrics for a list of timing measurements."""
        count = len(times)
        total_time = sum(times)
        min_time = min(times)
        max_time = max(times)
        avg_time = total_time / count
        median_time = statistics.median(times)
        
        # Calculate percentiles
        sorted_times = sorted(times)
        p95_index = int(0.95 * (count - 1))
        p99_index = int(0.99 * (count - 1))
        
        p95_time = sorted_times[p95_index] if p95_index < count else max_time
        p99_time = sorted_times[p99_index] if p99_index < count else max_time
        
        return LatencyStats(
            operation=operation,
            count=count,
            total_time=total_time,
            min_time=min_time,
            max_time=max_time,
            avg_time=avg_time,
            median_time=median_time,
            p95_time=p95_time,
            p99_time=p99_time
        )
    
    def get_pipeline_breakdown(self) -> Dict[str, Any]:
        """
        Get latency breakdown by RAG pipeline stages.
        
        Returns:
            Dictionary with pipeline stage statistics
        """
        stats = self.get_latency_stats()
        pipeline_stats = {}
        
        for stage in self.pipeline_stages:
            if stage in stats:
                pipeline_stats[stage] = stats[stage]
        
        return pipeline_stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a high-level performance summary.
        
        Returns:
            Dictionary with key performance indicators
        """
        stats = self.get_latency_stats()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_measurements': len(self._metrics),
            'operations_tracked': list(stats.keys()),
            'pipeline_performance': self.get_pipeline_breakdown(),
            'bottlenecks': self._identify_bottlenecks(stats),
            'recommendations': self._generate_recommendations(stats)
        }
        
        return summary
    
    def _identify_bottlenecks(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential performance bottlenecks."""
        bottlenecks = []
        
        for operation, data in stats.items():
            if data.get('avg_time', 0) > 1.0:  # Operations taking more than 1 second
                bottlenecks.append({
                    'operation': operation,
                    'avg_time': data['avg_time'],
                    'severity': 'HIGH' if data['avg_time'] > 5.0 else 'MEDIUM'
                })
            elif data.get('p95_time', 0) > 2.0:  # 95th percentile over 2 seconds
                bottlenecks.append({
                    'operation': operation,
                    'p95_time': data['p95_time'],
                    'severity': 'MEDIUM'
                })
        
        # Sort by average time
        bottlenecks.sort(key=lambda x: x.get('avg_time', x.get('p95_time', 0)), reverse=True)
        return bottlenecks
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check for slow vector operations
        if 'vector_retrieval' in stats:
            retrieval_time = stats['vector_retrieval']['avg_time']
            if retrieval_time > 0.5:
                recommendations.append(
                    "Vector retrieval is slow. Consider optimizing index parameters, "
                    "increasing hardware resources, or implementing caching."
                )
        
        # Check for slow LLM generation
        if 'llm_generation' in stats:
            generation_time = stats['llm_generation']['avg_time']
            if generation_time > 2.0:
                recommendations.append(
                    "LLM generation is slow. Consider using a faster model, "
                    "optimizing prompt length, or implementing streaming responses."
                )
        
        # Check for slow document processing
        if 'document_chunking' in stats:
            chunking_time = stats['document_chunking']['avg_time']
            if chunking_time > 1.0:
                recommendations.append(
                    "Document chunking is slow. Consider optimizing chunk size, "
                    "parallelizing processing, or using more efficient chunking algorithms."
                )
        
        # Check for high variance in response times
        for operation, data in stats.items():
            if data['count'] > 10:  # Only for operations with sufficient samples
                cv = data['max_time'] / data['avg_time'] if data['avg_time'] > 0 else 0
                if cv > 3.0:
                    recommendations.append(
                        f"High variance in {operation} response times. "
                        "Investigate inconsistent performance patterns."
                    )
        
        return recommendations
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """
        Export latency metrics to a file.
        
        Args:
            filepath: Path to export file
            format: Export format ('json' or 'csv')
        """
        # Get all data without locking to avoid deadlock
        metrics_copy = self._metrics.copy()
        operation_times_copy = {k: v.copy() for k, v in self._operation_times.items()}
        
        if format.lower() == 'json':
            self._export_json_simple(filepath, metrics_copy, operation_times_copy)
        elif format.lower() == 'csv':
            self._export_csv_simple(filepath, metrics_copy)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json_simple(self, filepath: str, metrics_copy: List[LatencyMetric], operation_times_copy: Dict[str, List[float]]):
        """Export metrics to JSON format without locking."""
        # Calculate stats from the copied data
        stats = {}
        for op, times in operation_times_copy.items():
            if times:
                count = len(times)
                total_time = sum(times)
                min_time = min(times)
                max_time = max(times)
                avg_time = total_time / count
                median_time = statistics.median(times)
                
                # Calculate percentiles
                sorted_times = sorted(times)
                p95_index = int(0.95 * (count - 1))
                p99_index = int(0.99 * (count - 1))
                
                p95_time = sorted_times[p95_index] if p95_index < count else max_time
                p99_time = sorted_times[p99_index] if p99_index < count else max_time
                
                stats[op] = {
                    'operation': op,
                    'count': count,
                    'total_time': total_time,
                    'min_time': min_time,
                    'max_time': max_time,
                    'avg_time': avg_time,
                    'median_time': median_time,
                    'p95_time': p95_time,
                    'p99_time': p99_time
                }
        
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_measurements': len(metrics_copy),
                'operations_tracked': list(stats.keys())
            },
            'raw_metrics': [metric.to_dict() for metric in metrics_copy],
            'statistics': stats
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def _export_csv_simple(self, filepath: str, metrics_copy: List[LatencyMetric]):
        """Export metrics to CSV format without locking."""
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['operation', 'duration', 'timestamp', 'metadata']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for metric in metrics_copy:
                writer.writerow({
                    'operation': metric.operation,
                    'duration': metric.duration,
                    'timestamp': metric.timestamp,
                    'metadata': json.dumps(metric.metadata)
                })
        
        logger.info(f"Metrics exported to {filepath}")
    
    def _export_csv(self, filepath: str):
        """Export metrics to CSV format."""
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['operation', 'duration', 'timestamp', 'metadata']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for metric in self._metrics:
                writer.writerow({
                    'operation': metric.operation,
                    'duration': metric.duration,
                    'timestamp': metric.timestamp,
                    'metadata': json.dumps(metric.metadata)
                })
        
        logger.info(f"Metrics exported to {filepath}")
    
    def reset(self):
        """Reset all collected metrics."""
        with self._lock:
            self._metrics.clear()
            self._operation_times.clear()
            logger.info("Latency metrics reset")
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """
        Get real-time statistics without locking for monitoring purposes.
        
        Returns:
            Dictionary with current statistics
        """
        return {
            'total_measurements': len(self._metrics),
            'operations_count': {op: len(times) for op, times in self._operation_times.items()},
            'recent_metrics': [m.to_dict() for m in self._metrics[-10:]]  # Last 10 measurements
        }

# Global latency tracker instance
latency_tracker = LatencyTracker()

# Convenience functions for common operations
def measure_document_ingestion(func):
    """Decorator for measuring document ingestion operations."""
    return latency_tracker.measure_function('document_ingestion')(func)

def measure_vector_retrieval(func):
    """Decorator for measuring vector database retrieval operations."""
    return latency_tracker.measure_function('vector_retrieval')(func)

def measure_llm_generation(func):
    """Decorator for measuring LLM generation operations."""
    return latency_tracker.measure_function('llm_generation')(func)

def measure_agent_execution(func):
    """Decorator for measuring agent graph execution."""
    return latency_tracker.measure_function('agent_graph_execution')(func)
