"""
Production Monitoring System for RAG Delta Processing

Provides comprehensive monitoring, observability, and alerting capabilities
for the delta processing system with real-time metrics and health checks.
"""

import asyncio
import json
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import psutil
import GPUtil
from prometheus_client import Gauge, Counter, Histogram, start_http_server
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System-level metrics for monitoring."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    network_io: Dict[str, float] = None
    
    def __post_init__(self):
        if self.network_io is None:
            self.network_io = {}

@dataclass
class ProcessingMetrics:
    """Delta processing specific metrics."""
    timestamp: datetime
    queue_size: int
    priority_queue_size: int
    active_workers: int
    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
    avg_processing_time: float
    throughput: float
    error_rate: float
    
@dataclass
class Alert:
    """Alert for monitoring system."""
    alert_id: str
    timestamp: datetime
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    metric_name: str
    current_value: float
    threshold: float
    resolved: bool = False

class MetricsCollector:
    """Collects and aggregates system and processing metrics."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.system_metrics = deque(maxlen=10000)  # Store last 10k entries
        self.processing_metrics = deque(maxlen=10000)
        self.alerts = deque(maxlen=1000)
        
        # Prometheus metrics
        self.setup_prometheus_metrics()
        
        # Collection settings
        self.collection_interval = 30  # seconds
        self.is_collecting = False
        self.collection_thread = None
        
    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics for external monitoring."""
        # System metrics
        self.cpu_gauge = Gauge('rag_system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_gauge = Gauge('rag_system_memory_usage_percent', 'Memory usage percentage')
        self.disk_gauge = Gauge('rag_system_disk_usage_percent', 'Disk usage percentage')
        self.gpu_gauge = Gauge('rag_system_gpu_usage_percent', 'GPU usage percentage')
        self.gpu_memory_gauge = Gauge('rag_system_gpu_memory_usage_percent', 'GPU memory usage percentage')
        
        # Processing metrics
        self.queue_size_gauge = Gauge('rag_processing_queue_size', 'Current queue size')
        self.priority_queue_size_gauge = Gauge('rag_processing_priority_queue_size', 'Priority queue size')
        self.active_workers_gauge = Gauge('rag_processing_active_workers', 'Active worker threads')
        self.completed_tasks_counter = Counter('rag_processing_completed_tasks_total', 'Total completed tasks')
        self.failed_tasks_counter = Counter('rag_processing_failed_tasks_total', 'Total failed tasks')
        self.processing_time_histogram = Histogram('rag_processing_time_seconds', 'Task processing time')
        self.throughput_gauge = Gauge('rag_processing_throughput_tasks_per_second', 'Processing throughput')
        
    def start_collection(self, prometheus_port: int = 8000):
        """Start metrics collection."""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        
        # Start Prometheus server
        try:
            start_http_server(prometheus_port)
            logger.info(f"Prometheus metrics server started on port {prometheus_port}")
        except Exception as e:
            logger.warning(f"Could not start Prometheus server: {e}")
        
        # Start collection thread
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.is_collecting = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.is_collecting:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Update Prometheus metrics
                self._update_prometheus_system_metrics(system_metrics)
                
                # Clean old metrics
                self._clean_old_metrics()
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # GPU usage (if available)
            gpu_usage = None
            gpu_memory = None
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_usage = gpu.load * 100
                    gpu_memory = (gpu.memoryUsed / gpu.memoryTotal) * 100
            except:
                pass
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory,
                network_io=network_io
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={}
            )
    
    def _update_prometheus_system_metrics(self, metrics: SystemMetrics):
        """Update Prometheus system metrics."""
        self.cpu_gauge.set(metrics.cpu_usage)
        self.memory_gauge.set(metrics.memory_usage)
        self.disk_gauge.set(metrics.disk_usage)
        
        if metrics.gpu_usage is not None:
            self.gpu_gauge.set(metrics.gpu_usage)
        if metrics.gpu_memory is not None:
            self.gpu_memory_gauge.set(metrics.gpu_memory)
    
    def update_processing_metrics(self, metrics: ProcessingMetrics):
        """Update processing metrics."""
        self.processing_metrics.append(metrics)
        
        # Update Prometheus processing metrics
        self.queue_size_gauge.set(metrics.queue_size)
        self.priority_queue_size_gauge.set(metrics.priority_queue_size)
        self.active_workers_gauge.set(metrics.active_workers)
        self.throughput_gauge.set(metrics.throughput)
        
        # Note: Counters and histograms are updated when tasks complete
    
    def _clean_old_metrics(self):
        """Clean old metrics beyond retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        # Clean system metrics
        while self.system_metrics and self.system_metrics[0].timestamp < cutoff_time:
            self.system_metrics.popleft()
        
        # Clean processing metrics
        while self.processing_metrics and self.processing_metrics[0].timestamp < cutoff_time:
            self.processing_metrics.popleft()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        if not self.system_metrics:
            return {'status': 'unknown', 'metrics': {}}
        
        latest = self.system_metrics[-1]
        
        # Determine health status
        status = 'healthy'
        issues = []
        
        if latest.cpu_usage > 80:
            status = 'warning'
            issues.append(f"High CPU usage: {latest.cpu_usage:.1f}%")
        
        if latest.memory_usage > 85:
            status = 'warning'
            issues.append(f"High memory usage: {latest.memory_usage:.1f}%")
        
        if latest.disk_usage > 90:
            status = 'critical'
            issues.append(f"Critical disk usage: {latest.disk_usage:.1f}%")
        
        if latest.gpu_usage and latest.gpu_usage > 90:
            status = 'warning'
            issues.append(f"High GPU usage: {latest.gpu_usage:.1f}%")
        
        return {
            'status': status,
            'issues': issues,
            'metrics': {
                'cpu_usage': latest.cpu_usage,
                'memory_usage': latest.memory_usage,
                'disk_usage': latest.disk_usage,
                'gpu_usage': latest.gpu_usage,
                'gpu_memory': latest.gpu_memory
            }
        }
    
    def get_processing_health(self) -> Dict[str, Any]:
        """Get current processing health status."""
        if not self.processing_metrics:
            return {'status': 'unknown', 'metrics': {}}
        
        latest = self.processing_metrics[-1]
        
        # Determine health status
        status = 'healthy'
        issues = []
        
        if latest.error_rate > 0.1:  # More than 10% error rate
            status = 'critical'
            issues.append(f"High error rate: {latest.error_rate:.1%}")
        
        if latest.queue_size > 1000:  # Large queue backlog
            status = 'warning'
            issues.append(f"Large queue backlog: {latest.queue_size}")
        
        if latest.throughput < 1.0:  # Very low throughput
            status = 'warning'
            issues.append(f"Low throughput: {latest.throughput:.1f} tasks/sec")
        
        return {
            'status': status,
            'issues': issues,
            'metrics': {
                'queue_size': latest.queue_size,
                'priority_queue_size': latest.priority_queue_size,
                'active_workers': latest.active_workers,
                'completed_tasks': latest.completed_tasks,
                'failed_tasks': latest.failed_tasks,
                'skipped_tasks': latest.skipped_tasks,
                'avg_processing_time': latest.avg_processing_time,
                'throughput': latest.throughput,
                'error_rate': latest.error_rate
            }
        }
    
    def get_historical_metrics(self, hours: int = 1) -> Dict[str, List]:
        """Get historical metrics for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        system_data = [
            asdict(m) for m in self.system_metrics 
            if m.timestamp >= cutoff_time
        ]
        
        processing_data = [
            asdict(m) for m in self.processing_metrics 
            if m.timestamp >= cutoff_time
        ]
        
        return {
            'system_metrics': system_data,
            'processing_metrics': processing_data,
            'time_period': f"{hours} hours"
        }

class AlertManager:
    """Manages alerts and notifications for the monitoring system."""
    
    def __init__(self):
        self.alerts = deque(maxlen=1000)
        self.active_alerts = {}  # alert_id -> Alert
        self.alert_callbacks = []  # List of callback functions
        
        # Alert thresholds
        self.thresholds = {
            'cpu_usage_high': 80.0,
            'memory_usage_high': 85.0,
            'disk_usage_critical': 90.0,
            'gpu_usage_high': 90.0,
            'error_rate_critical': 0.1,
            'queue_size_warning': 1000,
            'throughput_low': 1.0
        }
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics for alert conditions."""
        self._check_threshold(
            'cpu_usage_high', 
            metrics.cpu_usage, 
            f"High CPU usage: {metrics.cpu_usage:.1f}%",
            metrics
        )
        
        self._check_threshold(
            'memory_usage_high', 
            metrics.memory_usage, 
            f"High memory usage: {metrics.memory_usage:.1f}%",
            metrics
        )
        
        self._check_threshold(
            'disk_usage_critical', 
            metrics.disk_usage, 
            f"Critical disk usage: {metrics.disk_usage:.1f}%",
            metrics
        )
        
        if metrics.gpu_usage is not None:
            self._check_threshold(
                'gpu_usage_high', 
                metrics.gpu_usage, 
                f"High GPU usage: {metrics.gpu_usage:.1f}%",
                metrics
            )
    
    def check_processing_alerts(self, metrics: ProcessingMetrics):
        """Check processing metrics for alert conditions."""
        self._check_threshold(
            'error_rate_critical', 
            metrics.error_rate, 
            f"High error rate: {metrics.error_rate:.1%}",
            metrics
        )
        
        self._check_threshold(
            'queue_size_warning', 
            metrics.queue_size, 
            f"Large queue backlog: {metrics.queue_size}",
            metrics
        )
        
        self._check_threshold(
            'throughput_low', 
            metrics.throughput, 
            f"Low throughput: {metrics.throughput:.1f} tasks/sec",
            metrics
        )
    
    def _check_threshold(self, threshold_name: str, value: float, message: str, metrics: Union[SystemMetrics, ProcessingMetrics]):
        """Check if a metric exceeds its threshold and create alert if needed."""
        threshold = self.thresholds.get(threshold_name, float('inf'))
        
        # Determine if alert should be created
        should_alert = False
        severity = 'warning'
        
        if threshold_name in ['disk_usage_critical', 'error_rate_critical']:
            should_alert = value >= threshold
            severity = 'critical'
        else:
            should_alert = value >= threshold
            severity = 'warning'
        
        if should_alert:
            alert_id = f"{threshold_name}_{metrics.timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            if alert_id not in self.active_alerts:
                alert = Alert(
                    alert_id=alert_id,
                    timestamp=metrics.timestamp,
                    severity=severity,
                    message=message,
                    metric_name=threshold_name,
                    current_value=value,
                    threshold=threshold
                )
                
                self.active_alerts[alert_id] = alert
                self.alerts.append(alert)
                
                # Notify callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
        else:
            # Check if we should resolve existing alerts
            self._resolve_alerts(threshold_name, value)
    
    def _resolve_alerts(self, threshold_name: str, current_value: float):
        """Resolve alerts that are no longer active."""
        threshold = self.thresholds.get(threshold_name, float('inf'))
        
        # Determine resolution threshold (80% of alert threshold)
        resolution_threshold = threshold * 0.8
        
        alerts_to_resolve = []
        for alert_id, alert in self.active_alerts.items():
            if (alert.metric_name == threshold_name and 
                alert.current_value >= threshold and 
                current_value < resolution_threshold):
                alerts_to_resolve.append(alert_id)
        
        for alert_id in alerts_to_resolve:
            alert = self.active_alerts.pop(alert_id)
            alert.resolved = True
            alert.message = f"Resolved: {alert.message}"
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert resolution callback: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alerts."""
        active_alerts = self.get_active_alerts()
        
        summary = {
            'total_active': len(active_alerts),
            'by_severity': defaultdict(int),
            'by_metric': defaultdict(int),
            'recent_resolutions': []
        }
        
        for alert in active_alerts:
            summary['by_severity'][alert.severity] += 1
            summary['by_metric'][alert.metric_name] += 1
        
        # Get recent resolutions
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_resolutions = [
            alert for alert in self.alerts 
            if alert.resolved and alert.timestamp >= cutoff_time
        ]
        summary['recent_resolutions'] = recent_resolutions
        
        return summary

class MonitoringDashboard:
    """Provides a dashboard interface for monitoring system health."""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        
        # Dashboard settings
        self.update_interval = 60  # seconds
        self.is_running = False
        self.dashboard_thread = None
    
    def start_dashboard(self):
        """Start the monitoring dashboard."""
        if self.is_running:
            return
        
        self.is_running = True
        self.dashboard_thread = threading.Thread(target=self._dashboard_loop, daemon=True)
        self.dashboard_thread.start()
        
        logger.info("Started monitoring dashboard")
    
    def stop_dashboard(self):
        """Stop the monitoring dashboard."""
        self.is_running = False
        if self.dashboard_thread and self.dashboard_thread.is_alive():
            self.dashboard_thread.join(timeout=5)
        logger.info("Stopped monitoring dashboard")
    
    def _dashboard_loop(self):
        """Main dashboard update loop."""
        while self.is_running:
            try:
                # Get current status
                system_health = self.metrics_collector.get_system_health()
                processing_health = self.metrics_collector.get_processing_health()
                alert_summary = self.alert_manager.get_alert_summary()
                
                # Log dashboard status
                self._log_dashboard_status(system_health, processing_health, alert_summary)
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in dashboard loop: {e}")
                time.sleep(10)
    
    def _log_dashboard_status(self, system_health: Dict, processing_health: Dict, alert_summary: Dict):
        """Log current dashboard status."""
        logger.info("=" * 60)
        logger.info("MONITORING DASHBOARD STATUS")
        logger.info("=" * 60)
        
        # System health
        logger.info(f"System Health: {system_health['status']}")
        if system_health['issues']:
            for issue in system_health['issues']:
                logger.warning(f"  - {issue}")
        
        # Processing health
        logger.info(f"Processing Health: {processing_health['status']}")
        if processing_health['issues']:
            for issue in processing_health['issues']:
                logger.warning(f"  - {issue}")
        
        # Alerts
        logger.info(f"Active Alerts: {alert_summary['total_active']}")
        for severity, count in alert_summary['by_severity'].items():
            logger.info(f"  {severity.upper()}: {count}")
        
        logger.info("=" * 60)
    
    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate a comprehensive monitoring report."""
        # Get historical data
        historical_data = self.metrics_collector.get_historical_metrics(hours)
        
        # Get health status
        system_health = self.metrics_collector.get_system_health()
        processing_health = self.metrics_collector.get_processing_health()
        
        # Get alert summary
        alert_summary = self.alert_manager.get_alert_summary()
        
        # Calculate statistics
        stats = self._calculate_statistics(historical_data)
        
        return {
            'report_time': datetime.now().isoformat(),
            'time_period': f"{hours} hours",
            'system_health': system_health,
            'processing_health': processing_health,
            'alert_summary': alert_summary,
            'statistics': stats,
            'historical_data': historical_data
        }
    
    def _calculate_statistics(self, historical_data: Dict[str, List]) -> Dict[str, Any]:
        """Calculate statistics from historical data."""
        stats = {
            'system': {},
            'processing': {}
        }
        
        # System metrics statistics
        if historical_data['system_metrics']:
            cpu_values = [m['cpu_usage'] for m in historical_data['system_metrics']]
            memory_values = [m['memory_usage'] for m in historical_data['system_metrics']]
            disk_values = [m['disk_usage'] for m in historical_data['system_metrics']]
            
            stats['system'] = {
                'cpu': {
                    'avg': statistics.mean(cpu_values),
                    'max': max(cpu_values),
                    'min': min(cpu_values),
                    'std': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
                },
                'memory': {
                    'avg': statistics.mean(memory_values),
                    'max': max(memory_values),
                    'min': min(memory_values),
                    'std': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
                },
                'disk': {
                    'avg': statistics.mean(disk_values),
                    'max': max(disk_values),
                    'min': min(disk_values),
                    'std': statistics.stdev(disk_values) if len(disk_values) > 1 else 0
                }
            }
        
        # Processing metrics statistics
        if historical_data['processing_metrics']:
            throughput_values = [m['throughput'] for m in historical_data['processing_metrics']]
            error_rate_values = [m['error_rate'] for m in historical_data['processing_metrics']]
            queue_size_values = [m['queue_size'] for m in historical_data['processing_metrics']]
            
            stats['processing'] = {
                'throughput': {
                    'avg': statistics.mean(throughput_values),
                    'max': max(throughput_values),
                    'min': min(throughput_values),
                    'std': statistics.stdev(throughput_values) if len(throughput_values) > 1 else 0
                },
                'error_rate': {
                    'avg': statistics.mean(error_rate_values),
                    'max': max(error_rate_values),
                    'min': min(error_rate_values),
                    'std': statistics.stdev(error_rate_values) if len(error_rate_values) > 1 else 0
                },
                'queue_size': {
                    'avg': statistics.mean(queue_size_values),
                    'max': max(queue_size_values),
                    'min': min(queue_size_values),
                    'std': statistics.stdev(queue_size_values) if len(queue_size_values) > 1 else 0
                }
            }
        
        return stats
    
    def export_metrics_to_file(self, filename: str, hours: int = 24):
        """Export metrics to a JSON file."""
        historical_data = self.metrics_collector.get_historical_metrics(hours)
        
        with open(filename, 'w') as f:
            json.dump(historical_data, f, indent=2, default=str)
        
        logger.info(f"Exported metrics to {filename}")


# Convenience functions for easy integration
def setup_monitoring_system(rag_system, delta_processor, prometheus_port: int = 8000):
    """Setup complete monitoring system for RAG delta processing."""
    
    # Create monitoring components
    metrics_collector = MetricsCollector()
    alert_manager = AlertManager()
    dashboard = MonitoringDashboard(metrics_collector, alert_manager)
    
    # Start metrics collection
    metrics_collector.start_collection(prometheus_port)
    
    # Start dashboard
    dashboard.start_dashboard()
    
    # Setup alert callbacks
    def alert_callback(alert: Alert):
        logger.warning(f"ALERT [{alert.severity.upper()}]: {alert.message}")
    
    alert_manager.add_alert_callback(alert_callback)
    
    # Integrate with delta processor
    def update_processing_metrics():
        """Update processing metrics from delta processor."""
        if delta_processor:
            status = delta_processor.get_queue_status()
            processing_stats = delta_processor.get_processing_stats()
            
            # Create processing metrics object
            metrics = ProcessingMetrics(
                timestamp=datetime.now(),
                queue_size=status['queue_size'],
                priority_queue_size=status['priority_queue_size'],
                active_workers=status.get('active_workers', 0),
                completed_tasks=status['completed_tasks'],
                failed_tasks=status['failed_tasks'],
                skipped_tasks=status['skipped_tasks'],
                avg_processing_time=processing_stats.avg_processing_time,
                throughput=processing_stats.throughput,
                error_rate=processing_stats.failed_tasks / max(processing_stats.total_tasks, 1)
            )
            
            # Update metrics
            metrics_collector.update_processing_metrics(metrics)
            
            # Check for alerts
            alert_manager.check_processing_alerts(metrics)
    
    # Setup periodic updates
    def periodic_update():
        while True:
            try:
                update_processing_metrics()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error in periodic update: {e}")
                time.sleep(60)
    
    # Start periodic update thread
    update_thread = threading.Thread(target=periodic_update, daemon=True)
    update_thread.start()
    
    return {
        'metrics_collector': metrics_collector,
        'alert_manager': alert_manager,
        'dashboard': dashboard,
        'update_thread': update_thread
    }


if __name__ == "__main__":
    
    # rag_system = RAGSystem()  # Assume RAGSystem is imported
    # delta_processor = DeltaProcessor(rag_system, doc_manager)
    
    # monitoring_system = setup_monitoring_system(rag_system, delta_processor)
    
    # Generate report
    # report = monitoring_system['dashboard'].generate_report(hours=24)
    # print(json.dumps(report, indent=2))
    
    # Export metrics
    # monitoring_system['dashboard'].export_metrics_to_file('metrics_report.json')
    pass