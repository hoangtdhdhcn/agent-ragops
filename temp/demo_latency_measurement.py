"""
Demonstration script for latency measurement in the RAG system.

This script demonstrates how to:
1. Initialize the RAG system with latency tracking
2. Process queries with comprehensive latency measurement
3. Export latency metrics to CSV format
4. Generate performance reports
"""

import time
import json
from core.rag_system import RAGSystem
from eval.latency import latency_tracker
from eval.evaluation_manager import use_real_evaluators

def demo_latency_measurement():
    """Demonstrate comprehensive latency measurement in the RAG system."""
    
    print("🚀 Starting RAG System Latency Measurement Demo")
    print("=" * 60)
    
    # Initialize RAG system
    print("1. Initializing RAG System...")
    rag_system = RAGSystem()
    rag_system.initialize()
    
    # Switch to real evaluators for accurate measurement
    print("2. Switching to real LLM evaluators...")
    use_real_evaluators()
    
    # Sample queries for testing
    test_queries = [
        # "What is the capital of France?",
        # "How does photosynthesis work?",
        # "What are the main features of Python programming language?",
        # "Explain the concept of machine learning.",
        "What is the difference between supervised and unsupervised learning?"
    ]
    
    print(f"3. Processing {len(test_queries)} test queries with latency tracking...")
    print("-" * 60)
    
    # Process each query and measure latency
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        # Measure the complete query processing
        start_time = time.time()
        
        try:
            response = rag_system.query(query, expected_retrieval_count=3)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Extract relevant information
            query_result = {
                'query': query,
                'answer_length': len(response.get('answer', '')),
                'total_time': total_time,
                'evaluation_score': response.get('evaluation', {}).get('overall_score', 0),
                'timestamp': time.time()
            }
            
            results.append(query_result)
            print(f"✓ Answer generated in {total_time:.3f}s")
            print(f"  Answer length: {query_result['answer_length']} characters")
            print(f"  Evaluation score: {query_result['evaluation_score']:.2%}")
            
        except Exception as e:
            print(f"✗ Error processing query: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("📊 LATENCY ANALYSIS RESULTS")
    print("=" * 60)
    
    # Get comprehensive latency statistics
    latency_stats = latency_tracker.get_latency_stats()
    pipeline_breakdown = latency_tracker.get_pipeline_breakdown()
    performance_summary = latency_tracker.get_performance_summary()
    
    # Display pipeline stage analysis
    print("\n📈 Pipeline Stage Analysis:")
    print("-" * 40)
    for stage, stats in pipeline_breakdown.items():
        if stats.get('count', 0) > 0:
            print(f"{stage:25} | Count: {stats['count']:3} | Avg: {stats['avg_time']:6.3f}s | P95: {stats['p95_time']:6.3f}s")
    
    # Display overall performance summary
    print(f"\n🎯 Performance Summary:")
    print("-" * 40)
    print(f"Total measurements: {performance_summary['total_measurements']}")
    print(f"Operations tracked: {len(performance_summary['operations_tracked'])}")
    
    # Display bottlenecks
    if performance_summary['bottlenecks']:
        print(f"\n⚠️  Performance Bottlenecks:")
        print("-" * 40)
        for bottleneck in performance_summary['bottlenecks']:
            severity = bottleneck.get('severity', 'UNKNOWN')
            time_val = bottleneck.get('avg_time', bottleneck.get('p95_time', 0))
            print(f"{bottleneck['operation']:25} | {severity:6} | {time_val:6.3f}s")
    
    # Display recommendations
    if performance_summary['recommendations']:
        print(f"\n💡 Optimization Recommendations:")
        print("-" * 40)
        for i, rec in enumerate(performance_summary['recommendations'], 1):
            print(f"{i}. {rec}")
    
    # Export results to CSV
    print(f"\n💾 Exporting latency metrics to CSV...")
    print("-" * 40)
    
    # Export detailed metrics
    latency_tracker.export_metrics('report/latency_metrics.json', 'json')
    latency_tracker.export_metrics('report/latency_metrics.csv', 'csv')
    
    # Create a summary CSV with query results
    export_query_results(results)
    
    print("✓ Latency metrics exported successfully!")
    print("\n📁 Generated files:")
    print("  - latency_metrics.json (detailed metrics)")
    print("  - latency_metrics.csv (CSV format)")
    print("  - query_results.csv (query performance summary)")
    
    print("\n" + "=" * 60)
    print("🎉 Latency measurement demo completed!")
    print("=" * 60)

def export_query_results(results: list):
    """Export query results to CSV format."""
    import csv
    
    with open('query_results.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['query', 'answer_length', 'total_time', 'evaluation_score', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'query': result['query'],
                'answer_length': result['answer_length'],
                'total_time': f"{result['total_time']:.3f}",
                'evaluation_score': f"{result['evaluation_score']:.2%}",
                'timestamp': result['timestamp']
            })

def show_real_time_monitoring():
    """Demonstrate real-time latency monitoring capabilities."""
    print("\n🔍 Real-time Monitoring Demo")
    print("-" * 40)
    
    # Get real-time statistics
    real_time_stats = latency_tracker.get_real_time_stats()
    
    print(f"Total measurements: {real_time_stats['total_measurements']}")
    print(f"Operations tracked:")
    for op, count in real_time_stats['operations_count'].items():
        print(f"  - {op}: {count}")
    
    print(f"\nRecent measurements:")
    for i, metric in enumerate(real_time_stats['recent_metrics'][-5:], 1):
        print(f"  {i}. {metric['operation']}: {metric['duration']:.4f}s")

if __name__ == "__main__":
    try:
        demo_latency_measurement()
        show_real_time_monitoring()
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n Demo failed with error: {e}")
        import traceback
        traceback.print_exc()