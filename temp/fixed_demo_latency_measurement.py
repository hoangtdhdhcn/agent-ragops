"""
Fixed demonstration script for latency measurement in the RAG system.

This script demonstrates how to:
1. Initialize the RAG system with latency tracking
2. Process queries with comprehensive latency measurement
3. Export latency metrics to CSV format
4. Generate performance reports

This version includes better error handling and fallback mechanisms.
"""

import time
import json
import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'project'))

def demo_latency_measurement():
    """Demonstrate comprehensive latency measurement in the RAG system."""
    
    print("🚀 Starting RAG System Latency Measurement Demo (Fixed Version)")
    print("=" * 70)
    
    # Import latency tracker first (this should always work)
    try:
        from eval.latency import latency_tracker
        print("✓ Latency tracker imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import latency tracker: {e}")
        return
    
    # Try to initialize RAG system with fallback
    rag_system = None
    try:
        print("1. Initializing RAG System...")
        from core.rag_system import RAGSystem
        rag_system = RAGSystem()
        rag_system.initialize()
        print("✓ RAG system initialized successfully")
    except Exception as e:
        print(f"⚠️  RAG system initialization failed: {e}")
        print("   Continuing with simulated latency measurements...")
    
    # Try to switch to real evaluators with fallback
    try:
        print("2. Switching to real LLM evaluators...")
        from eval.evaluation_manager import use_real_evaluators
        use_real_evaluators()
        print("✓ Real evaluators activated")
    except Exception as e:
        print(f"⚠️  Failed to activate real evaluators: {e}")
        print("   Using simulated evaluation...")
    
    # Sample queries for testing
    test_queries = [
        # "What is the capital of France?",
        # "How does photosynthesis work?",
        # "What are the main features of Python programming language?",
        # "Explain the concept of machine learning.",
        "What is the difference between supervised and unsupervised learning?"
    ]
    
    print(f"3. Processing {len(test_queries)} test queries with latency tracking...")
    print("-" * 70)
    
    # Process each query and measure latency
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 50)
        
        # Measure the complete query processing
        start_time = time.time()
        
        try:
            if rag_system:
                # Try to use real RAG system
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
            else:
                # Simulate RAG system response for testing
                with latency_tracker.measure_query(query, 3):
                    # Simulate agent execution
                    with latency_tracker.measure_operation('agent_graph_execution', {
                        'question_length': len(query),
                        'expected_retrieval_count': 3
                    }):
                        # Simulate processing time
                        time.sleep(0.1 + len(query) * 0.001)
                    
                    # Simulate LLM generation
                    with latency_tracker.measure_operation('llm_generation', {
                        'model': 'simulated',
                        'prompt_length': len(query)
                    }):
                        time.sleep(0.5 + len(query) * 0.002)
                
                # Create simulated response
                simulated_answer = f"Simulated answer for: {query}"
                query_result = {
                    'query': query,
                    'answer_length': len(simulated_answer),
                    'total_time': time.time() - start_time,
                    'evaluation_score': 0.85,  # Simulated score
                    'timestamp': time.time()
                }
                
                results.append(query_result)
                print(f"✓ Simulated answer generated in {query_result['total_time']:.3f}s")
                print(f"  Answer length: {query_result['answer_length']} characters")
                print(f"  Evaluation score: {query_result['evaluation_score']:.2%}")
            
        except Exception as e:
            print(f"✗ Error processing query: {e}")
            # Still track some latency even if there's an error
            with latency_tracker.measure_operation('query_error', {'query': query}):
                time.sleep(0.01)
            continue
    
    print("\n" + "=" * 70)
    print("📊 LATENCY ANALYSIS RESULTS")
    print("=" * 70)
    
    # Get comprehensive latency statistics
    try:
        latency_stats = latency_tracker.get_latency_stats()
        pipeline_breakdown = latency_tracker.get_pipeline_breakdown()
        performance_summary = latency_tracker.get_performance_summary()
        
        # Display pipeline stage analysis
        print("\n📈 Pipeline Stage Analysis:")
        print("-" * 50)
        for stage, stats in pipeline_breakdown.items():
            if stats.get('count', 0) > 0:
                print(f"{stage:25} | Count: {stats['count']:3} | Avg: {stats['avg_time']:6.3f}s | P95: {stats['p95_time']:6.3f}s")
        
        # Display overall performance summary
        print(f"\n🎯 Performance Summary:")
        print("-" * 50)
        print(f"Total measurements: {performance_summary['total_measurements']}")
        print(f"Operations tracked: {len(performance_summary['operations_tracked'])}")
        
        # Display bottlenecks
        if performance_summary['bottlenecks']:
            print(f"\n⚠️  Performance Bottlenecks:")
            print("-" * 50)
            for bottleneck in performance_summary['bottlenecks']:
                severity = bottleneck.get('severity', 'UNKNOWN')
                time_val = bottleneck.get('avg_time', bottleneck.get('p95_time', 0))
                print(f"{bottleneck['operation']:25} | {severity:6} | {time_val:6.3f}s")
        
        # Display recommendations
        if performance_summary['recommendations']:
            print(f"\n💡 Optimization Recommendations:")
            print("-" * 50)
            for i, rec in enumerate(performance_summary['recommendations'], 1):
                print(f"{i}. {rec}")
        
        # Export results to CSV
        print(f"\n💾 Exporting latency metrics to CSV...")
        print("-" * 50)
        
        # Export detailed metrics
        latency_tracker.export_metrics('report/fixed_latency_metrics.json', 'json')
        latency_tracker.export_metrics('report/fixed_latency_metrics.csv', 'csv')
        
        # Create a summary CSV with query results
        export_query_results(results, 'report/fixed_query_results.csv')
        
        print("✓ Latency metrics exported successfully!")
        print("\n📁 Generated files:")
        print("  - fixed_latency_metrics.json (detailed metrics)")
        print("  - fixed_latency_metrics.csv (CSV format)")
        print("  - fixed_query_results.csv (query performance summary)")
        
    except Exception as e:
        print(f"✗ Failed to generate analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("🎉 Latency measurement demo completed!")
    print("=" * 70)

def export_query_results(results: list, filename: str = 'query_results.csv'):
    """Export query results to CSV format."""
    import csv
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
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
    print("-" * 50)
    
    try:
        # Import latency tracker for this function
        from eval.latency import latency_tracker
        
        # Get real-time statistics
        real_time_stats = latency_tracker.get_real_time_stats()
        
        print(f"Total measurements: {real_time_stats['total_measurements']}")
        print(f"Operations tracked:")
        for op, count in real_time_stats['operations_count'].items():
            print(f"  - {op}: {count}")
        
        print(f"\nRecent measurements:")
        for i, metric in enumerate(real_time_stats['recent_metrics'][-5:], 1):
            print(f"  {i}. {metric['operation']}: {metric['duration']:.4f}s")
    except Exception as e:
        print(f"Failed to show real-time monitoring: {e}")

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