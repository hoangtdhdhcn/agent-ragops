#!/usr/bin/env python3
"""
Test script to verify latency measurement integration works correctly.

This script tests the latency measurement system without requiring
the full RAG system setup.
"""

import time
import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'project'))

def test_latency_tracker():
    """Test the basic latency tracker functionality."""
    print("🧪 Testing Latency Tracker...")
    
    from eval.latency import latency_tracker
    
    # Test basic operation measurement
    with latency_tracker.measure_operation('test_operation', {'test': True}):
        time.sleep(0.1)
    
    # Test query measurement
    with latency_tracker.measure_query("Test query", 3):
        time.sleep(0.05)
    
    # Test function decorator
    @latency_tracker.measure_function('decorated_function')
    def test_function():
        time.sleep(0.02)
        return "test result"
    
    result = test_function()
    print(f"✓ Decorated function returned: {result}")
    
    # Get statistics
    stats = latency_tracker.get_latency_stats()
    print(f"✓ Collected {len(stats)} operation types")
    
    # Export to CSV
    latency_tracker.export_metrics('report/test_latency.csv', 'csv')
    print("✓ Exported metrics to test_latency.csv")
    
    return True

def test_convenience_decorators():
    """Test the convenience decorator functions."""
    print("\n🧪 Testing Convenience Decorators...")
    
    from eval.latency import measure_document_ingestion, measure_vector_retrieval
    
    @measure_document_ingestion
    def test_document_processing():
        time.sleep(0.03)
        return "processed"
    
    @measure_vector_retrieval  
    def test_vector_search():
        time.sleep(0.02)
        return ["doc1", "doc2"]
    
    result1 = test_document_processing()
    result2 = test_vector_search()
    
    print(f"✓ Document processing: {result1}")
    print(f"✓ Vector search: {result2}")
    
    return True

def test_pipeline_breakdown():
    """Test pipeline stage analysis."""
    print("\n🧪 Testing Pipeline Breakdown...")
    
    from eval.latency import latency_tracker
    
    # Simulate different pipeline stages
    stages = [
        ('document_ingestion', 0.1),
        ('document_chunking', 0.2),
        ('vector_insertion', 0.3),
        ('vector_retrieval', 0.15),
        ('agent_graph_execution', 0.5),
        ('llm_generation', 1.0),
        ('evaluation_pipeline', 0.2),
        ('end_to_end_query', 1.5)
    ]
    
    for stage, duration in stages:
        with latency_tracker.measure_operation(stage):
            time.sleep(duration)
    
    breakdown = latency_tracker.get_pipeline_breakdown()
    print(f"✓ Pipeline breakdown: {len(breakdown)} stages tracked")
    
    # Show summary
    summary = latency_tracker.get_performance_summary()
    print(f"✓ Total measurements: {summary['total_measurements']}")
    
    return True

def main():
    """Run all latency measurement tests."""
    print("🚀 Testing RAG System Latency Measurement Integration")
    print("=" * 60)
    
    try:
        # Test basic functionality
        test_latency_tracker()
        
        # Test decorators
        test_convenience_decorators()
        
        # Test pipeline analysis
        test_pipeline_breakdown()
        
        print("\n" + "=" * 60)
        print("✅ All latency measurement tests passed!")
        print("\n📁 Generated files:")
        print("  - test_latency.csv (test metrics)")
        
        return 0
        
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)