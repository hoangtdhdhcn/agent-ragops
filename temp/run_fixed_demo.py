#!/usr/bin/env python3
"""
Simple script to run the fixed latency measurement demo.

This version includes better error handling and fallback mechanisms.
"""

import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'project'))

def main():
    """Run the fixed latency measurement demo."""
    print("🚀 Running Fixed RAG System Latency Measurement Demo")
    print("=" * 60)
    
    try:
        # Import and run the fixed demo
        from project.temp.fixed_demo_latency_measurement import demo_latency_measurement
        
        # Run the demonstration
        demo_latency_measurement()
        
        print("\n✅ Fixed demo completed successfully!")
        print("\n📁 Generated files:")
        print("  - fixed_latency_metrics.json (detailed metrics)")
        print("  - fixed_latency_metrics.csv (CSV format)")
        print("  - fixed_query_results.csv (query performance summary)")
        
        return 0
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure running this script from the project root directory.")
        return 1
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)