# Latency Measurement Guide for RAG System

## Overview

This guide explains how latency measurement is implemented in the RAG system and how to use it effectively for performance analysis and optimization.

## What is Measured

The latency tracking system measures performance across all major stages of the RAG pipeline:

### 1. Document Ingestion and Chunking
- **Operation**: `document_ingestion`, `document_chunking`
- **What's measured**: Time to process raw documents into chunks
- **Metadata tracked**: 
  - Directory path
  - File count
  - Chunk sizes
  - Processing parameters

### 2. Vector Database Operations
- **Operation**: `vector_insertion`, `vector_retrieval`
- **What's measured**: 
  - Time to insert vectors into the database
  - Time to retrieve relevant documents using similarity search
- **Metadata tracked**:
  - Collection name
  - Number of documents retrieved (k)
  - Score threshold
  - Index type and parameters

### 3. Parent Store Operations
- **Operation**: `parent_store_operations`
- **What's measured**: Time for metadata management and parent-child relationships
- **Metadata tracked**: Operation type, document IDs

### 4. Agent Graph Execution
- **Operation**: `agent_graph_execution`
- **What's measured**: Time for the entire agent workflow including:
  - Query analysis and rewriting
  - Tool selection and execution
  - Response synthesis
- **Metadata tracked**: Query complexity, tool usage, execution path

### 5. LLM Generation
- **Operation**: `llm_generation`
- **What's measured**: Time for language model to generate responses
- **Metadata tracked**: Model name, prompt length, temperature settings

### 6. Evaluation Pipeline
- **Operation**: `evaluation_pipeline`
- **What's measured**: Time for evaluation metrics computation
- **Metadata tracked**: Evaluation type, document count, scoring parameters

### 7. End-to-End Query Processing
- **Operation**: `end_to_end_query`
- **What's measured**: Complete time from query input to final response
- **Metadata tracked**: Query text, expected retrieval count, user session

## How to Use

### Basic Usage

```python
from eval.latency import latency_tracker

# Measure a single operation
with latency_tracker.measure_operation('my_operation'):
    # Your code here
    pass

# Measure a complete RAG query
with latency_tracker.measure_query("What is AI?"):
    response = rag_system.query("What is AI?")
```

### Decorator Usage

```python
from eval.latency import measure_document_ingestion, measure_vector_retrieval

@measure_document_ingestion
def process_documents():
    # Document processing code
    pass

@measure_vector_retrieval  
def search_documents(query):
    # Vector search code
    pass
```

### Integration with RAG System

The RAG system is already integrated with latency tracking:

```python
from project.core.rag_system import RAGSystem

# Initialize RAG system (automatically includes latency tracking)
rag_system = RAGSystem()
rag_system.initialize()

# Process query with automatic latency measurement
response = rag_system.query("Your question here")

# Get latency metrics
latency_metrics = response['latency_metrics']
```

## Getting Results

### Real-time Statistics

```python
# Get current statistics
stats = latency_tracker.get_latency_stats()

# Get pipeline breakdown
pipeline_stats = latency_tracker.get_pipeline_breakdown()

# Get performance summary
summary = latency_tracker.get_performance_summary()
```

### Exporting Data

```python
# Export to JSON
latency_tracker.export_metrics('latency_report.json', 'json')

# Export to CSV
latency_tracker.export_metrics('latency_report.csv', 'csv')
```

### CSV Format

The exported CSV contains the following columns:
- `operation`: Name of the operation
- `duration`: Execution time in seconds
- `timestamp`: Unix timestamp of measurement
- `metadata`: JSON string with additional metadata

## Performance Analysis

### Identifying Bottlenecks

The system automatically identifies performance bottlenecks:

```python
summary = latency_tracker.get_performance_summary()
bottlenecks = summary['bottlenecks']

for bottleneck in bottlenecks:
    print(f"Operation: {bottleneck['operation']}")
    print(f"Severity: {bottleneck['severity']}")
    print(f"Average time: {bottleneck['avg_time']:.3f}s")
```

### Optimization Recommendations

The system provides automatic recommendations:

```python
recommendations = summary['recommendations']
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")
```

Common recommendations include:
- **Vector retrieval slow**: Optimize index parameters, increase hardware, implement caching
- **LLM generation slow**: Use faster model, optimize prompt length, implement streaming
- **Document chunking slow**: Optimize chunk size, parallelize processing, use efficient algorithms
- **High variance**: Investigate inconsistent performance patterns

## Statistics Provided

For each operation, the system tracks:

- **Count**: Number of measurements
- **Total time**: Sum of all execution times
- **Min time**: Fastest execution
- **Max time**: Slowest execution  
- **Average time**: Mean execution time
- **Median time**: 50th percentile
- **P95 time**: 95th percentile
- **P99 time**: 99th percentile

## Thread Safety

The latency tracker is thread-safe and can be used in multi-threaded environments:

```python
import threading

def worker_function():
    with latency_tracker.measure_operation('worker_task'):
        # Thread-safe operation
        pass

# Multiple threads can safely use the tracker
threads = []
for i in range(10):
    t = threading.Thread(target=worker_function)
    threads.append(t)
    t.start()
```

## Best Practices

### 1. Granular Measurement
Measure at appropriate granularity:
```python
# Good: Measure meaningful operations
with latency_tracker.measure_operation('document_processing'):
    process_batch(documents)

# Avoid: Too granular
for doc in documents:
    with latency_tracker.measure_operation('single_doc'):
        process_document(doc)
```

### 2. Meaningful Operation Names
Use descriptive operation names:
```python
# Good
latency_tracker.measure_operation('vector_similarity_search')

# Avoid
latency_tracker.measure_operation('op1')
```

### 3. Include Relevant Metadata
Add useful metadata for analysis:
```python
with latency_tracker.measure_operation('llm_generation', {
    'model': 'gpt-4',
    'prompt_length': len(prompt),
    'temperature': 0.7
}):
    response = llm.generate(prompt)
```

### 4. Regular Export and Analysis
Export metrics regularly for long-term analysis:
```python
# Daily export
if datetime.now().hour == 0:  # Midnight
    latency_tracker.export_metrics(f'latency_{date.today()}.csv')
```

### 5. Monitor Key Metrics
Focus on these key performance indicators:
- **P95 latency**: Represents worst-case user experience
- **Average latency**: Overall system performance
- **Variance**: Consistency of performance
- **Bottleneck identification**: Areas needing optimization

## Troubleshooting

### High Latency Issues

1. **Check vector retrieval times**:
   ```python
   retrieval_stats = latency_tracker.get_latency_stats('vector_retrieval')
   if retrieval_stats['avg_time'] > 1.0:
       print("Vector retrieval is slow")
   ```

2. **Monitor LLM generation**:
   ```python
   llm_stats = latency_tracker.get_latency_stats('llm_generation')
   if llm_stats['p95_time'] > 5.0:
       print("LLM generation has high variance")
   ```

3. **Analyze document processing**:
   ```python
   chunking_stats = latency_tracker.get_latency_stats('document_chunking')
   if chunking_stats['max_time'] > chunking_stats['avg_time'] * 3:
       print("Inconsistent chunking performance")
   ```

### Memory Usage

The latency tracker stores all measurements in memory. For long-running systems:

```python
# Periodically export and reset
if len(latency_tracker._metrics) > 10000:
    latency_tracker.export_metrics('backup.csv')
    latency_tracker.reset()
```

## Integration Examples

### With Existing Code

```python
# Before
def process_query(query):
    return rag_system.query(query)

# After
def process_query(query):
    with latency_tracker.measure_query(query):
        return rag_system.query(query)
```

### Custom Operations

```python
def custom_rag_pipeline(query):
    with latency_tracker.measure_operation('custom_pipeline'):
        # Step 1: Preprocessing
        with latency_tracker.measure_operation('preprocessing'):
            processed_query = preprocess(query)
        
        # Step 2: Retrieval
        with latency_tracker.measure_operation('retrieval'):
            docs = retrieve_documents(processed_query)
        
        # Step 3: Generation
        with latency_tracker.measure_operation('generation'):
            response = generate_response(processed_query, docs)
        
        return response
```

## Conclusion

The latency measurement system provides comprehensive insights into RAG system performance. By tracking all major operations and providing detailed statistics, it enables data-driven optimization decisions and helps maintain high-quality user experience.

Regular monitoring and analysis of latency metrics will help identify performance issues early and guide optimization efforts effectively.