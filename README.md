# Agentic RAG - Production Edition

This is an enhanced version of the Agentic RAG project with production-ready features for enterprise document management and retrieval.

## 🚀 New Production Features

### 1. Multi-Format Ingestion Pipeline
- **Supported Formats**: PDF, HTML, DOCX, CSV, Excel, PowerPoint, Images (with OCR), Web pages
- **Smart Detection**: Automatic format detection and routing
- **Robust Processing**: Error handling and fallback mechanisms

### 2. Incremental Ingestion & Delta Processing
- **Change Detection**: Hash-based document change detection
- **Partial Processing**: Only reprocess changed content
- **Background Processing**: Queue-based processing for large batches

### 3. Deduplication System
- **Exact Deduplication**: SHA-256 hash-based duplicate detection
- **Near-Duplicate Detection**: Semantic similarity for similar documents
- **Cross-Document Handling**: Detect duplicates across different sources

### 4. Stable Document IDs & Rich Metadata
- **UUID-based IDs**: Stable, unique document identifiers
- **Comprehensive Metadata**: Author, creation date, tags, version, source tracking
- **Custom Fields**: Extensible metadata schema

### 5. Versioning System
- **Document Versioning**: Track changes across document updates
- **Embedding Versioning**: Version-aware vector embeddings
- **Rollback Support**: Restore previous document versions

### 6. Idempotent Indexing
- **Transactional Operations**: Atomic indexing operations
- **Retry Mechanisms**: Automatic retry with deduplication
- **Consistent State**: Guaranteed consistency across failures

## 📁 Project Structure

```
project/
├── multi_format_ingestion.py      # Multi-format document processing
├── enhanced_document_manager.py   # Production document management
├── core/
│   ├── document_manager.py        # Original document manager (updated)
│   ├── rag_system.py             # RAG system core
│   └── chat_interface.py         # Chat interface
├── rag_agent/                    # Multi-agent system
├── db/                          # Database management
├── ui/                          # User interface
└── requirements.txt             # Updated dependencies
```

## 🛠️ Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Additional System Dependencies

For OCR support:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/tesseract-ocr/tesseract
```

### 3. Setup Environment

```bash
# Copy environment template
cp .env.template .env

# Configure your settings in .env
```

## 📖 Usage

### Basic Usage

```python
from enhanced_document_manager import EnhancedDocumentManager
from rag_system import RAGSystem

# Initialize RAG system
rag_system = RAGSystem()
rag_system.initialize()

# Create enhanced document manager
doc_manager = EnhancedDocumentManager(rag_system)

# Add documents
success, message, metadata = doc_manager.add_document("path/to/document.pdf")
print(f"Added: {message}")

# Add web page
success, message, metadata = doc_manager.add_web_page("https://example.com")
print(f"Added: {message}")

# Add entire directory
results = doc_manager.add_directory("path/to/documents/", recursive=True)
print(f"Processed: {results['processed']}, Added: {results['added']}")
```

### Advanced Features

```python
# Get document statistics
stats = doc_manager.get_statistics()
print(f"Total documents: {stats['total_documents']}")
print(f"By type: {stats['by_type']}")

# List documents with filtering
documents = doc_manager.list_documents(document_type='pdf', tags=['important'])
for doc in documents:
    print(f"ID: {doc['document_id']}, Title: {doc.get('title', 'N/A')}")

# Get document information
doc_info = doc_manager.get_document_info("doc_id_here")
print(f"Document: {doc_info['original_filename']}")
print(f"Version: {doc_info['version']}")
```

### Multi-Format Support Examples

```python
# PDF documents
doc_manager.add_document("report.pdf")

# Word documents
doc_manager.add_document("proposal.docx")

# Excel spreadsheets
doc_manager.add_document("data.xlsx")

# CSV files
doc_manager.add_document("dataset.csv")

# PowerPoint presentations
doc_manager.add_document("presentation.pptx")

# HTML files
doc_manager.add_document("page.html")

# Web pages
doc_manager.add_web_page("https://docs.example.com")

# Images with OCR
doc_manager.add_document("scanned_document.jpg")
```

## 🔧 Configuration

### Document Processing Settings

```python
# Enable/disable features
doc_manager = EnhancedDocumentManager(
    rag_system,
    enable_deduplication=True,  # Enable duplicate detection
    enable_versioning=True      # Enable version tracking
)
```

### Metadata Extraction

```python
# Custom metadata
metadata = {
    'author': 'John Doe',
    'tags': ['important', 'finance'],
    'department': 'Accounting'
}

doc_manager.add_document("file.pdf", metadata=metadata)
```

### Batch Processing

```python
# Process directory with progress callback
def progress_callback(progress, message):
    print(f"Progress: {progress:.1%} - {message}")

results = doc_manager.add_directory(
    "documents/", 
    recursive=True, 
    progress_callback=progress_callback
)
```

## 🏗️ Architecture

### Multi-Format Ingestion Pipeline

```
Input Document → Format Detection → Document Processor → Markdown → RAG System
     ↓              ↓                    ↓              ↓           ↓
  PDF/DOCX/etc   PDF/HTML/DOCX/etc   PDFProcessor    Text     Vector DB
                HTML/CSV/Excel/etc   HTMLProcessor   Chunks   Parent Store
                Images/Web/etc       DOCXProcessor
                                     CSVProcessor
                                     ExcelProcessor
                                     PPTXProcessor
                                     ImageProcessor
```

### Production Features Integration

```
Document → Hash Calculation → Duplicate Check → Version Check → Process → Index
   ↓            ↓                ↓              ↓            ↓        ↓
Metadata ← Metadata Extraction ← ID Generation ← Backup ← Save ← Vector DB
```

## 📊 Monitoring & Statistics

### Document Statistics

```python
stats = doc_manager.get_statistics()
print(json.dumps(stats, indent=2))
```

Output:
```json
{
  "total_documents": 150,
  "by_type": {
    "pdf": 80,
    "docx": 30,
    "html": 25,
    "csv": 10,
    "image": 5
  },
  "by_status": {
    "completed": 145,
    "failed": 5
  },
  "total_duplicates_checked": 200,
  "metadata_files": 150
}
```

### Document Registry

Documents are tracked in a comprehensive registry with:
- Document IDs and metadata
- Processing status and timestamps
- Version history
- Error tracking

## 🔒 Security & Compliance

### Data Protection
- **Encryption**: Sensitive metadata encryption
- **Access Control**: Document-level access permissions
- **Audit Trail**: Complete processing history

### Compliance Features
- **GDPR Compliance**: Data deletion and anonymization
- **Audit Logging**: Complete processing logs
- **Backup & Recovery**: Automated backup systems

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the Docker image
docker build -f project/Dockerfile -t rag-assistant .

# Run the container
docker run -p 7860:7860 rag-assistant

# Or use docker-compose for local development
docker-compose up -d
```

### GitHub Actions CI/CD

The project includes automated Docker image building via GitHub Actions:

- **Triggers**: Push to main/master branches or pull requests
- **Registry**: Images are pushed to GitHub Container Registry (ghcr.io)
- **Tags**: Automatic tagging with branch names, SHA, and latest
- **Caching**: Docker layer caching for faster builds

### Docker Compose for Development

```bash
# Start the application with persistent storage
docker-compose up -d

# View logs
docker-compose logs -f

# Stop and clean up
docker-compose down -v
```

## 🚀 Performance Optimization

### Batch Processing
- **Parallel Processing**: Multi-threaded document processing
- **Memory Management**: Efficient memory usage for large documents
- **Caching**: Intelligent caching for repeated operations

### Scalability
- **Horizontal Scaling**: Distributed processing support
- **Load Balancing**: Automatic load distribution
- **Resource Management**: Dynamic resource allocation

## 🧪 Testing

### Unit Tests

```bash
# Run unit tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_document_manager.py::test_add_document -v
```

### Integration Tests

```bash
# Run integration tests
python -m pytest tests/integration/ -v

# Test multi-format ingestion
python -m pytest tests/integration/test_multi_format.py -v
```

### Performance Tests

```bash
# Run performance benchmarks
python tests/performance/benchmark.py

# Test large document processing
python tests/performance/test_large_documents.py
```

## 📋 Production Checklist

### Before Deployment

- [ ] Install all dependencies
- [ ] Configure environment variables
- [ ] Set up vector database (Qdrant)
- [ ] Configure OCR (Tesseract)
- [ ] Set up monitoring and logging
- [ ] Configure backup systems
- [ ] Test with sample documents
- [ ] Validate security settings

### Monitoring Setup

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Monitor document processing
def monitor_callback(event, data):
    if event == 'document_processed':
        print(f"Processed: {data['filename']}")
    elif event == 'error':
        print(f"Error: {data['message']}")

doc_manager.set_monitor_callback(monitor_callback)
```

### Backup Configuration

```python
# Configure automatic backups
doc_manager.configure_backup(
    backup_dir="/path/to/backups",
    backup_interval=3600,  # 1 hour
    retention_days=30
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and ensure they pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Create a GitHub issue
- Check the documentation
- Review the examples
- Join our community discussions

## 📚 Documentation

- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Deployment Guide](docs/deployment.md)
- [Performance Tuning](docs/performance.md)
- [Security Guide](docs/security.md)

---

**Note**: This is a production-ready enhancement of the original Agentic RAG project. All original functionality is preserved while adding enterprise-grade features for document management and retrieval.