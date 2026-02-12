from pathlib import Path
import shutil
import config
from util import pdfs_to_markdowns
from production.multi_format_ingestion import MultiFormatIngestionPipeline, DocumentTypeDetector

class DocumentManager:

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.markdown_dir = Path(config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.ingestion_pipeline = MultiFormatIngestionPipeline(self.markdown_dir)
        
    def add_documents(self, document_paths, progress_callback=None):
        """Add documents to the RAG system with multi-format support."""
        if not document_paths:
            return 0, 0
            
        document_paths = [document_paths] if isinstance(document_paths, str) else document_paths
        document_paths = [p for p in document_paths if p and Path(p).exists()]
        
        if not document_paths:
            return 0, 0
            
        added = 0
        skipped = 0
            
        for i, doc_path in enumerate(document_paths):
            if progress_callback:
                progress_callback((i + 1) / len(document_paths), f"Processing {Path(doc_path).name}")
                
            doc_name = Path(doc_path).stem
            md_path = self.markdown_dir / f"{doc_name}.md"
            
            if md_path.exists():
                skipped += 1
                continue
                
            try:
                # Use multi-format ingestion pipeline
                success, processed_path = self.ingestion_pipeline.ingest_file(doc_path)
                
                if not success or not processed_path:
                    skipped += 1
                    continue
                
                # Process the generated markdown
                parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_path)
                
                if not child_chunks:
                    skipped += 1
                    continue
                
                collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
                collection.add_documents(child_chunks)
                self.rag_system.parent_store.save_many(parent_chunks)
                
                added += 1
                
            except Exception as e:
                print(f"Error processing {doc_path}: {e}")
                skipped += 1
            
        return added, skipped
    
    def add_web_page(self, url, metadata=None, progress_callback=None):
        """Add a web page to the RAG system."""
        if progress_callback:
            progress_callback(0.1, f"Fetching {url}")
        
        try:
            success, processed_path = self.ingestion_pipeline.ingest_web_page(url, metadata)
            
            if not success or not processed_path:
                return False, "Failed to process web page"
            
            if progress_callback:
                progress_callback(0.5, "Processing content")
            
            # Process the generated markdown
            md_path = Path(processed_path)
            parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_path)
            
            if not child_chunks:
                return False, "No content extracted from web page"
            
            collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
            collection.add_documents(child_chunks)
            self.rag_system.parent_store.save_many(parent_chunks)
            
            if progress_callback:
                progress_callback(1.0, "Completed")
            
            return True, f"Successfully added web page: {url}"
            
        except Exception as e:
            return False, f"Error processing web page {url}: {e}"
    
    def add_directory(self, directory_path, recursive=True, progress_callback=None):
        """Add all supported documents from a directory."""
        try:
            stats = self.ingestion_pipeline.ingest_directory(directory_path, recursive)
            
            # Process all generated markdown files
            markdown_files = list(self.markdown_dir.glob("*.md"))
            
            total_files = len(markdown_files)
            processed = 0
            added = 0
            skipped = 0
            
            for md_file in markdown_files:
                if progress_callback:
                    progress = (processed + 1) / total_files
                    progress_callback(progress, f"Processing {md_file.name}")
                
                try:
                    parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_file)
                    
                    if child_chunks:
                        collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
                        collection.add_documents(child_chunks)
                        self.rag_system.parent_store.save_many(parent_chunks)
                        added += 1
                    else:
                        skipped += 1
                        
                except Exception as e:
                    print(f"Error processing {md_file}: {e}")
                    skipped += 1
                
                processed += 1
            
            return added, skipped, stats
            
        except Exception as e:
            print(f"Error adding directory {directory_path}: {e}")
            return 0, 0, {'error': str(e)}
    
    def get_supported_formats(self):
        """Get list of supported document formats."""
        return list(DocumentTypeDetector.SUPPORTED_FORMATS.keys())
    
    def get_markdown_files(self):
        if not self.markdown_dir.exists():
            return []
        return sorted([p.name.replace(".md", "") for p in self.markdown_dir.glob("*.md")])
    
    def clear_all(self):
        if self.markdown_dir.exists():
            shutil.rmtree(self.markdown_dir)
            self.markdown_dir.mkdir(parents=True, exist_ok=True)
        
        self.rag_system.parent_store.clear_store()
        self.rag_system.vector_db.delete_collection(self.rag_system.collection_name)
        self.rag_system.vector_db.create_collection(self.rag_system.collection_name)
