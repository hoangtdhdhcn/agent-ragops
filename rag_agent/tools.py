from typing import List
from langchain_core.tools import tool
from db.parent_store_manager import ParentStoreManager

class ToolFactory:
    
    def __init__(self, collection):
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()
    
    def _search_child_chunks(self, query: str, limit: int) -> tuple:
        """Search for the top K most relevant child chunks.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            tuple: (list of Document objects, formatted string)
        """
        try:
            # print(f"\n🔍 RAG Retrieval: Searching for '{query}' (limit: {limit})")
            results = self.collection.similarity_search(query, k=limit, score_threshold=0.7)
            
            if not results:
                # print("No relevant chunks found")
                return [], "NO_RELEVANT_CHUNKS"

            # Format the retrieved chunks
            formatted_chunks = "\n\n".join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])
            
            # Print the retrieved chunks to terminal
            # print(f"\n📋 Retrieved {len(results)} chunks:")
            # print("=" * 80)
            # print(formatted_chunks)
            # print("=" * 80)
            
            return results, formatted_chunks            

        except Exception as e:
            print(f"Retrieval error: {str(e)}")
            return [], f"RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_many_parent_chunks(self, parent_ids: List[str]) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_ids: List of parent chunk IDs to retrieve
        """
        try:
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)
            raw_parents = self.parent_store_manager.load_content_many(ids)
            if not raw_parents:
                return "NO_PARENT_DOCUMENTS"

            return "\n\n".join([
                f"Parent ID: {doc.get('parent_id', 'n/a')}\n"
                f"File Name: {doc.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {doc.get('content', '').strip()}"
                for doc in raw_parents
            ])            

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_id: Parent chunk ID to retrieve
        """
        try:
            # print(f"\n📄 Parent Retrieval: Loading parent document with ID '{parent_id}'")
            parent = self.parent_store_manager.load_content(parent_id)
            if not parent:
                # print("No parent document found")
                return "NO_PARENT_DOCUMENT"

            parent_content = (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )
            
            # print(f"\n📖 Retrieved parent document:")
            # print("-" * 80)
            # print(parent_content)
            # print("-" * 80)
            
            return parent_content          

        except Exception as e:
            print(f"Parent retrieval error: {str(e)}")
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def create_tools(self) -> List:
        """Create and return the list of tools."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        
        return [search_tool, retrieve_tool]