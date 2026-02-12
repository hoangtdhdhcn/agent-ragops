from langchain_core.messages import HumanMessage

class ChatInterface:
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.last_evaluation_results = None
        self.last_question = None
        self.last_answer = None
        
    def chat(self, message, history):
        if not self.rag_system.agent_graph:
            return "⚠️ System not initialized!"
            
        try:
            result = self.rag_system.agent_graph.invoke(
                {"messages": [HumanMessage(content=message.strip())]},
                self.rag_system.get_config()
            )
            
            response = result["messages"][-1].content
            
            # Store the question and answer for potential evaluation
            self.last_question = message.strip()
            self.last_answer = response
            
            # Capture retrieved chunks from the RAG system
            self._capture_retrieved_chunks(message.strip())
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _capture_retrieved_chunks(self, query: str):
        """Capture retrieved chunks from the RAG system tools."""
        try:
            # Access the collection from the RAG system
            if hasattr(self.rag_system, 'vector_db') and hasattr(self.rag_system, 'collection_name'):
                collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
                
                # Create a temporary tool factory to perform similarity search
                from rag_agent.tools import ToolFactory
                tool_factory = ToolFactory(collection)
                
                # Perform similarity search to get real chunks
                raw_chunks, formatted_chunks = tool_factory._search_child_chunks(query, limit=3)
                
                # Store the raw Document objects for UI display
                self.store_retrieved_chunks(raw_chunks)
                
                print(f"✅ Captured {len(raw_chunks)} chunks for UI display")
            else:
                print("⚠️ Cannot access RAG system collection for chunk retrieval")
                self.store_retrieved_chunks([])
                
        except Exception as e:
            print(f"Failed to capture retrieved chunks: {e}")
            self.store_retrieved_chunks([])
    
    def evaluate_last_response(self, retrieved_documents=None):
        """Evaluate the last chat response using real retrieved chunks if evaluation manager is available."""
        if not hasattr(self.rag_system, 'evaluation_manager'):
            return False
            
        if not self.last_question or not self.last_answer:
            return False
            
        try:
            # Use the new evaluation method that retrieves real chunks from RAG system
            if hasattr(self.rag_system.evaluation_manager, 'evaluate_response_with_real_retrieval'):
                # Use real retrieval from RAG system (limit to first 3 chunks)
                results = self.rag_system.evaluation_manager.evaluate_response_with_real_retrieval(
                    self.last_question, 
                    self.last_answer
                )
            else:
                # Fallback to old method if real retrieval is not available
                documents = retrieved_documents or []
                results = self.rag_system.evaluation_manager.evaluate_response(
                    self.last_question, 
                    self.last_answer, 
                    documents
                )
            
            self.last_evaluation_results = results
            return True
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return False
        
    def store_retrieved_chunks(self, chunks):
        """Store the actual Document objects for UI display."""
        self.last_retrieved_chunks = chunks
    
    def clear_session(self):
        self.rag_system.reset_thread()
        self.last_evaluation_results = None
        self.last_question = None
        self.last_answer = None
