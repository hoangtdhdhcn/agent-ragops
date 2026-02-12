import uuid
from langchain_groq import ChatGroq     #https://console.groq.com/docs/deprecations #https://console.groq.com/docs/rate-limits
import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChuncker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from eval.evaluation_manager import EvaluationManager
from eval.latency import latency_tracker, measure_document_ingestion, measure_vector_retrieval, measure_llm_generation, measure_agent_execution
import os
from dotenv import load_dotenv

load_dotenv()

# Set Groq API key here
os.getenv("GROQ_API_KEY")

class RAGSystem:
    
    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChuncker()
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        # Pass self to evaluation manager so it can access RAG system for real retrieval
        self.evaluation_manager = EvaluationManager(rag_system=self)
        
    def initialize(self):
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)

        # llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
        # llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=config.LLM_TEMPERATURE)
        llm = ChatGroq(model="moonshotai/kimi-k2-instruct", temperature=config.LLM_TEMPERATURE)
        tools = ToolFactory(collection).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)
        
    def get_config(self):
        return {"configurable": {"thread_id": self.thread_id}}
    
    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())

    def query(self, question: str, expected_retrieval_count: int = 3):
        """
        Process a query through the RAG system with latency tracking.
        
        Args:
            question: The user's question
            expected_retrieval_count: Expected number of documents to retrieve
            
        Returns:
            The response from the RAG system
        """
        with latency_tracker.measure_query(question, expected_retrieval_count):
            # Measure agent graph execution using context manager instead of decorator
            with latency_tracker.measure_operation('agent_graph_execution', {
                'question_length': len(question),
                'expected_retrieval_count': expected_retrieval_count
            }):
                config = self.get_config()
                response = self.agent_graph.invoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config
                )
            
            # Extract the final answer from the response
            # The response structure may vary, so handle different formats
            final_answer = ""
            if isinstance(response, dict):
                messages = response.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, 'content'):
                        final_answer = last_message.content
                    elif isinstance(last_message, dict) and 'content' in last_message:
                        final_answer = last_message['content']
            elif hasattr(response, 'content'):
                final_answer = response.content
            elif isinstance(response, str):
                final_answer = response
            
            # Evaluate the response with real retrieval
            evaluation_results = self.evaluation_manager.evaluate_response_with_real_retrieval(
                question, final_answer
            )
            
            return {
                "answer": final_answer,
                "evaluation": evaluation_results,
                "latency_metrics": latency_tracker.get_latency_stats()
            }
    
    def get_retriever(self):
        if self.retriever is None:
            raise ValueError("Retriever not initialized! Call initialize() first.")
        return self.retriever

