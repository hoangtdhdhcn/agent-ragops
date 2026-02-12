import json
import logging
import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Union
from pydantic import BaseModel, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------------
# Gemini API key
# -------------------------------
os.environ.get("GEMINI_API_KEY")

# Import real Document class from langchain
try:
    from langchain_core.documents import Document
    logger.info("Using real langchain Document class")
except ImportError:
    # Fallback to fake Document class if langchain is not available
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}
    logger.warning("Using fallback Document class - langchain_core not available")

# -------------------------------
# Grader Output Schema
# -------------------------------
class GraderOutput(BaseModel):
    explanation: str = ""
    result: bool

# -------------------------------
# Real LLM evaluator functions (uncomment to use real LLM)
# -------------------------------
def create_real_llm_evaluator(prompt_template: str):
    """Create a real LLM evaluator function."""
    def real_evaluator(inputs, outputs):
        try:
            # Import LLM here to avoid circular imports
            from core.rag_system import RAGSystem
            import config

            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1,
            )
            
            # Format the prompt with inputs and outputs
            prompt = prompt_template.format(
                question=inputs.get("question", ""),
                answer=outputs.get("answer", ""),
                documents="\n\n".join([
                    doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    for doc in outputs.get("documents", [])
                ])
            )
            
            # Call the LLM
            response = llm.invoke(prompt)
            
            # Parse the response
            return response.content
            
        except ImportError:
            logger.error("Cannot create real LLM evaluator - RAG system not available")
            return json.dumps({"explanation": "LLM evaluation failed", "result": False})
        except Exception as e:
            logger.error(f"Real LLM evaluation failed: {e}")
            return json.dumps({"explanation": f"LLM error: {e}", "result": False})
    
    return real_evaluator

# Real evaluator prompt templates
GROUNDENESS_PROMPT = """
    Evaluate if the answer is grounded in the provided documents.

    Question: {question}
    Answer: {answer}
    Documents: {documents}

    Return JSON format: {{"explanation": "short explanation", "result": true/false}}
"""

RELEVANCE_PROMPT = """
    Evaluate if the answer is relevant to the question.

    Question: {question}
    Answer: {answer}

    Return JSON format: {{"explanation": "short explanation", "result": true/false}}
"""

RETRIEVAL_ACCURACY_PROMPT = """
    Evaluate if the retrieved documents are relevant to the question.

    Question: {question}
    Documents: {documents}

    Return JSON format: {{"explanation": "short explanation", "result": true/false}}
"""

def use_real_evaluators():
    """Switch to real LLM evaluators."""
    logger.info("🔄 Switching to REAL LLM evaluators")
    global raw_groundedness, raw_relevance, raw_retrieval_relevance
    
    raw_groundedness = create_real_llm_evaluator(GROUNDENESS_PROMPT)
    raw_relevance = create_real_llm_evaluator(RELEVANCE_PROMPT)
    raw_retrieval_relevance = create_real_llm_evaluator(RETRIEVAL_ACCURACY_PROMPT)
    
    logger.info("✅ Real LLM evaluators activated")

# Automatically switch to real evaluators on import
logger.info("🔄 Automatically switching to REAL LLM evaluators")
raw_groundedness = create_real_llm_evaluator(GROUNDENESS_PROMPT)
raw_relevance = create_real_llm_evaluator(RELEVANCE_PROMPT)
raw_retrieval_relevance = create_real_llm_evaluator(RETRIEVAL_ACCURACY_PROMPT)
logger.info("✅ Real LLM evaluators activated automatically")

def check_evaluation_status():
    """Check if evaluation is using real LLM or fake data."""
    status = {
        "is_real": False,
        "warnings": [],
        "recommendations": []
    }
    
    # Check if we're using fake evaluators
    if "simulated" in raw_groundedness.__name__ or "fake" in raw_groundedness.__name__:
        status["warnings"].append("Using FAKE groundedness evaluator (simulated)")
        status["recommendations"].append("Call use_real_evaluators() to switch to real LLM")
    
    if "simulated" in raw_relevance.__name__ or "fake" in raw_relevance.__name__:
        status["warnings"].append("Using FAKE relevance evaluator (simulated)")
        status["recommendations"].append("Call use_real_evaluators() to switch to real LLM")
    
    if "simulated" in raw_retrieval_relevance.__name__ or "fake" in raw_retrieval_relevance.__name__:
        status["warnings"].append("Using FAKE retrieval accuracy evaluator (simulated)")
        status["recommendations"].append("Call use_real_evaluators() to switch to real LLM")
    
    # Check if we have real retrieval
    if hasattr(status, 'rag_system') and status['rag_system']:
        status["warnings"].append("✅ Real chunk retrieval is working")
    else:
        status["warnings"].append("Real chunk retrieval may not be connected")
        status["recommendations"].append("Ensure RAG system is properly initialized")
    
    # If no warnings about fake evaluators, it's real
    fake_evaluator_warnings = [w for w in status["warnings"] if "FAKE" in w]
    if not fake_evaluator_warnings:
        status["is_real"] = True
    
    return status

# -------------------------------
# Parse grader output robustly
# -------------------------------
def parse_grader_output(raw_response: Union[str, dict]) -> GraderOutput:
    try:
        if isinstance(raw_response, str):
            raw_response = raw_response.strip()
            if raw_response.startswith("```") and raw_response.endswith("```"):
                raw_response = "\n".join(raw_response.split("\n")[1:-1])
            data = json.loads(raw_response)
        elif isinstance(raw_response, dict):
            data = raw_response
        else:
            raise ValueError(f"Unexpected type: {type(raw_response)}")
        return GraderOutput(**data)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        logger.error(f"Failed to parse grader output: {e}. Raw: {raw_response}")
        return GraderOutput(result=False, explanation=f"Parsing failed: {e}")

# -------------------------------
# Wrapped evaluators with logging
# -------------------------------
def wrapped_groundedness(inputs, outputs) -> bool:
    raw = raw_groundedness(inputs, outputs)
    logger.info(f"🔍 Groundedness Grader Raw Output: {raw}")
    parsed = parse_grader_output(raw)
    logger.info(f"✅ Groundedness Grader Parsed: {parsed.explanation} | Result: {parsed.result}")
    return parsed.result

def wrapped_relevance(inputs, outputs) -> bool:
    raw = raw_relevance(inputs, outputs)
    logger.info(f"🔍 Relevance Grader Raw Output: {raw}")
    parsed = parse_grader_output(raw)
    logger.info(f"✅ Relevance Grader Parsed: {parsed.explanation} | Result: {parsed.result}")
    return parsed.result

def wrapped_retrieval_accuracy(inputs, outputs) -> bool:
    raw = raw_retrieval_relevance(inputs, outputs)
    logger.info(f"🔍 Retrieval Accuracy Grader Raw Output: {raw}")
    parsed = parse_grader_output(raw)
    logger.info(f"✅ Retrieval Accuracy Grader Parsed: {parsed.explanation} | Result: {parsed.result}")
    return parsed.result

# -------------------------------
# Utility to handle documents safely
# -------------------------------
def safe_join_documents(documents: List[Union[Document, str]]) -> str:
    pieces = []
    for doc in documents:
        if isinstance(doc, Document):
            content = doc.page_content
        elif isinstance(doc, str):
            content = doc
        else:
            logger.warning(f"Skipping invalid document type: {type(doc)}")
            continue
        # Optional truncation for LLM safety
        if len(content) > 2000:
            content = content[:2000] + "..."
        pieces.append(content)
    return "\n\n".join(pieces)

# -------------------------------
# Evaluation Manager
# -------------------------------
class EvaluationManager:
    def __init__(self, csv_file_path: str = "report/evaluation_results.csv", rag_system=None):
        self.metrics = {
            'groundedness': wrapped_groundedness,
            'relevance': wrapped_relevance,
            'retrieval_accuracy': wrapped_retrieval_accuracy
        }
        self.csv_file_path = csv_file_path
        self.rag_system = rag_system  # Store reference to RAG system for real retrieval
        self._ensure_csv_header()
        self.latest_results = None
        self.latest_retrieved_chunks = []  # Store real retrieved chunks

    def _ensure_csv_header(self):
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'question', 'answer', 'documents', 
                    'groundedness_score', 'groundedness_status',
                    'relevance_score', 'relevance_status', 
                    'retrieval_accuracy_score', 'retrieval_accuracy_status',
                    'overall_score'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    def save_to_csv(self, question, answer, documents, evaluation_results, overall_score):
        try:
            doc_text = safe_join_documents(documents)
            with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'question', 'answer', 'documents', 
                    'groundedness_score', 'groundedness_status',
                    'relevance_score', 'relevance_status', 
                    'retrieval_accuracy_score', 'retrieval_accuracy_status',
                    'overall_score'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                row_data = {
                    'timestamp': datetime.now().isoformat(),
                    'question': question,
                    'answer': answer,
                    'documents': doc_text,
                    'groundedness_score': evaluation_results.get('groundedness', {}).get('score', False),
                    'groundedness_status': evaluation_results.get('groundedness', {}).get('status', 'UNKNOWN'),
                    'relevance_score': evaluation_results.get('relevance', {}).get('score', False),
                    'relevance_status': evaluation_results.get('relevance', {}).get('status', 'UNKNOWN'),
                    'retrieval_accuracy_score': evaluation_results.get('retrieval_accuracy', {}).get('score', False),
                    'retrieval_accuracy_status': evaluation_results.get('retrieval_accuracy', {}).get('status', 'UNKNOWN'),
                    'overall_score': overall_score
                }
                writer.writerow(row_data)
                logger.info(f"Evaluation results saved to {self.csv_file_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")

    def retrieve_real_chunks(self, question: str, limit: int = 3) -> List[Document]:
        """Retrieve real chunks from the RAG system instead of using fake data."""
        if not self.rag_system or not self.rag_system.agent_graph:
            logger.warning("No RAG system available for real retrieval, using empty list")
            return []
        
        try:
            # Get the collection from the RAG system
            # Use the collection name from the RAG system
            collection_name = getattr(self.rag_system, 'collection_name', None)
            if not collection_name:
                logger.warning("RAG system has no collection_name attribute")
                return []
                
            collection = self.rag_system.vector_db.get_collection(collection_name)
            
            # Perform similarity search to get real chunks
            results = collection.similarity_search(question, k=limit, score_threshold=0.7)
            
            # Convert to Document objects if they aren't already
            real_chunks = []
            for doc in results[:limit]:  # Limit to first 3 chunks as requested
                if isinstance(doc, Document):
                    real_chunks.append(doc)
                else:
                    # Create a Document object from the retrieved result
                    real_chunks.append(Document(
                        page_content=doc.page_content,
                        metadata=doc.metadata
                    ))
            
            self.latest_retrieved_chunks = real_chunks
            logger.info(f"Retrieved {len(real_chunks)} real chunks for evaluation")
            return real_chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve real chunks: {e}")
            return []

    def evaluate_response_with_real_retrieval(self, question: str, answer: str) -> Dict[str, Any]:
        """Evaluate response using real retrieved chunks instead of fake data."""
        # Retrieve real chunks from the RAG system (limit to first 3 as requested)
        real_chunks = self.retrieve_real_chunks(question, limit=3)
        
        # If no real chunks found, fall back to empty list
        if not real_chunks:
            logger.warning("No real chunks retrieved, using empty list for evaluation")
            real_chunks = []
        
        # Use real chunks for evaluation
        return self.evaluate_response(question, answer, real_chunks)

    def evaluate_response(self, question: str, answer: str, documents: List[Union[Document, str]]) -> Dict[str, Any]:
        inputs = {"question": question}
        outputs = {"answer": answer, "documents": documents}

        results = {}
        for metric_name, evaluator in self.metrics.items():
            try:
                score_bool = evaluator(inputs, outputs)
                results[metric_name] = {"score": score_bool, "status": "PASS" if score_bool else "FAIL"}
            except Exception as e:
                results[metric_name] = {"score": False, "status": "ERROR", "error": str(e)}

        overall_score = self.get_overall_score(results)
        self.save_to_csv(question, answer, documents, results, overall_score)
        results['overall_score'] = overall_score
        self.latest_results = results
        return results

    def get_overall_score(self, evaluation_results: Dict[str, Any]) -> float:
        valid_scores = []
        for metric_name, metric_result in evaluation_results.items():
            if metric_name == "overall_score":
                continue
            if isinstance(metric_result, dict):
                if metric_result.get("status") == "PASS":
                    valid_scores.append(1.0)
                elif metric_result.get("status") == "FAIL":
                    valid_scores.append(0.0)
        if not valid_scores:
            return 0.0
        return sum(valid_scores) / len(valid_scores)

    def format_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        report_lines = ["📊 **Evaluation Results**", ""]
        for metric_name, result in evaluation_results.items():
            if metric_name == "overall_score":
                continue
            if isinstance(result, dict):
                status = result.get("status", "UNKNOWN")
                emoji = "✅" if status == "PASS" else "⚠️" if status == "ERROR" else ""
                report_lines.append(f"{emoji} **{metric_name.title()}**: {status}")
                if "error" in result:
                    report_lines.append(f"   Error: {result['error']}")
            else:
                report_lines.append(f"**{metric_name.title()}**: {result}")
        overall_score = evaluation_results.get("overall_score", self.get_overall_score(evaluation_results))
        report_lines.extend(["", f"📈 **Overall Score**: {overall_score:.2%}"])
        return "\n".join(report_lines)


