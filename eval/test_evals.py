# test_evaluation.py
import logging
from evaluation_manager import EvaluationManager
from langchain_core.documents import Document

# Enable logging to see debug messages
logging.basicConfig(level=logging.INFO)

def main():
    # Create EvaluationManager instance
    eval_manager = EvaluationManager(csv_file_path="test_evaluation_results.csv")

    # Fake RAG knowledge documents
    docs1 = [
        Document(page_content="To Xuan Hoang is a machine learning engineer with experience in MLOps."),
        Document(page_content="He has worked on multiple ML projects and research assistance roles.")
    ]
    docs2 = [
        Document(page_content="Python is a popular programming language for data science."),
        Document(page_content="LangChain is a framework for building RAG systems.")
    ]

    # List of (question, answer, documents) tuples
    test_cases = [
        ("Who is To Xuan Hoang?", 
         "To Xuan Hoang is an ML engineer experienced in MLOps and research assistance.", 
         docs1),
        ("What is LangChain?", 
         "LangChain is a framework for building RAG applications in Python.", 
         docs2)
    ]

    # Evaluate each question and print the report
    for i, (question, answer, docs) in enumerate(test_cases, start=1):
        print(f"\n--- Test Case {i} ---")
        results = eval_manager.evaluate_response(question, answer, docs)
        report = eval_manager.format_evaluation_report(results)
        print(report)

        # Also show that latest_results updates
        print("\nLatest Results Dict:")
        print(eval_manager.get_latest_results())

if __name__ == "__main__":
    main()
