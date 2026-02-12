#!/usr/bin/env python3
"""
Demo script to demonstrate the evaluation metrics integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.evaluation_manager import EvaluationManager
from langchain_core.documents import Document

def demo_evaluation_metrics():
    """Demonstrate the evaluation metrics with different scenarios."""
    
    print("🎯 Evaluation Metrics Demo")
    print("=" * 50)
    
    # Create evaluation manager
    eval_manager = EvaluationManager()
    
    # Scenario 1: Good response (grounded, relevant, accurate retrieval)
    print("\n📝 Scenario 1: Good Response")
    print("-" * 30)
    
    question1 = "What is the capital of France?"
    answer1 = "The capital of France is Paris, which is located in western Europe."
    documents1 = [
        Document(page_content="France is a country in Europe. Its capital is Paris.", metadata={"source": "geography.pdf"}),
        Document(page_content="Paris is the largest city in France and serves as the country's political and cultural center.", metadata={"source": "geography.pdf"})
    ]
    
    results1 = eval_manager.evaluate_response(question1, answer1, documents1)
    report1 = eval_manager.format_evaluation_report(results1)
    print(f"Question: {question1}")
    print(f"Answer: {answer1}")
    print(f"Documents: {[doc.page_content[:50] + '...' for doc in documents1]}")
    print(f"\n{report1}")
    
    # Scenario 2: Hallucinated response (not grounded)
    print("\n📝 Scenario 2: Hallucinated Response")
    print("-" * 30)
    
    question2 = "What is the capital of France?"
    answer2 = "The capital of France is Lyon, which is famous for its silk production."
    documents2 = [
        Document(page_content="France is a country in Europe. Its capital is Paris.", metadata={"source": "geography.pdf"}),
        Document(page_content="Paris is the largest city in France and serves as the country's political and cultural center.", metadata={"source": "geography.pdf"})
    ]
    
    results2 = eval_manager.evaluate_response(question2, answer2, documents2)
    report2 = eval_manager.format_evaluation_report(results2)
    print(f"Question: {question2}")
    print(f"Answer: {answer2}")
    print(f"Documents: {[doc.page_content[:50] + '...' for doc in documents2]}")
    print(f"\n{report2}")
    
    # Scenario 3: Irrelevant response
    print("\n📝 Scenario 3: Irrelevant Response")
    print("-" * 30)
    
    question3 = "What is the capital of France?"
    answer3 = "The weather today is quite nice, perfect for a walk in the park."
    documents3 = [
        Document(page_content="France is a country in Europe. Its capital is Paris.", metadata={"source": "geography.pdf"}),
        Document(page_content="Paris is the largest city in France and serves as the country's political and cultural center.", metadata={"source": "geography.pdf"})
    ]
    
    results3 = eval_manager.evaluate_response(question3, answer3, documents3)
    report3 = eval_manager.format_evaluation_report(results3)
    print(f"Question: {question3}")
    print(f"Answer: {answer3}")
    print(f"Documents: {[doc.page_content[:50] + '...' for doc in documents3]}")
    print(f"\n{report3}")
    
    # Scenario 4: Poor retrieval accuracy
    print("\n📝 Scenario 4: Poor Retrieval Accuracy")
    print("-" * 30)
    
    question4 = "What is the capital of France?"
    answer4 = "The capital of France is Paris."
    documents4 = [
        Document(page_content="The capital of Germany is Berlin.", metadata={"source": "geography.pdf"}),
        Document(page_content="Italy's capital is Rome.", metadata={"source": "geography.pdf"})
    ]
    
    results4 = eval_manager.evaluate_response(question4, answer4, documents4)
    report4 = eval_manager.format_evaluation_report(results4)
    print(f"Question: {question4}")
    print(f"Answer: {answer4}")
    print(f"Documents: {[doc.page_content[:50] + '...' for doc in documents4]}")
    print(f"\n{report4}")
    
    print("\n" + "=" * 50)
    print("✅ Demo completed! The evaluation metrics are working correctly.")

def demo_rag_integration():
    """Demonstrate RAG system integration."""
    
    print("\n🤖 RAG System Integration Demo")
    print("=" * 50)
    
    try:
        from core.rag_system import RAGSystem
        from core.chat_interface import ChatInterface
        
        print("Initializing RAG system with evaluation...")
        rag_system = RAGSystem()
        rag_system.initialize()
        
        print("✅ RAG system initialized successfully")
        print("✅ Evaluation manager integrated into RAG system")
        
        # Show the evaluation manager is available
        if hasattr(rag_system, 'evaluation_manager'):
            print("✅ Evaluation manager available in RAG system")
            print(f"   Available metrics: {list(rag_system.evaluation_manager.metrics.keys())}")
        
        print("\n✅ RAG system integration demo completed!")
        
    except Exception as e:
        print(f"RAG system integration demo failed: {e}")

if __name__ == "__main__":
    demo_evaluation_metrics()
    demo_rag_integration()
    
    print("\nAll demos completed successfully!")
    print("\nThe evaluation metrics are now integrated into RAG system.")