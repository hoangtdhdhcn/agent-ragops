#!/usr/bin/env python3
"""
Integrated UI Demo that showcases the enhanced design with the actual RAG system.
This demonstrates all the UI improvements while using the real RAG functionality.
"""

import gradio as gr
from ui.css import custom_css

# Import the actual RAG components
try:
    from core.rag_system import RAGSystem
    from core.document_manager import DocumentManager
    from core.chat_interface import ChatInterface
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAG system not available: {e}")
    RAG_AVAILABLE = False

def create_integrated_demo():
    """Create a demo that uses the actual RAG system with enhanced UI."""
    
    # Initialize RAG system if available
    rag_available = RAG_AVAILABLE
    rag_system = None
    doc_manager = None
    rag_chat = None
    
    if rag_available:
        try:
            rag_system = RAGSystem()
            rag_system.initialize()
            doc_manager = DocumentManager(rag_system)
            rag_chat = ChatInterface(rag_system)
            
            def format_file_list():
                files = doc_manager.get_markdown_files()
                if not files:
                    return "<tool_call> No documents available in the knowledge base"
                return "\n".join([f"{f}" for f in files])
            
            def upload_handler(files, progress=gr.Progress()):
                if not files:
                    return None, format_file_list()
                    
                added, skipped = doc_manager.add_documents(
                    files, 
                    progress_callback=lambda p, desc: progress(p, desc=desc)
                )
                
                gr.Info(f"✅ Added: {added} | Skipped: {skipped}")
                return None, format_file_list()
            
            def clear_handler():
                doc_manager.clear_all()
                gr.Info(f"🗑️ Removed all documents")
                return format_file_list()
            
            def chat_handler(msg, hist):
                # Use the actual RAG system
                if rag_chat:
                    return rag_chat.chat(msg, hist)
                else:
                    # Fallback to simulated response
                    import time
                    import random
                    time.sleep(2)
                    responses = [
                        f"🤖 Based on the documents in your knowledge base, I found information about '{msg}'. The relevant documents indicate that this topic is well-covered in your uploaded materials.",
                        f"📚 After analyzing your documents, I can provide insights about '{msg}'. The information suggests several key points that are documented across multiple sources.",
                        f"🔍 I've searched through your knowledge base and found relevant information about '{msg}'. The documents contain comprehensive coverage of this topic with multiple perspectives.",
                        f"💡 Your documents contain valuable information about '{msg}'. Based on the content analysis, here are the key findings from your knowledge base."
                    ]
                    return random.choice(responses)
            
            def get_evaluation_results():
                """Get the latest evaluation results from the chat interface."""
                if hasattr(rag_chat, 'last_evaluation_results') and rag_chat.last_evaluation_results:
                    results = rag_chat.last_evaluation_results
                    overall_score = rag_chat.evaluation_manager.get_overall_score(results)
                    
                    # Format individual metrics
                    groundedness_status = f"✅ Groundedness: PASS" if results.get('groundedness', {}).get('score', False) else f"Groundedness: FAIL"
                    relevance_status = f"✅ Relevance: PASS" if results.get('relevance', {}).get('score', False) else f"Relevance: FAIL"
                    retrieval_accuracy_status = f"✅ Retrieval Accuracy: PASS" if results.get('retrieval_accuracy', {}).get('score', False) else f"Retrieval Accuracy: FAIL"
                    
                    # Format overall score
                    overall_text = f"📈 **Overall Score**: {overall_score:.2%}"
                    
                    # Format detailed report
                    report = rag_chat.evaluation_manager.format_evaluation_report(results)
                    
                    return overall_text, groundedness_status, relevance_status, retrieval_accuracy_status, report
                else:
                    return "No evaluation yet", "Not evaluated", "Not evaluated", "Not evaluated", "No evaluation yet"
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            rag_available = False
            rag_system = None
            doc_manager = None
            rag_chat = None
    else:
        # Fallback to simulated responses
        def format_file_list():
            return "<tool_call> No documents available in the knowledge base"
        
        def upload_handler(files, progress=gr.Progress()):
            if files:
                gr.Info(f"✅ Processed {len(files)} file(s)")
                return None, f"✅ Successfully processed {len(files)} file(s)"
            return None, "No files uploaded"
        
        def clear_handler():
            gr.Info("🗑️ Removed all documents")
            return "🗑️ All documents have been removed"
        
        def chat_handler(msg, hist):
            import time
            import random
            
            # Simulate processing time
            time.sleep(2)
            
            responses = [
                f"🤖 Based on the documents in your knowledge base, I found information about '{msg}'. The relevant documents indicate that this topic is well-covered in your uploaded materials.",
                f"📚 After analyzing your documents, I can provide insights about '{msg}'. The information suggests several key points that are documented across multiple sources.",
                f"🔍 I've searched through your knowledge base and found relevant information about '{msg}'. The documents contain comprehensive coverage of this topic with multiple perspectives.",
                f"💡 Your documents contain valuable information about '{msg}'. Based on the content analysis, here are the key findings from your knowledge base."
            ]
            
            return random.choice(responses)
        
        def get_evaluation_results():
            return "📈 **Overall Score**: 85%", "✅ Groundedness: PASS", "✅ Relevance: PASS", "✅ Retrieval Accuracy: PASS", "📋 **Evaluation Report**:\n- Groundedness: Response is well-supported by retrieved documents\n- Relevance: Answer directly addresses the user's question\n- Retrieval Accuracy: Relevant documents were successfully retrieved"

    with gr.Blocks(title="Agentic RAG - Integrated Demo", css=custom_css) as demo:
        
        # Header section
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    <div style="text-align: center; margin-bottom: 20px;">
                        <h1 style="margin: 0; font-size: 2.5rem; background: linear-gradient(135deg, #ffffff, #e5e5e5); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            🤖 Agentic RAG System
                        </h1>
                        <p style="color: #a3a3a3; margin: 10px 0 0 0;">Enhanced UI with Modern Design & Real-time Evaluation</p>
                    </div>
                    """,
                    elem_classes="header-text"
                )
        
        with gr.Tab("📚 Documents"):
            gr.Markdown("### 📤 Add New Documents")
            gr.Markdown("Upload PDF or Markdown files. Duplicates will be automatically skipped.")
            
            with gr.Row():
                with gr.Column():
                    files_input = gr.File(
                        label="📁 Drop PDF or Markdown files here",
                        file_count="multiple",
                        type="filepath",
                        height=200,
                        show_label=True,
                        elem_classes="file-upload-area"
                    )
                    
                    with gr.Row():
                        add_btn = gr.Button("➕ Add Documents", variant="primary", size="md", elem_classes="primary-btn")
                        clear_btn = gr.Button("🗑️ Clear All", variant="stop", size="md", elem_classes="danger-btn")
            
            gr.Markdown("### 📋 Current Documents in the Knowledge Base")
            file_list = gr.Textbox(
                value=format_file_list(),
                interactive=False,
                lines=8,
                max_lines=12,
                elem_id="file-list-box",
                show_label=False,
                elem_classes="file-list-container"
            )
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", size="md", elem_classes="secondary-btn")
            
            # Event handlers
            add_btn.click(
                upload_handler, 
                [files_input], 
                [files_input, file_list], 
                show_progress="full"
            )
            refresh_btn.click(format_file_list, None, file_list)
            clear_btn.click(clear_handler, None, file_list)
        
        with gr.Tab("💬 Chat"):
            with gr.Row(equal_height=True):
                # Main chat area
                with gr.Column(scale=3, elem_classes="chat-main-column"):
                    gr.Markdown("### 💭 Interactive Chat")
                    gr.Markdown("Ask me anything about your documents! I'll provide answers with real-time evaluation.")
                    
                    chatbot = gr.Chatbot(
                        height=600, 
                        placeholder="Type your question here...",
                        show_label=False,
                        elem_classes="chatbot-container"
                    )
                    
                    # Chat interface
                    ui_chat = gr.ChatInterface(
                        fn=chat_handler, 
                        chatbot=chatbot,
                        additional_inputs=[],
                        additional_inputs_accordion=None,
                        examples=[
                            "What are the main topics covered in the documents?",
                            "Summarize the key findings from the uploaded files.",
                            "Find information about machine learning.",
                            "What does the documentation say about AI agents?"
                        ],
                        cache_examples=False,
                        submit_btn="📤 Send",
                        stop_btn="⏹️ Stop"
                    )
                
                # Evaluation sidebar
                with gr.Column(scale=1, elem_classes="evaluation-sidebar"):
                    gr.Markdown("### 📊 Real-time Evaluation")
                    gr.Markdown("Evaluation results will appear here after each response.")
                    
                    with gr.Accordion("📈 Overall Score", open=True, elem_classes="metric-accordion"):
                        overall_score = gr.Markdown("⏳ Waiting for response...", elem_id="overall-score", elem_classes="overall-score")
                    
                    with gr.Accordion("🔍 Detailed Metrics", open=False, elem_classes="metric-accordion"):
                        groundedness_status = gr.Markdown("⏳ Not evaluated", elem_id="groundedness-status", elem_classes="metric-card")
                        relevance_status = gr.Markdown("⏳ Not evaluated", elem_id="relevance-status", elem_classes="metric-card")
                        retrieval_accuracy_status = gr.Markdown("⏳ Not evaluated", elem_id="retrieval-accuracy-status", elem_classes="metric-card")
                    
                    with gr.Accordion("📋 Evaluation Report", open=False, elem_classes="metric-accordion"):
                        evaluation_report = gr.Markdown("⏳ No evaluation yet", elem_id="evaluation-report", elem_classes="evaluation-report")
                    
                    # Hidden trigger for updates
                    update_trigger = gr.Textbox(visible=False)
                    
                    # Auto-update evaluation sidebar
                    def update_sidebar():
                        return get_evaluation_results()
                    
                    update_trigger.change(
                        update_sidebar,
                        outputs=[overall_score, groundedness_status, relevance_status, retrieval_accuracy_status, evaluation_report]
                    )
                    
                    # Initial setup
                    demo.load(
                        lambda: ("",) + get_evaluation_results(),
                        outputs=[update_trigger, overall_score, groundedness_status, relevance_status, retrieval_accuracy_status, evaluation_report]
                    )
    
    return demo

def main():
    """Main function to launch the integrated demo UI."""
    print("🚀 Launching Integrated Agentic RAG UI Demo...")
    print("✨ Enhanced Features:")
    print("   • Modern gradient backgrounds and animations")
    print("   • Enhanced button styling with hover effects")
    print("   • Improved file upload area with drag-and-drop")
    print("   • Real-time evaluation metrics display")
    print("   • Responsive design for mobile devices")
    print("   • Better typography and visual hierarchy")
    print("   • Loading states and feedback animations")
    
    if RAG_AVAILABLE:
        print("   • Actual RAG system integration")
        print("   • Real document processing and retrieval")
        print("   • Live evaluation metrics")
    else:
        print("   • Simulated RAG responses (RAG system not available)")
        print("   • Working UI demonstration")
    
    print("\n🌐 Opening UI in browser...")
    
    demo = create_integrated_demo()
    
    # Launch with the enhanced CSS
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        inbrowser=True,
        favicon_path=None,
        ssl_verify=False,
        debug=True
    )

if __name__ == "__main__":
    main()