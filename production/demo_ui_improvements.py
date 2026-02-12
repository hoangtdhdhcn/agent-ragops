#!/usr/bin/env python3
"""
Demo script to showcase the UI improvements for the Agentic RAG System.
This creates a simplified version of the UI to demonstrate the enhanced design.
"""

import gradio as gr
from ui.css import custom_css

def create_demo_ui():
    """Create a demo UI showcasing the enhanced design elements."""
    
    def demo_chat_handler(msg, hist):
        """Demo chat handler that simulates responses."""
        import time
        time.sleep(1)  # Simulate processing time
        
        # Simulate a response
        response = f"🤖 I've analyzed your question about '{msg}' and found relevant information in your documents."
        
        # Simulate evaluation results
        return response, [
            ("📈 **Overall Score**: 85%", "✅ Groundedness: PASS", "✅ Relevance: PASS", "✅ Retrieval Accuracy: PASS", 
             "📋 **Evaluation Report**:\n- Groundedness: Response is well-supported by retrieved documents\n- Relevance: Answer directly addresses the user's question\n- Retrieval Accuracy: Relevant documents were successfully retrieved")
        ]

    def demo_upload_handler(files):
        """Demo file upload handler."""
        if files:
            return f"✅ Successfully processed {len(files)} file(s)"
        return "No files uploaded"

    with gr.Blocks(title="Agentic RAG - UI Demo", css=custom_css) as demo:
        
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
                value="📭 No documents available in the knowledge base",
                interactive=False,
                lines=8,
                max_lines=12,
                elem_id="file-list-box",
                show_label=False,
                elem_classes="file-list-container"
            )
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", size="md", elem_classes="secondary-btn")
            
            # Demo event handlers
            add_btn.click(
                demo_upload_handler, 
                [files_input], 
                [file_list], 
                show_progress="full"
            )
            refresh_btn.click(lambda: "📭 No documents available in the knowledge base", None, file_list)
            clear_btn.click(lambda: "🗑️ Removed all documents", None, file_list)
        
        with gr.Tab("💬 Chat"):
            with gr.Row(equal_height=True):
                # Main chat area
                with gr.Column(scale=3, elem_classes="chat-main-column"):
                    gr.Markdown("### 💭 Interactive Chat")
                    gr.Markdown("Ask me anything about your documents!")
                    
                    chatbot = gr.Chatbot(
                        height=600, 
                        placeholder="Type your question here...",
                        show_label=False,
                        elem_classes="chatbot-container"
                    )
                    
                    # Demo chat interface
                    chat_interface = gr.ChatInterface(
                        fn=demo_chat_handler, 
                        chatbot=chatbot,
                        additional_inputs=[],
                        additional_inputs_accordion=None,
                        examples=[
                            "What are the main topics covered in the documents?",
                            "Summarize the key findings from the uploaded files.",
                            "Find information about machine learning."
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
    
    return demo

def main():
    """Main function to launch the demo UI."""
    print("🚀 Launching Agentic RAG UI Demo...")
    print("✨ Enhanced Features:")
    print("   • Modern gradient backgrounds and animations")
    print("   • Enhanced button styling with hover effects")
    print("   • Improved file upload area with drag-and-drop")
    print("   • Real-time evaluation metrics display")
    print("   • Responsive design for mobile devices")
    print("   • Better typography and visual hierarchy")
    print("   • Loading states and feedback animations")
    print("\n🌐 Opening UI in browser...")
    
    demo = create_demo_ui()
    
    # Launch with the enhanced CSS and theme
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        inbrowser=True,
        favicon_path=None,
        ssl_verify=False
    )

if __name__ == "__main__":
    main()