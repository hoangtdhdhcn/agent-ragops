#!/usr/bin/env python3
"""
Integrated UI Demo that showcases the enhanced design with the actual RAG system.
This demonstrates all the UI improvements while using the real RAG functionality.
"""

import gradio as gr
import time
import json
import os
import sys
from ui.css import custom_css

# ---------------------------------------------------------------------
# Optional RAG imports
# ---------------------------------------------------------------------
try:
    from core.rag_system import RAGSystem
    from core.document_manager import DocumentManager
    from core.chat_interface import ChatInterface
    from eval.latency import latency_tracker

    RAG_IMPORT_OK = True
except ImportError as e:
    print(f"Warning: RAG imports failed: {e}")
    RAG_IMPORT_OK = False


def create_integrated_demo():
    """Create a demo that uses the actual RAG system with enhanced UI."""

    rag_system = None
    doc_manager = None
    rag_chat = None
    rag_ready = False

    # ------------------------------------------------------------------
    # SAFE DEFAULT HANDLERS (ALWAYS DEFINED)
    # ------------------------------------------------------------------

    def format_file_list():
        return "⚠️ RAG system not available"

    def upload_handler(files, progress=gr.Progress()):
        gr.Warning("RAG system not available")
        return format_file_list()

    def clear_handler():
        gr.Warning("RAG system not available")
        return format_file_list()

    def chat_handler(msg, hist):
        return "⚠️ RAG system not available", "update"

    def get_evaluation_results():
        return (
            "⚠️ RAG system not available",
            "⚠️",
            "⚠️",
            "⚠️",
            "⚠️",
        )

    # ------------------------------------------------------------------
    # ATTEMPT RAG INITIALIZATION
    # ------------------------------------------------------------------

    if RAG_IMPORT_OK:
        try:
            rag_system = RAGSystem()
            rag_system.initialize()  # ← WILL FAIL if LangGraph node signatures are wrong

            doc_manager = DocumentManager(rag_system)
            rag_chat = ChatInterface(rag_system)
            rag_ready = True

            # -------------------- REAL HANDLERS --------------------

            def format_file_list():
                files = doc_manager.get_markdown_files()
                if not files:
                    return "<tool_call> No documents available in the knowledge base"
                return "\n".join(files)

            def upload_handler(files, progress=gr.Progress()):
                if not files:
                    return format_file_list()

                # Measure document processing latency
                with latency_tracker.measure_operation('document_ingestion', {
                    'file_count': len(files),
                    'file_types': [f.split('.')[-1] for f in [os.path.basename(f) for f in files]]
                }):
                    added, skipped = doc_manager.add_documents(
                        files,
                        progress_callback=lambda p, desc: progress(p, desc=desc),
                    )

                gr.Info(f"✅ Added: {added} | Skipped: {skipped}")
                return format_file_list()

            def clear_handler():
                doc_manager.clear_all()
                gr.Info("🗑️ Removed all documents")
                return format_file_list()

            def chat_handler(msg, hist):
                # Measure the complete query processing with latency tracking
                with latency_tracker.measure_query(msg, 3):
                    response = rag_chat.chat(msg, hist)
                    # Trigger evaluation of the response
                    rag_chat.evaluate_last_response()
                return response, "update"

            def get_evaluation_results():
                if (
                    rag_chat.last_evaluation_results
                    and hasattr(rag_system, 'evaluation_manager')
                ):
                    results = rag_chat.last_evaluation_results
                    overall_score = rag_system.evaluation_manager.get_overall_score(results)

                    groundedness = (
                        "✅ Groundedness: PASS"
                        if results.get("groundedness", {}).get("score")
                        else "Groundedness: FAIL"
                    )
                    relevance = (
                        "✅ Relevance: PASS"
                        if results.get("relevance", {}).get("score")
                        else "Relevance: FAIL"
                    )
                    retrieval = (
                        "✅ Retrieval Accuracy: PASS"
                        if results.get("retrieval_accuracy", {}).get("score")
                        else "Retrieval Accuracy: FAIL"
                    )

                    # report = rag_system.evaluation_manager.format_evaluation_report(results)

                    return (
                        f"📈 **Overall Score**: {overall_score:.2%}",
                        groundedness,
                        relevance,
                        retrieval,
                        # report,
                    )

                return (
                    "⏳ No evaluation yet",
                    "⏳",
                    "⏳",
                    "⏳",
                    "⏳",
                )

            def get_retrieved_chunks():
                """Get the first 3 retrieved chunks for display in a compact format."""
                if hasattr(rag_chat, 'last_retrieved_chunks') and rag_chat.last_retrieved_chunks:
                    chunks = rag_chat.last_retrieved_chunks[:3]  # Get first 3 chunks
                    if chunks:
                        # Smart summary format - compact and scannable
                        chunk_text = f"**Found {len(chunks)} relevant chunks**\n\n"
                        
                        # Add detailed view in collapsible section
                        # chunk_text += "---\n\n<details><summary>🔍 View Full Content</summary>\n\n"
                        for i, chunk in enumerate(chunks, 1):
                            chunk_text += f"### 📄 Chunk {i}\n\n"
                            chunk_text += f"**Source**: {getattr(chunk, 'metadata', {}).get('source', 'Unknown')}\n\n"
                            chunk_text += f"**Content**:\n```\n{chunk.page_content[:500]}{'...' if len(chunk.page_content) > 500 else ''}\n```\n\n"
                            chunk_text += "---\n\n"
                        chunk_text += "</details>"
                        
                        return chunk_text
                    else:
                        return "<tool_call> No chunks retrieved"
                else:
                    return "⏳ Waiting for retrieval..."

        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            rag_ready = False

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    with gr.Blocks(title="Integrated Demo") as demo:

        gr.Markdown(
            """
            <div style="text-align:center; margin-bottom:20px;">
                <h1>🤖 Helpful Document Question Answering</h1>
                <p>Enhanced UI with Real-time Evaluation</p>
            </div>
            """
        )

        # -------------------- Documents Tab --------------------
        with gr.Tab("📚 Documents"):
            with gr.Row():
                with gr.Column(scale=1):
                    files_input = gr.File(
                        file_count="multiple",
                        type="filepath",
                        label="Upload PDF / Markdown files",
                    )

                    with gr.Row():
                        add_btn = gr.Button("➕ Add Documents")
                        clear_btn = gr.Button("🗑️ Clear All")

                with gr.Column(scale=1):
                    file_list = gr.Textbox(
                        value="",
                        interactive=False,
                        lines=8,
                    )

                    add_btn.click(
                        fn=upload_handler,
                        inputs=files_input,
                        outputs=file_list,   # ✅ File is INPUT ONLY
                    )

                    clear_btn.click(
                        fn=clear_handler,
                        outputs=file_list,
                    )

                    demo.load(
                        fn=format_file_list,
                        outputs=file_list,
                    )

        # -------------------- Chat Tab --------------------
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=400)
                    update_trigger = gr.Textbox(visible=False)

                    gr.ChatInterface(
                        fn=chat_handler,
                        chatbot=chatbot,
                        additional_outputs=[update_trigger],
                        submit_btn="📤 Send",
                        stop_btn="⏹️ Stop",
                    )

                with gr.Column(scale=1):
                    with gr.Row():
                        evaluation_toggle = gr.Checkbox(
                            value=True,
                            label="📊 Show Evaluation Metrics",
                            interactive=True
                        )
                    
                    # Evaluation metrics section (toggleable)
                    with gr.Column(visible=True) as evaluation_metrics_section:
                        gr.Markdown("### 📊 Evaluation Metrics")

                        groundedness_status = gr.Markdown("⏳")
                        relevance_status = gr.Markdown("⏳")
                        retrieval_accuracy_status = gr.Markdown("⏳")
                        overall_score = gr.Markdown("⏳")
                    
                    # Retrieved chunks section (always visible)
                    with gr.Column(visible=True):
                        gr.Markdown("### 🔍 Retrieved Chunks")
                        
                        # Smart retrieved chunks display - scrollable content
                        with gr.Accordion("📄 Chunks Preview", open=False):
                            with gr.Column(elem_classes="retrieved-chunks-container"):
                                retrieved_chunks_display = gr.Markdown(
                                    value="⏳ Waiting for retrieval...",
                                    elem_classes="retrieved-chunks-content"
                                )

                    def update_all_results():
                        """Update both evaluation results and retrieved chunks."""
                        eval_results = get_evaluation_results()
                        chunk_results = get_retrieved_chunks()
                        return eval_results + (chunk_results,)

                    update_trigger.change(
                        fn=update_all_results,
                        outputs=[
                            groundedness_status,
                            relevance_status,
                            retrieval_accuracy_status,
                            overall_score,
                            retrieved_chunks_display,
                        ],
                    )

                    # Add refresh button for manual updates
                    with gr.Row():
                        refresh_eval_btn = gr.Button("🔄 Refresh Evaluation", variant="secondary", size="sm")

                    refresh_eval_btn.click(
                        fn=update_all_results,
                        outputs=[
                            groundedness_status,
                            relevance_status,
                            retrieval_accuracy_status,
                            overall_score,
                            retrieved_chunks_display,
                        ],
                    )

                    # Toggle evaluation metrics section visibility
                    def toggle_evaluation(show_evaluation):
                        return gr.update(visible=show_evaluation)

                    evaluation_toggle.change(
                        fn=toggle_evaluation,
                        inputs=[evaluation_toggle],
                        outputs=[evaluation_metrics_section]
                    )

        # -------------------- Latency Monitor Tab --------------------
        with gr.Tab("⚡ Latency Monitor"):
            gr.Markdown("### 📊 Real-time Latency Monitoring")
            gr.Markdown("Monitor the performance of RAG system across all pipeline stages. This tab automatically tracks latency from real queries in the Chat tab.")
            
            # Top row: Controls and Key Metrics
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 🎛️ Controls")
                    with gr.Row():
                        export_csv_btn = gr.Button("💾 CSV", variant="primary", size="sm")
                        export_json_btn = gr.Button("📄 JSON", variant="secondary", size="sm")
                    with gr.Row():
                        clear_metrics_btn = gr.Button("🧹 Clear", variant="stop", size="sm")
                        refresh_stats_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")
                    demo_status = gr.Markdown("⏳ Ready to track latency from queries", elem_classes="latency-status")
                
                with gr.Column(scale=3):
                    gr.Markdown("#### 📈 Key Metrics")
                    with gr.Row():
                        with gr.Column():
                            total_measurements = gr.Markdown("📊 **Total**: 0", elem_classes="metric-card-small")
                            avg_latency = gr.Markdown("⏱️ **Avg**: 0.00s", elem_classes="metric-card-small")
                        with gr.Column():
                            operations_tracked = gr.Markdown("📋 **Operations**: 0", elem_classes="metric-card-small")
                            p95_latency = gr.Markdown("📈 **P95**: 0.00s", elem_classes="metric-card-small")
            
            # Middle row: Pipeline Analysis and Recent Metrics
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📊 Pipeline Stage Analysis")
                    pipeline_stats = gr.Dataframe(
                        label="Stage Performance",
                        headers=["Stage", "Count", "Avg (s)", "P95 (s)", "Max (s)"],
                        datatype=["str", "number", "number", "number", "number"],
                        elem_classes="compact-dataframe"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("#### 📋 Recent Measurements")
                    recent_metrics = gr.Dataframe(
                        label="Recent Latency Measurements",
                        headers=["Operation", "Duration (s)", "Time"],
                        datatype=["str", "number", "str"],
                        elem_classes="compact-dataframe"
                    )
            
            # Export functions
            def export_to_csv():
                try:
                    latency_tracker.export_metrics('report/ui_latency_metrics.csv', 'csv')
                    return "✅ Exported to report/ui_latency_metrics.csv"
                except Exception as e:
                    return f" Export failed: {str(e)}"
            
            def export_to_json():
                try:
                    latency_tracker.export_metrics('report/ui_latency_metrics.json', 'json')
                    return "✅ Exported to report/ui_latency_metrics.json"
                except Exception as e:
                    return f" Export failed: {str(e)}"
            
            # Stats update function
            def update_stats():
                try:
                    stats = latency_tracker.get_latency_stats()
                    summary = latency_tracker.get_performance_summary()
                    
                    # Calculate overall stats
                    total_measurements_text = f"📊 **Total Measurements**: {summary.get('total_measurements', 0)}"
                    operations_tracked_text = f"📋 **Operations Tracked**: {len(summary.get('operations_tracked', []))}"
                    
                    # Calculate average latency
                    all_times = []
                    for op_stats in stats.values():
                        if 'avg_time' in op_stats:
                            all_times.append(op_stats['avg_time'])
                    
                    avg_latency_text = f"⏱️ **Average Latency**: {sum(all_times)/len(all_times):.3f}s" if all_times else "⏱️ **Average Latency**: 0.00s"
                    
                    # Calculate P95 latency
                    all_durations = []
                    for metric in latency_tracker._metrics:
                        all_durations.append(metric.duration)
                    
                    if all_durations:
                        sorted_durations = sorted(all_durations)
                        p95_index = int(0.95 * (len(sorted_durations) - 1))
                        p95_latency_text = f"📈 **P95 Latency**: {sorted_durations[p95_index]:.3f}s"
                    else:
                        p95_latency_text = "📈 **P95 Latency**: 0.00s"
                    
                    # Bottlenecks
                    bottlenecks = summary.get('bottlenecks', [])
                    if bottlenecks:
                        bottleneck_text = "⚠️ **Bottlenecks**:"
                        for bottleneck in bottlenecks[:3]:  # Show top 3
                            bottleneck_text += f"\n- {bottleneck['operation']}: {bottleneck['avg_time']:.3f}s ({bottleneck['severity']})"
                    else:
                        bottleneck_text = "⚠️ **Bottlenecks**: None detected"
                    
                    # Recommendations
                    recommendations = summary.get('recommendations', [])
                    if recommendations:
                        recommendations_text = "💡 **Recommendations**:"
                        for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                            recommendations_text += f"\n{i}. {rec}"
                    else:
                        recommendations_text = "💡 **Recommendations**: None available"
                    
                    # Pipeline stats table
                    pipeline_data = []
                    for stage in ['document_chunking', 'vector_retrieval', 'llm_generation', 'end_to_end_query']:
                        if stage in stats:
                            stage_stats = stats[stage]
                            pipeline_data.append([
                                stage,
                                stage_stats.get('count', 0),
                                round(stage_stats.get('avg_time', 0), 3),
                                round(stage_stats.get('p95_time', 0), 3),
                                round(stage_stats.get('max_time', 0), 3)
                            ])
                    
                    # Recent metrics table
                    recent_data = []
                    recent_metrics_list = latency_tracker._metrics[-10:]  # Last 10 metrics
                    for metric in recent_metrics_list:
                        recent_data.append([
                            metric.operation,
                            round(metric.duration, 3),
                            time.strftime('%H:%M:%S', time.localtime(metric.timestamp)),
                            json.dumps(metric.metadata) if metric.metadata else "{}"
                        ])
                    
                    return (
                        total_measurements_text,
                        operations_tracked_text,
                        avg_latency_text,
                        p95_latency_text,
                        pipeline_data,
                        recent_data
                    )
                except Exception as e:
                    return (
                        "📊 **Total Measurements**: Error",
                        "📋 **Operations Tracked**: Error",
                        "⏱️ **Average Latency**: Error",
                        "📈 **P95 Latency**: Error",
                        f"⚠️ **Bottlenecks**: {str(e)}",
                        "💡 **Recommendations**: Error",
                        [],
                        []
                    )
            
            # Clear metrics function
            def clear_metrics():
                try:
                    latency_tracker.reset()
                    return "🧹 Metrics cleared successfully"
                except Exception as e:
                    return f" Clear failed: {str(e)}"
            
            # Event handlers
            export_csv_btn.click(
                export_to_csv,
                outputs=[demo_status]
            )
            
            export_json_btn.click(
                export_to_json,
                outputs=[demo_status]
            )
            
            clear_metrics_btn.click(
                clear_metrics,
                outputs=[demo_status]
            )
            
            refresh_stats_btn.click(
                update_stats,
                outputs=[
                    total_measurements,
                    operations_tracked,
                    avg_latency,
                    p95_latency,
                    pipeline_stats,
                    recent_metrics
                ]
            )
            
            # Auto-refresh stats on tab load
            demo.load(
                update_stats,
                outputs=[
                    total_measurements,
                    operations_tracked,
                    avg_latency,
                    p95_latency,
                    pipeline_stats,
                    recent_metrics
                ]
            )

    return demo


def main():
    demo = create_integrated_demo()
    demo.launch(
        server_name="localhost",
        server_port=7860,
        inbrowser=True,
        debug=True,
        css=custom_css,  # Gradio 6.x correct placement
    )


if __name__ == "__main__":
    main()
