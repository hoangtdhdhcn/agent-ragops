custom_css = """
    /* ============================================
       MAIN CONTAINER
       ============================================ */
    .progress-text { 
        display: none !important;
    }
    
    .gradio-container { 
        max-width: 1200px !important;
        width: 100% !important;
        margin: 0 auto !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        min-height: 100vh !important;
    }
    
    /* ============================================
       GLOBAL STYLES
       ============================================ */
    :root {
        --primary-color: #3b82f6;
        --primary-hover: #2563eb;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --bg-card: #ffffff;
        --bg-card-hover: #f9fafb;
        --border-color: #e5e7eb;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.15);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
    }
    
    /* Animated background elements */
    .gradio-container::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 20%, rgba(59, 130, 246, 0.05), transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(139, 92, 246, 0.05), transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* ============================================
       STICKY CHATBOX
       ============================================ */
    .sticky-chatbot {
        position: sticky !important;
        top: 20px !important;
        z-index: 10 !important;
        max-height: 70vh !important;
        overflow-y: auto !important;
        scroll-behavior: smooth !important;
    }
    
    /* Ensure chat messages can scroll within the sticky container */
    .sticky-chatbot .message-container {
        overflow-y: auto !important;
        max-height: 60vh !important;
    }
    
    /* Smooth scrolling for chat content */
    .sticky-chatbot::-webkit-scrollbar {
        width: 6px !important;
    }
    
    .sticky-chatbot::-webkit-scrollbar-track {
        background: #f1f5f9 !important;
        border-radius: 3px !important;
    }
    
    .sticky-chatbot::-webkit-scrollbar-thumb {
        background: #cbd5e1 !important;
        border-radius: 3px !important;
        transition: background 0.3s ease !important;
    }
    
    .sticky-chatbot::-webkit-scrollbar-thumb:hover {
        background: #94a3b8 !important;
    }
    
    /* ============================================
       RETRIEVED CHUNKS SCROLLING
       ============================================ */
    .retrieved-chunks-container {
        max-height: 300px !important;
        overflow-y: auto !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        background: #ffffff !important;
    }
    
    .retrieved-chunks-content {
        max-height: 280px !important;
        overflow-y: auto !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Custom scrollbar for retrieved chunks */
    .retrieved-chunks-container::-webkit-scrollbar {
        width: 6px !important;
    }
    
    .retrieved-chunks-container::-webkit-scrollbar-track {
        background: #f1f5f9 !important;
        border-radius: 3px !important;
    }
    
    .retrieved-chunks-container::-webkit-scrollbar-thumb {
        background: #cbd5e1 !important;
        border-radius: 3px !important;
        transition: background 0.3s ease !important;
    }
    
    .retrieved-chunks-container::-webkit-scrollbar-thumb:hover {
        background: #94a3b8 !important;
    }
    
    /* ============================================
       LATENCY MONITOR TAB
       ============================================ */
    /* Compact metric cards for key metrics */
    .metric-card-small {
        background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        margin-bottom: 8px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-align: center !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all 0.3s ease !important;
    }
    
    .metric-card-small:hover {
        transform: translateY(-2px) !important;
        border-color: var(--primary-color) !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    /* Latency status indicator */
    .latency-status {
        background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        text-align: center !important;
    }
    
    /* Compact dataframes for latency tables */
    .compact-dataframe {
        max-height: 250px !important;
        overflow-y: auto !important;
        font-size: 12px !important;
        line-height: 1.2 !important;
    }
    
    /* Compact dataframe headers */
    .compact-dataframe .table th {
        font-size: 11px !important;
        padding: 6px !important;
        background-color: #f8fafc !important;
        border-bottom: 1px solid var(--border-color) !important;
        color: #1f2937 !important;
        font-weight: 600 !important;
    }
    
    /* Compact dataframe cells - white/grey text */
    .compact-dataframe .table td {
        font-size: 12px !important;
        padding: 6px !important;
        border-bottom: 1px solid #f1f5f9 !important;
        color: #6b7280 !important;  /* Grey text for content */
        font-weight: 500 !important;
        background-color: #ffffff !important;
    }
    
    /* Compact dataframe labels */
    .compact-dataframe .gr-form .gr-label {
        color: #1f2937 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    
    /* Ensure dataframe content is grey except headers */
    .compact-dataframe .table tbody * {
        color: #6b7280 !important;
    }
    
    /* Keep headers black */
    .compact-dataframe .table thead * {
        color: #1f2937 !important;
    }
    
    /* Compact buttons for latency controls */
    .latency-controls .gr-button {
        font-size: 12px !important;
        padding: 8px 12px !important;
        height: auto !important;
        min-height: 32px !important;
    }
    
    /* Ensure latency tab fits in viewport */
    .latency-tab-content {
        max-height: 80vh !important;
        overflow-y: auto !important;
    }
    
    /* Selected tab - gradient underline */
    button[role="tab"][aria-selected="true"] {
        color: var(--text-primary) !important;
        border-bottom: none !important;
        background: transparent !important;
    }
    
    button[role="tab"][aria-selected="true"]::after {
        content: '' !important;
        position: absolute !important;
        bottom: 0 !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 60% !important;
        height: 3px !important;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)) !important;
        border-radius: 2px !important;
        animation: tabUnderline 0.3s ease-out !important;
    }
    
    /* Remove any pseudo-elements that might create lines */
    button[role="tab"]::before,
    button[role="tab"]::after,
    .tabs::before,
    .tabs::after,
    .tab-nav::before,
    .tab-nav::after {
        display: none !important;
        content: none !important;
    }
    
    /* Center document management tab */
    #doc-management-tab {
        max-width: 600px !important;
        margin: 0 auto !important;
        padding: 24px !important;
    }
    
    /* Tab underline animation */
    @keyframes tabUnderline {
        from {
            width: 0% !important;
            opacity: 0 !important;
        }
        to {
            width: 60% !important;
            opacity: 1 !important;
        }
    }
    
    /* ============================================
       BUTTONS - ENHANCED
       ============================================ */
    button {
        border-radius: var(--radius-sm) !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.025em !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: var(--shadow-sm) !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: uppercase !important;
    }
    
    /* Primary button - gradient with hover effects */
    .primary {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
        color: white !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
    }
    
    .primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-lg) !important;
        filter: brightness(1.1) !important;
    }
    
    .primary:active {
        transform: translateY(0) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* Secondary button */
    .secondary {
        background: rgba(255, 255, 255, 0.8) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .secondary:hover {
        background: rgba(255, 255, 255, 1) !important;
        border-color: var(--primary-color) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Stop/danger button */
    .stop {
        background: linear-gradient(135deg, var(--danger-color), #dc2626) !important;
        color: white !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
    }
    
    .stop:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-lg) !important;
        filter: brightness(1.1) !important;
    }
    
    /* Success button */
    .success {
        background: linear-gradient(135deg, var(--success-color), #059669) !important;
        color: white !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
    }
    
    .success:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-lg) !important;
        filter: brightness(1.1) !important;
    }
    
    /* Button ripple effect */
    button::before {
        content: '' !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        width: 0 !important;
        height: 0 !important;
        background: rgba(255, 255, 255, 0.3) !important;
        border-radius: 50% !important;
        transform: translate(-50%, -50%) !important;
        transition: width 0.6s, height 0.6s !important;
    }
    
    button:active::before {
        width: 300px !important;
        height: 300px !important;
    }
    
    /* ============================================
       CHAT INPUT BOX - MODIFIED
       ============================================ */
    /* Target chat input textarea - more aggressive selectors */
    textarea[placeholder="Type a message..."],
    textarea[data-testid*="textbox"]:not(#file-list-box textarea) {
        background: #ffffff !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: none !important;
        border-radius: 8px !important;
    }
    
    textarea[placeholder="Type a message..."]:focus {
        background: #ffffff !important;
        border: 1px solid var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Target the wrapper/container of chat input */
    .gr-text-input:has(textarea[placeholder="Type a message..."]),
    [class*="chatbot"] + * [data-testid="textbox"],
    form:has(textarea[placeholder="Type a message..."]) > div {
        background: transparent !important;
        border: none !important;
        gap: 12px !important;
    }
    
    /* Remove background from submit button in chat */
    form:has(textarea[placeholder="Type a message..."]) button,
    [class*="chatbot"] ~ * button[type="submit"] {
        background: transparent !important;
        border: none !important;
        padding: 8px !important;
    }
    
    form:has(textarea[placeholder="Type a message..."]) button:hover {
        background: rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Add spacing to the form container */
    form:has(textarea[placeholder="Type a message..."]) {
        gap: 12px !important;
        display: flex !important;
    }
    
    /* ============================================
       FILE UPLOAD - ENHANCED
       ============================================ */
    .file-preview, 
    [data-testid="file-upload"] {
        background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        color: #1f2937 !important;
        min-height: 200px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .file-preview::before,
    [data-testid="file-upload"]::before {
        content: '📁' !important;
        position: absolute !important;
        top: 20px !important;
        right: 20px !important;
        font-size: 48px !important;
        opacity: 0.1 !important;
        pointer-events: none !important;
        transition: all 0.3s ease !important;
    }
    
    .file-preview:hover, 
    [data-testid="file-upload"]:hover {
        border-color: var(--primary-color) !important;
        background: linear-gradient(145deg, #f9fafb, #ffffff) !important;
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    .file-preview:hover::before,
    [data-testid="file-upload"]:hover::before {
        opacity: 0.3 !important;
        transform: rotate(5deg) !important;
    }
    
    .file-preview.drag-over,
    [data-testid="file-upload"].drag-over {
        border-color: var(--success-color) !important;
        background: linear-gradient(145deg, #ecfdf5, #f0fdf4) !important;
        animation: pulse 1s infinite !important;
    }
    
    /* Text inside file upload - dark color */
    .file-preview *,
    [data-testid="file-upload"] * {
        color: #1f2937 !important;
    }
    
    /* Hide file upload label */
    .file-preview .label,
    [data-testid="file-upload"] .label {
        display: none !important;
    }
    
    /* File upload text styling */
    .file-preview .file-preview-text,
    [data-testid="file-upload"] .file-preview-text {
        font-size: 14px !important;
        color: var(--text-secondary) !important;
        text-align: center !important;
        padding: 20px !important;
    }
    
    /* Pulse animation for drag over state */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* ============================================
       INPUTS & TEXTAREAS
       ============================================ */
    input, 
    textarea {
        background: #ffffff !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: #1f2937 !important;
        transition: border-color 0.2s ease !important;
    }
    
    input:focus, 
    textarea:focus {
        border-color: var(--primary-color) !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    textarea[readonly] {
        background: #ffffff !important;
        color: #6b7280 !important;
    }
    
    /* ============================================
       FILE LIST BOX
       ============================================ */
    #file-list-box {
        background: #ffffff !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    
    #file-list-box textarea {
        background: transparent !important;
        border: none !important;
        color: #1f2937 !important;
        padding: 0 !important;
    }
    
    /* ============================================
       CHATBOT - ENHANCED
       ============================================ */
    .chatbot {
        border-radius: var(--radius-md) !important;
        background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow-md) !important;
        overflow: hidden !important;
        position: relative !important;
    }
    
    /* Chat message styling */
    .message {
        border-radius: var(--radius-md) !important;
        width: fit-content !important;
        max-width: 80% !important;
        padding: 14px 18px !important;
        margin: 8px 0 !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        animation: messageSlideIn 0.3s ease-out !important;
        font-size: 16px !important; /* Original chat text size */
        line-height: 1.6 !important;
    }
    
    .message.user {
        background: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid var(--border-color) !important;
        margin-left: auto !important;
        border-bottom-right-radius: 4px !important;
    }
    
    .message.bot {
        background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
        color: #1f2937 !important;
        border: 1px solid var(--border-color) !important;
        border-bottom-left-radius: 4px !important;
    }
    
    .message:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-lg) !important;
    }
    
    /* Message timestamp styling */
    .message .timestamp {
        font-size: 11px !important;
        opacity: 0.6 !important;
        margin-top: 4px !important;
        display: block !important;
    }
    
    /* Source citations and file names - ensure they are black */
    .message .source-citation,
    .message .file-name,
    .message strong,
    .message b {
        color: #1f2937 !important;
        font-weight: 600 !important;
    }
    
    /* Ensure any text within messages is dark */
    .message * {
        color: #1f2937 !important;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: inline-flex !important;
        gap: 6px !important;
        align-items: center !important;
        padding: 14px 18px !important;
        border-radius: var(--radius-md) !important;
        background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
        border: 1px solid var(--border-color) !important;
        margin: 8px 0 !important;
    }
    
    .typing-dot {
        width: 8px !important;
        height: 8px !important;
        background: var(--text-secondary) !important;
        border-radius: 50% !important;
        display: inline-block !important;
        animation: typing 1.4s ease-in-out infinite !important;
    }
    
    .typing-dot:nth-child(1) { animation-delay: 0s !important; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s !important; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s !important; }
    
    /* Message animations */
    @keyframes messageSlideIn {
        from {
            opacity: 0 !important;
            transform: translateY(10px) !important;
        }
        to {
            opacity: 1 !important;
            transform: translateY(0) !important;
        }
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0) !important;
            opacity: 0.4 !important;
        }
        30% {
            transform: translateY(-10px) !important;
            opacity: 1 !important;
        }
    }
    
    /* ============================================
       EVALUATION METRICS - ENHANCED
       ============================================ */
    /* Evaluation sidebar styling */
    .evaluation-sidebar {
        background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        padding: 20px !important;
        box-shadow: var(--shadow-md) !important;
        position: sticky !important;
        top: 20px !important;
        height: fit-content !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-sm) !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
    }
    
    .metric-card:hover {
        transform: translateY(-2px) !important;
        border-color: var(--primary-color) !important;
        box-shadow: var(--shadow-lg) !important;
    }
    
    /* Status indicators */
    .status-pass {
        color: var(--success-color) !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.3) !important;
    }
    
    .status-fail {
        color: var(--danger-color) !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(239, 68, 68, 0.3) !important;
    }
    
    .status-warning {
        color: var(--warning-color) !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(245, 158, 11, 0.3) !important;
    }
    
    /* Overall score styling */
    .overall-score {
        font-size: 24px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-shadow: 0 0 20px rgba(59, 130, 246, 0.3) !important;
        display: inline-block !important;
    }
    
    /* Accordion styling */
    .gr-accordion {
        background: transparent !important;
        border: none !important;
    }
    
    .gr-accordion-header {
        background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-accordion-header:hover {
        background: linear-gradient(145deg, #f9fafb, #ffffff) !important;
        border-color: var(--primary-color) !important;
    }
    
    .gr-accordion-content {
        background: #ffffff !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
        padding: 16px !important;
    }
    
    /* Evaluation report styling */
    .evaluation-report {
        background: #ffffff !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-sm) !important;
        padding: 16px !important;
        font-family: 'Courier New', monospace !important;
        font-size: 12px !important;
        line-height: 1.6 !important;
        color: #1f2937 !important;
        max-height: 300px !important;
        overflow-y: auto !important;
    }
    
    /* Metric icons */
    .metric-icon {
        display: inline-block !important;
        width: 20px !important;
        height: 20px !important;
        border-radius: 50% !important;
        margin-right: 8px !important;
        animation: pulse 2s infinite !important;
    }
    
    .icon-groundedness { background: var(--success-color) !important; }
    .icon-relevance { background: var(--primary-color) !important; }
    .icon-retrieval { background: var(--warning-color) !important; }
    
    /* ============================================
       PROGRESS BAR - ENHANCED
       ============================================ */
    .progress-bar-wrap {
        border-radius: var(--radius-sm) !important;
        overflow: hidden !important;
        background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
        border: 1px solid var(--border-color) !important;
        height: 8px !important;
    }

    .progress-bar {
        border-radius: var(--radius-sm) !important;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)) !important;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.5) !important;
        animation: progressGlow 2s ease-in-out infinite !important;
    }
    
    @keyframes progressGlow {
        0%, 100% { box-shadow: 0 0 10px rgba(59, 130, 246, 0.5); }
        50% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.8); }
    }
    
    /* ============================================
       TYPOGRAPHY - ENHANCED
       ============================================ */
    /* Enhanced typography with better hierarchy */
    h1 {
        font-size: 28px !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        color: #1f2937 !important;
        margin-bottom: 16px !important;
    }
    
    h2 {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
        margin-bottom: 12px !important;
        letter-spacing: -0.01em !important;
    }
    
    h3 {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin-bottom: 8px !important;
    }
    
    h4, h5, h6 {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-bottom: 6px !important;
    }
    
    /* Body text styling */
    p, .gr-markdown {
        color: var(--text-secondary) !important;
        line-height: 1.6 !important;
        font-size: 14px !important;
        margin-bottom: 12px !important;
    }
    
    /* Emphasis text */
    strong, b {
        color: #1f2937 !important;
        font-weight: 700 !important;
    }
    
    /* Code styling */
    code {
        background: #f3f4f6 !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-family: 'Courier New', monospace !important;
        color: #1f2937 !important;
        font-size: 12px !important;
    }
    
    /* ============================================
       LOADING STATES & FEEDBACK
       ============================================ */
    /* Loading spinner */
    .loading-spinner {
        display: inline-block !important;
        width: 20px !important;
        height: 20px !important;
        border: 3px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 50% !important;
        border-top-color: var(--primary-color) !important;
        animation: spin 1s ease-in-out infinite !important;
        margin-right: 8px !important;
    }
    
    /* Enhanced loading spinner for evaluation updates */
    .evaluation-loading {
        display: inline-flex !important;
        align-items: center !important;
        gap: 12px !important;
        padding: 8px 16px !important;
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid var(--primary-color) !important;
        border-radius: 8px !important;
        color: var(--primary-color) !important;
        font-weight: 600 !important;
        animation: pulse 1.5s ease-in-out infinite !important;
    }
    
    .evaluation-loading .spinner {
        width: 16px !important;
        height: 16px !important;
        border: 2px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 50% !important;
        border-top-color: var(--primary-color) !important;
        animation: spin 1s ease-in-out infinite !important;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Enhanced spin animation for evaluation loading */
    @keyframes evaluationSpin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Toast notifications */
    .toast {
        position: fixed !important;
        top: 20px !important;
        right: 20px !important;
        background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-sm) !important;
        padding: 16px 20px !important;
        box-shadow: var(--shadow-lg) !important;
        z-index: 1000 !important;
        transform: translateX(120%) !important;
        transition: transform 0.3s ease !important;
        animation: toastSlideIn 0.3s ease-out !important;
    }
    
    .toast.show {
        transform: translateX(0) !important;
    }
    
    .toast.success {
        border-left: 4px solid var(--success-color) !important;
    }
    
    .toast.error {
        border-left: 4px solid var(--danger-color) !important;
    }
    
    .toast.info {
        border-left: 4px solid var(--primary-color) !important;
    }
    
    @keyframes toastSlideIn {
        from {
            transform: translateX(120%) !important;
            opacity: 0 !important;
        }
        to {
            transform: translateX(0) !important;
            opacity: 1 !important;
        }
    }
    
    /* ============================================
       CARDS & CONTAINERS
       ============================================ */
    /* Card styling */
    .card {
        background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        padding: 20px !important;
        box-shadow: var(--shadow-md) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .card:hover {
        transform: translateY(-2px) !important;
        border-color: var(--primary-color) !important;
        box-shadow: var(--shadow-lg) !important;
    }
    
    /* Section dividers */
    .divider {
        height: 1px !important;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent) !important;
        margin: 20px 0 !important;
        width: 100% !important;
    }
    
    /* ============================================
       RESPONSIVE DESIGN
       ============================================ */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 16px !important;
        }
        
        #doc-management-tab {
            padding: 16px !important;
            max-width: 100% !important;
        }
        
        .evaluation-sidebar {
            position: relative !important;
            top: 0 !important;
            margin-bottom: 20px !important;
        }
        
        .message {
            max-width: 90% !important;
        }
        
        button[role="tab"] {
            padding: 12px 16px !important;
            font-size: 12px !important;
        }
        
        /* Mobile-specific font size adjustments */
        body {
            font-size: 16px !important;
        }
        
        .message {
            font-size: 13.5px !important;
            padding: 14px 18px !important;
        }
        
        .primary, .secondary, .stop, .success {
            font-size: 13.5px !important;
            padding: 12px 24px !important;
        }
    }
    
    /* ============================================
       GLOBAL OVERRIDES
       ============================================ */
    * {
        box-shadow: none !important;
    }
    
    footer {
        visibility: hidden;
    }
"""