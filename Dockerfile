FROM python:3.11

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . .

# Create start script
USER root
COPY <<EOF /app/start.sh
#!/bin/bash
echo "Starting RAG Assistant on port 7860..."
cd project
python app.py
EOF
RUN chmod +x /app/start.sh && chown user:user /app/start.sh

# Switch to non-root user
USER user

# Expose port
EXPOSE 7860

# Start the application
CMD ["/app/start.sh"]
