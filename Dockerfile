# Use the official Python image
FROM python:3.11-slim

# Build argument for HF token (optional, for higher rate limits during build)
ARG HF_TOKEN

# Set working directory
WORKDIR /app

# Set environment variables for memory optimization and model caching
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10 \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Install dependencies FIRST (this layer is cached if requirements.txt doesn't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model SECOND with explicit cache location
# Use HF_TOKEN if provided to avoid rate limits during build
RUN mkdir -p /app/.cache/sentence_transformers && \
    if [ -n "$HF_TOKEN" ]; then export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN; fi && \
    export SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers && \
    python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); print('Model downloaded to:', model._model_card_vars.get('model_path', 'unknown'))" && \
    echo "=== Cache directory contents ===" && \
    find /app/.cache -type f | head -20 && \
    echo "=== Total cache size ===" && \
    du -sh /app/.cache

# Copy app code LAST (changes frequently, but previous layers are cached)
COPY . .

# Expose the port Streamlit runs on (Cloud Run defaults to 8080)
EXPOSE 8080

# Run Streamlit on container startup with memory optimizations
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.maxUploadSize=10", \
     "--browser.gatherUsageStats=false"]