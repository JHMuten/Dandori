# Use the official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables for memory optimization
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10 \
    TRANSFORMERS_CACHE=/tmp/transformers_cache \
    HF_HOME=/tmp/hf_home \
    TORCH_HOME=/tmp/torch_home

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose the port Streamlit runs on (Cloud Run defaults to 8080)
EXPOSE 8080

# Run Streamlit on container startup with memory optimizations
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.maxUploadSize=10", \
     "--browser.gatherUsageStats=false"]