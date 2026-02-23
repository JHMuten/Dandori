# Use the official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose the port Streamlit runs on (Cloud Run defaults to 8080)
EXPOSE 8080

# Run Streamlit on container startup
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]