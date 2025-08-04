FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    apt-utils \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt --no-warn-script-location

# Pre-download HuggingFace model (adjust model name to what you use in download_embeddings)
RUN python -c "from langchain_community.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')"

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose Cloud Run port
EXPOSE 8080

# Start Flask
CMD ["python3", "app.py"]
