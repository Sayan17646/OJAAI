# Use official lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OCR, PDF conversion, and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install python dependencies (excluding heavy ML packages for lightweight cloud build)
COPY requirements.txt .
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/transformers/d' requirements.txt && \
    sed -i '/accelerate/d' requirements.txt && \
    sed -i '/sentencepiece/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt


# Copy application source code
COPY src/ ./src/
COPY tests/ ./tests/

# Expose port (Render sets $PORT dynamically, so we launch via a shell command)
EXPOSE 8000

# Start server using the shell to evaluate $PORT variable dynamically
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port $PORT"]
