# Mandatory AMD ROCm Base Image
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# Install runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy pre-baked OCR model weights to /models (AMD official specification) and /app/models
COPY models /models
COPY models /app/models

# Populate ~/.EasyOCR/model for seamless offline execution
RUN mkdir -p /root/.EasyOCR/model && cp /models/* /root/.EasyOCR/model/

# Copy application code
COPY app.py /app/app.py

# Ensure standard input/output directories exist
RUN mkdir -p /app/input /app/output

CMD ["python3", "/app/app.py", "--help"]
