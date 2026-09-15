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

# Pre-download OCR model weights to ensure instantaneous inference under 30s limit
RUN python3 -c "import easyocr; reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)"

# Copy application code
COPY app.py /app/app.py

# Ensure standard input/output directories exist
RUN mkdir -p /app/input /app/output

CMD ["python3", "/app/app.py", "--help"]
