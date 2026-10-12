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

# Explicitly verify PyTorch is still the official ROCm build
RUN python3 -c "import torch; assert '+rocm' in torch.__version__, f'Fatal: PyTorch overwritten by non-ROCm version: {torch.__version__}'; print('ROCm PyTorch Verified:', torch.__version__)"

# Copy pre-baked OCR model weights to /models (AMD official specification) and /app/models
COPY models /models
COPY models /app/models

# Populate ~/.EasyOCR/model for seamless offline execution
RUN mkdir -p /root/.EasyOCR/model && cp /models/* /root/.EasyOCR/model/

# Copy application code
COPY app.py /app/app.py

# Ensure standard input/output directories exist
RUN mkdir -p /app/input /app/output

# Standard daemon command to keep container alive for docker exec
CMD ["sleep", "infinity"]
