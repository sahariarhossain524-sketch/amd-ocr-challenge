# AMD AI Academy Challenge — Mini-Challenge 2: Optical Character Recognition (OCR)

This repository contains the complete, production-ready, AMD ROCm-compliant solution for **Mini-Challenge 2 (Optical Character Recognition)** in the **Lablab x AMD AI Academy Challenge**.

---

## 🎯 Architecture & Specifications

- **Mandated Base Image**: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`
- **Supported Formats**: PNG, JPEG, TIFF
- **VRAM Enforcement**: Automatically maintains 1 GiB – 48 GiB active VRAM allocation per AMD evaluation contract
- **Multilingual Support**: English, Numerics, and Chinese Province characters (`京`, `沪`, etc.)
- **Noise & Low-Light Resilience**: CLAHE contrast-enhancement and fast non-local means denoising
- **State Banner Stripping**: Drops US state names (CALIFORNIA, TEXAS, NEW YORK) and slogan banners while preserving registration plate characters.

---

## 📂 Project Structure

```
amd-ocr-challenge/
├── app.py              # Main OCR inference engine & Section 2 invocation contract
├── Dockerfile          # ROCm mandated base image & baked model weights
├── requirements.txt    # Python runtime dependencies
├── test_local.py       # Test harness simulating the 10 challenge categories
├── input/              # Test input images (PNG, JPEG, TIFF)
└── output/             # JSON output answers (e.g. image_01_output.json)
```

---

## 🚀 How to Build & Submit to Lablab.ai

### Step 1: Build Docker Container
```bash
docker build -t <your-dockerhub-username>/amd-ocr-challenge:latest .
```

### Step 2: Test Locally
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output <your-dockerhub-username>/amd-ocr-challenge:latest python3 /app/app.py --input-image /app/input/image_01.png
```

### Step 3: Push to Public Docker Registry
```bash
docker push <your-dockerhub-username>/amd-ocr-challenge:latest
```

### Step 4: Submit on Lablab.ai
Submit your public Docker image URL:
```
<your-dockerhub-username>/amd-ocr-challenge:latest
```
