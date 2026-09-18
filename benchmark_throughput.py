import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Import from app.py
from app import get_reader, load_and_enhance_image, denoise_fallback, clean_and_normalize

def eager_preprocess_and_infer(reader, image_path):
    img = cv2.imread(image_path)
    if img is None:
        pil_img = Image.open(image_path).convert('RGB')
        img = np.array(pil_img)[:, :, ::-1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Eager slow denoise executed on every image
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)

    results = reader.readtext(enhanced)
    if not results or len(results) == 0:
        results = reader.readtext(img)
    if not results or len(results) == 0:
        results = reader.readtext(denoised)

    return clean_and_normalize(results)

def lazy_preprocess_and_infer(reader, image_path):
    img, enhanced = load_and_enhance_image(image_path)
    results = reader.readtext(enhanced)
    if not results or len(results) == 0:
        results = reader.readtext(img)
    if not results or len(results) == 0:
        denoised = denoise_fallback(enhanced)
        results = reader.readtext(denoised)

    return clean_and_normalize(results)

def run_benchmark(image_path, runs=20):
    print("================================================================")
    print("AMD ROCm OCR Engine Throughput Benchmark (Hongwei Guo Protocol)")
    print("================================================================")
    print(f"Target Image: {image_path}")
    print(f"Iterations  : {runs} runs")

    print("\nWarming up EasyOCR model & ROCm GPU memory allocation...")
    reader = get_reader()
    _ = lazy_preprocess_and_infer(reader, image_path)
    print("Warmup complete.\n")

    # 1. Benchmark Eager Baseline
    print(f"Benchmarking [Baseline: Eager Denoise Every Frame] ({runs} runs)...")
    eager_times = []
    eager_text, eager_conf = None, None
    for i in range(runs):
        t0 = time.perf_counter()
        eager_text, eager_conf = eager_preprocess_and_infer(reader, image_path)
        t1 = time.perf_counter()
        eager_times.append(t1 - t0)

    avg_eager = sum(eager_times) / len(eager_times)

    # 2. Benchmark Lazy Optimized
    print(f"Benchmarking [Optimized: Lazy Conditional Denoise] ({runs} runs)...")
    lazy_times = []
    lazy_text, lazy_conf = None, None
    for i in range(runs):
        t0 = time.perf_counter()
        lazy_text, lazy_conf = lazy_preprocess_and_infer(reader, image_path)
        t1 = time.perf_counter()
        lazy_times.append(t1 - t0)

    avg_lazy = sum(lazy_times) / len(lazy_times)
    speedup = ((avg_eager - avg_lazy) / avg_eager) * 100.0

    print("\n------------------------- RESULTS ------------------------------")
    print(f"Baseline Path (Eager Denoise)  : {avg_eager*1000:.2f} ms / frame ({avg_eager:.3f} s)")
    print(f"Optimized Path (Lazy Denoise)   : {avg_lazy*1000:.2f} ms / frame ({avg_lazy:.3f} s)")
    print(f"Net Latency Reduction           : -{(avg_eager - avg_lazy)*1000:.2f} ms ({speedup:.1f}% faster)")
    print(f"Target Threshold (< 1.2s)       : {'MET (PASS)' if avg_lazy < 1.2 else 'PENDING'}")
    print(f"Text Integrity Verification     : {'PRESERVED' if eager_text == lazy_text else 'MISMATCH'}")
    print(f"Recognized Text                 : '{lazy_text}' (conf: {lazy_conf})")
    print("================================================================\n")

if __name__ == '__main__':
    default_img = os.path.join(os.path.dirname(__file__), "input", "image_01.png")
    if not os.path.exists(default_img):
        os.makedirs(os.path.dirname(default_img), exist_ok=True)
        img = Image.new('RGB', (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((40, 50), "CALIFORNIA\n7ABC123", fill=(0, 0, 0))
        img.save(default_img)
    run_benchmark(default_img, runs=20)
