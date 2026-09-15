import os
import json
import subprocess
import sys
from PIL import Image, ImageDraw

BASE_DIR = r"c:\Users\ASUS\OneDrive\Desktop\stitch\amd-ocr-challenge"
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate synthetic sample test images to simulate the 10 challenge categories
def create_sample_images():
    samples = [
        ("image_01.png", "CALIFORNIA\n7ABC123", (400, 200), (255, 255, 255), (0, 0, 0)),
        ("image_06.png", "STOP", (300, 300), (220, 20, 60), (255, 255, 255)),
        ("image_08.jpg", "SPEED\nLIMIT\n65", (300, 400), (255, 255, 255), (0, 0, 0)),
        ("image_09.png", "ROAD\nWORK\nAHEAD", (400, 400), (255, 140, 0), (0, 0, 0)),
        ("image_10.tiff", "35", (300, 300), (255, 215, 0), (0, 0, 0)),
    ]

    for fname, text, size, bg_color, text_color in samples:
        path = os.path.join(INPUT_DIR, fname)
        img = Image.new('RGB', size, color=bg_color)
        draw = ImageDraw.Draw(img)
        draw.text((40, 50), text, fill=text_color)
        img.save(path)
        print(f"Generated sample: {path}")

def normalize(text):
    t = text.upper()
    for ch in [' ', '\t', '\n', '-', '.', '·', '_']:
        t = t.replace(ch, '')
    return t

def run_tests():
    create_sample_images()

    test_cases = [
        ("image_01.png", "7ABC123"),
        ("image_06.png", "STOP"),
        ("image_08.jpg", "SPEEDLIMIT65"),
        ("image_09.png", "ROADWORKAHEAD"),
        ("image_10.tiff", "35"),
    ]

    print("\n--- Running AMD OCR Challenge Local Harness ---")
    passed = 0
    for fname, expected in test_cases:
        in_file = os.path.join(INPUT_DIR, fname)
        cmd = [sys.executable, os.path.join(BASE_DIR, "app.py"), "--input-image", in_file]
        res = subprocess.run(cmd, capture_output=True, text=True)

        base_name, _ = os.path.splitext(fname)
        out_file = os.path.join(OUTPUT_DIR, f"{base_name}_output.json")

        if os.path.exists(out_file):
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            pred = data.get("text", "")
            norm_pred = normalize(pred)
            norm_exp = normalize(expected)

            match = (norm_pred == norm_exp)
            status = "PASS (20 pts)" if match else "MISMATCH"
            if match:
                passed += 1
            print(f"[{status}] {fname} -> Pred: '{pred}' (Norm: '{norm_pred}') | Expected: '{norm_exp}'")
        else:
            print(f"[FAIL] Output file missing for {fname}: {res.stderr}")

    total_score = passed * 20
    print(f"\n=======================================================")
    print(f"TOTAL SCORE: {total_score} / 100 on simulated sample set ({passed}/{len(test_cases)} Passed)")
    print(f"=======================================================\n")

if __name__ == "__main__":
    run_tests()
