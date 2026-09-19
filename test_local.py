import os
import json
import subprocess
import sys
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate synthetic sample test images to simulate the challenge categories
def create_sample_images():
    # Format: filename, text_lines, (width, height), bg_color_bgr, text_color_bgr
    samples = [
        ("image_01.png", ["CALIFORNIA", "7ABC123"], (600, 300), (255, 255, 255), (20, 20, 20)),
        ("image_06.png", ["STOP"], (400, 400), (40, 40, 220), (255, 255, 255)),
        ("image_08.jpg", ["SPEED", "LIMIT", "65"], (400, 500), (255, 255, 255), (20, 20, 20)),
        ("image_09.png", ["ROAD", "WORK", "AHEAD"], (500, 500), (0, 140, 255), (20, 20, 20)),
        ("image_10.tiff", ["35"], (400, 400), (0, 215, 255), (20, 20, 20)),
    ]

    for fname, lines, (w, h), bg_color, text_color in samples:
        img = np.full((h, w, 3), bg_color, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.3
        thickness = 2

        total_lines = len(lines)
        step_y = h // (total_lines + 1)

        for i, line in enumerate(lines):
            text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
            text_x = (w - text_size[0]) // 2
            text_y = (i + 1) * step_y + (text_size[1] // 2)
            cv2.putText(img, line, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        path = os.path.join(INPUT_DIR, fname)
        cv2.imwrite(path, img)
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
    max_score = len(test_cases) * 20
    print(f"\n=======================================================")
    print(f"LOCAL TEST SCORE: {total_score} / {max_score} on simulated sample set ({passed}/{len(test_cases)} Passed)")
    print(f"NOTE: Official challenge uses 10 hidden images (200 pts total) on AMD ROCm cluster.")
    print(f"=======================================================\n")

if __name__ == "__main__":
    run_tests()
