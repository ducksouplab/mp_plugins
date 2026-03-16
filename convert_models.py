#!/usr/bin/env python3
"""
Convert MediaPipe face_landmarker.task models to ONNX for TensorRT.

The .task file is a ZIP containing two TFLite models:
  - face_detector.tflite       -> face_detector.onnx
  - face_landmarks_detector.tflite -> face_landmarks.onnx

Usage:
  pip install tf2onnx tensorflow-lite flatbuffers
  python3 convert_models.py face_landmarker.task

The ONNX files will be created in the same directory as the .task file.
Place them alongside the .task file for the mozza_mp_gpu plugin to find.
"""

import os
import sys
import zipfile
import tempfile
import subprocess


def extract_tflite_from_task(task_path: str, output_dir: str) -> dict:
    """Extract .tflite models from the .task ZIP archive."""
    models = {}
    with zipfile.ZipFile(task_path, "r") as zf:
        for entry in zf.namelist():
            print(f"  ZIP entry: {entry}")
            if entry.endswith(".tflite"):
                data = zf.read(entry)
                basename = os.path.basename(entry)
                out_path = os.path.join(output_dir, basename)
                with open(out_path, "wb") as f:
                    f.write(data)
                models[basename] = out_path
                print(f"  Extracted: {basename} ({len(data)} bytes)")
    return models


def tflite_to_onnx(tflite_path: str, onnx_path: str) -> bool:
    """Convert a TFLite model to ONNX using tf2onnx."""
    print(f"  Converting {os.path.basename(tflite_path)} -> {os.path.basename(onnx_path)}...")
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "tf2onnx.convert",
                "--tflite", tflite_path,
                "--output", onnx_path,
                "--opset", "13",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  ERROR: tf2onnx failed:\n{result.stderr}")
            return False
        print(f"  OK: {onnx_path}")
        return True
    except FileNotFoundError:
        print("  ERROR: tf2onnx not found. Install with: pip install tf2onnx")
        return False
    except subprocess.TimeoutExpired:
        print("  ERROR: Conversion timed out after 120s")
        return False


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <face_landmarker.task>")
        sys.exit(1)

    task_path = sys.argv[1]
    if not os.path.exists(task_path):
        print(f"ERROR: File not found: {task_path}")
        sys.exit(1)

    output_dir = os.path.dirname(os.path.abspath(task_path))
    print(f"Task file: {task_path}")
    print(f"Output dir: {output_dir}")

    # Step 1: Extract TFLite models
    print("\n[1/3] Extracting TFLite models from .task bundle...")
    with tempfile.TemporaryDirectory() as tmpdir:
        models = extract_tflite_from_task(task_path, tmpdir)

        if not models:
            print("ERROR: No .tflite models found in .task file")
            sys.exit(1)

        # Step 2: Convert detector
        print("\n[2/3] Converting face detector...")
        det_tflite = None
        for name in ["face_detector.tflite", "FaceDetectorShortRange.tflite", "detector.tflite"]:
            if name in models:
                det_tflite = models[name]
                break

        if not det_tflite:
            print("WARNING: No face detector .tflite found. Available:", list(models.keys()))
            # Try the first available model
            if models:
                det_tflite = list(models.values())[0]
                print(f"  Using: {os.path.basename(det_tflite)}")

        det_onnx = os.path.join(output_dir, "face_detector.onnx")
        if det_tflite and not tflite_to_onnx(det_tflite, det_onnx):
            print("WARNING: Detector conversion failed")

        # Step 3: Convert landmarks
        print("\n[3/3] Converting face landmarks...")
        lm_tflite = None
        for name in ["face_landmarks_detector.tflite", "FaceLandmarksDetector.tflite", "landmarks.tflite"]:
            if name in models:
                lm_tflite = models[name]
                break

        if not lm_tflite:
            # Try any remaining model
            remaining = [v for k, v in models.items() if v != det_tflite]
            if remaining:
                lm_tflite = remaining[0]
                print(f"  Using: {os.path.basename(lm_tflite)}")

        lm_onnx = os.path.join(output_dir, "face_landmarks.onnx")
        if lm_tflite and not tflite_to_onnx(lm_tflite, lm_onnx):
            print("WARNING: Landmarks conversion failed")

    # Summary
    print("\n=== Summary ===")
    for name in ["face_detector.onnx", "face_landmarks.onnx"]:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            sz = os.path.getsize(path)
            print(f"  OK:   {path} ({sz:,} bytes)")
        else:
            print(f"  FAIL: {path} (not created)")

    print("\nPlace these ONNX files in the same directory as the .task file.")
    print("The mozza_mp_gpu plugin will find them automatically.")
    print("TensorRT will build optimized engines on first run (~30s, cached after).")


if __name__ == "__main__":
    main()
