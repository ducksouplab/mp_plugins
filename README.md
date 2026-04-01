# Mediapipe GStreamer Plugins

High-performance GStreamer plugins for real-time face landmark detection and facial geometry transformation (warping). Built for both edge devices (CPU) and high-density GPU servers.

---

## 🚀 Quick Start
If you are new to this project, start with our **[Tutorial Notebook](tutorial/tutorial.ipynb)**. It guides you through using our pre-built Docker image to transform images and videos with just a few lines of Python.

---

## What is this?
This repository provides three primary GStreamer filters:
1.  **`facelandmarks`**: A lightweight overlay that detects 478 face landmarks and draws them on the video stream.
2.  **`mozza_mp`**: A CPU-optimized transformer that uses MediaPipe and OpenCV's Moving Least Squares (MLS) to realistically deform facial expressions.
3.  **`mozza_mp_gpu`**: A high-performance version of the transformer using NVIDIA TensorRT and custom CUDA kernels, achieving ~10x speedup over the CPU version.

## How it works
The project uses a two-stage pipeline:
- **Stage 1 (Detection)**: Uses a BlazeFace SSD model to locate the face and primary keypoints.
- **Stage 2 (Landmarking)**: Crops the face and runs a high-resolution regressor to find all 478 landmarks.
- **Transformation**: Uses rule-based `.dfm` files to map source landmarks to target destinations, creating effects like smiles, frowns, or morphology changes via MLS warping.

## Usage Modes
- **Within GStreamer**: Use these plugins as standard elements in your pipelines (e.g., `... ! mozza_mp_gpu model=... ! ...`).
- **Raw Video Transformation**: Use our Python wrapper `mozza_process.py` to transform existing `.mp4` or `.jpg` files without writing GStreamer code.

---

# Build the plugins with Docker

The build is a multi-stage process that compiles all plugins and assembles the final runtime image.

## Build the image
```bash
DOCKER_BUILDKIT=1 docker build -t mp_plugins:latest .
```

## (Optional) Export build artifacts to host
If you need the `.so` files locally (e.g., for DuckSoup deployment), you can export them:
```bash
DOCKER_BUILDKIT=1 docker build --target artifacts --output type=local,dest=mp-out .
```
This will place the plugins in `mp-out/plugins/` and libraries in `mp-out/lib/`.

## Verify plugins
```bash
docker run --rm --gpus all mp_plugins:latest gst-inspect-1.0 mozza_mp_gpu
```

## Get the .so files from an existing image
```bash
chmod +x get_so_file.sh
./get_so_file.sh mp_plugins:latest
```

# Testing
We provide a script to verify all plugins are working correctly.

**Inside Docker:**
```bash
docker run --rm --gpus all -v "$PWD:/work" \
  mp_plugins:latest \
  bash -c "cd /work && ./test_plugins.sh"
```

The script will generate:
- `test_out_landmarks.png`: Landmarks overlay (CPU facelandmarks)
- `test_out_mozza_cpu.png`: Deformation (CPU mozza_mp)
- `test_out_mozza_gpu.png`: Deformation (GPU mozza_mp_gpu)

### Batch Processing
You can regenerate all transformations for all assets in the `assets/` folder by running:
```bash
./generate_all_outputs.sh
```
The results will be stored in the `output/` directory.

# Quick runs

## Check gst-inspect-1.0
```bash
docker run --rm --gpus all mp_plugins:latest gst-inspect-1.0 mozza_mp_gpu
```

## Process a video with CPU (deformation)
```bash
python3 mozza_process.py --input assets/video_example.mp4 --output output/cpu_smile.mp4 \
  --mode cpu --deform smile.dfm --alpha 1.5 --warp-mode per-group-roi --show-landmarks false
```

## Process a video with GPU (deformation)
```bash
python3 mozza_process.py --input assets/video_example.mp4 --output output/gpu_smile.mp4 \
  --mode gpu --deform smile.dfm --alpha 1.5 --warp-mode per-group-roi --show-landmarks false
```

# References & Citation
If you use this work in your research or product, please cite:

```text
DuckSoup Lab. (2026). MediaPipe GStreamer Plugins for Facial Transformation. 
Retrieved from https://github.com/ducksouplab/mp_plugins
```

- **MediaPipe**: [Face Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- **GStreamer**: [GstVideoFilter API](https://gstreamer.freedesktop.org/documentation/video/gstvideofilter.html)
