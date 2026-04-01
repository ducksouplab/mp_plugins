# Mediapipe GStreamer Plugins

High-performance GStreamer plugins for real-time face landmark detection and facial geometry transformation (warping). Built for both edge devices (CPU) and high-density GPU servers.

---

## 🚀 Quick Start
If you are new to this project, start with our **[Tutorial Notebook](tutorial/tutorial.ipynb)**. It guides you through using our pre-built Docker image to transform images and videos with just a few lines of Python.

---

## What is this?
This repository provides three primary GStreamer filters:

### 1. `facelandmarks` (CPU)
A lightweight overlay that detects 478 face landmarks and draws them on the video stream. Useful for verifying that the AI correctly "sees" the face before applying deformations.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | string | required | Path to the `.task` model file. |
| `max-faces` | int | 1 | Maximum number of faces to detect. |
| `draw` | boolean | true | Whether to draw the landmark dots. |
| `radius` | int | 2 | Radius of the landmark dots in pixels. |
| `color` | string | 0x0066CCFF | Hex RGBA color of the dots. |
| `threads` | int | 4 | Number of CPU threads for MediaPipe. |

### 2. `mozza_mp` (CPU)
A CPU-optimized transformer that uses MediaPipe and OpenCV's Moving Least Squares (MLS) to realistically deform facial expressions using rule-based `.dfm` files.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | string | required | Path to the `.task` model file. |
| `deform` | string | none | Path to the `.dfm` rule file. |
| `alpha` | float | 1.0 | Intensity multiplier for the deformation. |
| `mls-alpha` | float | 1.4 | MLS rigidity (higher = stiffer skin). |
| `mls-grid` | int | 5 | Grid size for warping calculation. |
| `warp-mode` | string | global | `global` or `per-group-roi` (recommended). |
| `roi-pad` | int | 24 | Padding around facial groups in ROI mode. |
| `show-landmarks` | boolean | false | Draw landmarks over the deformed image. |

### 3. `mozza_mp_gpu` (GPU)
A high-performance version of the transformer using NVIDIA TensorRT and custom CUDA kernels, achieving ~10x speedup over the CPU version.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `model_path`| string | required | Path to the `.task` model file. |
| `deform` | string | none | Path to the `.dfm` rule file. |
| `alpha` | float | 1.0 | Intensity multiplier for the deformation. |
| `mls-alpha` | float | 1.4 | MLS rigidity (higher = stiffer skin). |
| `mls-grid` | int | 5 | Grid size for warping calculation. |
| `warp-mode` | int | 0 | `0`=global, `1`=per-group-roi. |
| `roi-pad` | int | 24 | Padding around facial groups in ROI mode. |
| `smooth` | float | 0.5 | High-level temporal smoothing factor. |
| `min-cutoff`| float | 1.0 | OneEuroFilter min_cutoff (lower = less jitter). |
| `beta` | float | 0.01 | OneEuroFilter beta (higher = less lag). |
| `show-landmarks`| boolean | false | Draw landmarks over the deformed image. |
| `gpu-id` | int | 0 | CUDA device index. |

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

## DuckSoup usage

If running these plugins within DuckSoup, copy the .so files to your DuckSoup plugin repository.
```bash
# First remove old .so files from your path, for instance, if you are using a deploy user to run ducksoup, something like:
# Make sure that you don't need the files, this will remove the files from your computer!
sudo rm -r /home/deploy/deploy-ducksoup/app/plugins/mp_plugins

#Now copy the new files:
sudo cp -r mp-out/plugins /home/deploy/deploy-ducksoup/app/plugins/mp_plugins

#Also copy the required models:
sudo cp face_landmarker.task /home/deploy/deploy-ducksoup/app/plugins/face_landmarker.task
sudo cp face_detector.onnx /home/deploy/deploy-ducksoup/app/plugins/face_detector.onnx
sudo cp face_landmarks.onnx /home/deploy/deploy-ducksoup/app/plugins/face_landmarks.onnx

# Copy the dfm if needed
sudo cp smile.dfm /home/deploy/deploy-ducksoup/app/plugins/smile_mp.dfm

#Copy shared library
sudo cp -r mp-out/lib /home/deploy/deploy-ducksoup/app/plugins/mp_plugins/lib
```

Now you can use the plugin within ducksoup using the following arguments:
```bash

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
If you use this work in your, please cite:

```text
Arias, P., Soladie, C., Bouafif, O., Roebel, A., Seguier, R., & Aucouturier, J. J. (2018). Realistic transformation of facial and vocal smiles in real-time audiovisual streams. IEEE Transactions on Affective Computing, 11(3), 507-518.

Arias-Sarah, P., Denis, G., Hall, L., Aucouturier, J. J., Schyns, P. G., Jack, R. E., & Johansson, P. DuckSoup: a videoconference experimental platform to transform participants’ voice and face in real-time during social interactions.

Retrieved from https://github.com/ducksouplab/mp_plugins
```

- **MediaPipe**: [Face Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- **GStreamer**: [GstVideoFilter API](https://gstreamer.freedesktop.org/documentation/video/gstvideofilter.html)
