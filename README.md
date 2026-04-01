# Mediapipe plugins in GStreamer

This repository contains three lean GStreamer video filters (`facelandmarks`, `mozza_mp`, and `mozza_mp_gpu`) which run face landmark detection and optional deformations.

- **facelandmarks** (CPU): MediaPipe Face Landmarker (C++ Tasks) on CPU.
- **mozza_mp** (CPU): MediaPipe Face Landmarker + OpenCV MLS warping on CPU.
- **mozza_mp_gpu** (GPU): TensorRT Face Landmarker + CUDA MLS warping on GPU.

- Base image: `ducksouplab/debian-gstreamer:deb12-with-plugins-cuda12.2-gst1.28.0`
- GStreamer plugin base class: **GstVideoFilter** (`transform_frame_ip`).
- MediaPipe Face Landmarker uses a `.task` model and `LIVE_STREAM` mode.

Note:
- Each plugin has its own set of parameters.
- We currently recommend keeping `ignore-timestamps=false` for typical scenarios.
- Plugins are still under development, use at your own risk.

# Plugin : mozza_mp
mozza_mp is an implementation of ARIAS 2018 using the mediapipe facetracker. It enables, among others, to transform the smiles of individuals in the video streams in real time. It implements a Moving Least Square algorithm using the imgwarp library. This enables the user to do other types of transformations than just smile manipulation, such as face morphology manipulation.

Reference : Arias, P., Soladie, C., Bouafif, O., Roebel, A., Seguier, R., & Aucouturier, J. J. (2018). Realistic transformation of facial and vocal smiles in real-time audiovisual streams. IEEE Transactions on Affective Computing, 11(3), 507-518.


## Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Path to `face_landmarker.task`. |
| `deform` | string | none | Path to deformation `.dfm` file. |
| `dfm` | string | none | Alias for `deform`. |
| `alpha` | float [-10..10] | 1.0 | Smile intensity multiplier (negative values frown). |
| `mls-alpha` | float | 1.4 | Rigidity parameter for MLS warping. |
| `mls-grid` | int | 5 | MLS grid size in pixels (smaller = denser). |
| `warp-mode` | string | global | MLS warp strategy: `global` or `per-group-roi`. |
| `roi-pad` | int [0..200] | 24 | Padding around group ROI when `warp-mode=per-group-roi` (pixels). |
| `overlay` | boolean | false | Draw source/destination control points and vectors. |
| `drop` | boolean | false | Drop frame when no face is detected. |
| `show-landmarks` | boolean | false | Draw all detected landmarks (uses `landmark-radius` and `landmark-color`). |
| `landmark-radius` | int | 2 | Radius of landmark dots in pixels (when `show-landmarks=true`). |
| `landmark-color` | uint | 0x0066CCFF | Packed RGBA color for landmarks (default: blue). |
| `max-faces` | int | 1 | Maximum number of faces to detect. |
| `threads` | int | 4 | Number of CPU threads for MediaPipe (0=system default). |
| `strict-dfm` | boolean | false | Fail to start if a deform file is given but cannot be loaded. |
| `force-rgb` | boolean | false | No-op; pads already require RGBA. |
| `ignore-timestamps` | boolean | false | Pass `0` as timestamp to the detector; recommended to keep `false`. |
| `log-every` | uint | 60 | Emit a log message every N frames (0 disables). |
| `user-id` | string | none | Accepted but ignored; useful for uniform configs. |

Recommended invocation:

`mozza_mp model=/app/plugins/face_landmarker.task deform=/app/plugins/smile_corners_only.dfm alpha=1.7 threads=4 ignore-timestamps=false warp-mode=per-group-roi`

When using DuckSoup, and transforming smiles in a server live, we recommend:
mozza_mp deform=/app/plugins/smile_mp.dfm alpha=2 model=/app/plugins/face_landmarker.task warp-mode=per-group-roi

With smile_mp.dfm such as:
```
# Left corner (61): use two upper-lip/cheek points near-above it (146 and 91)
0,61,   146,  91,  61,   -0.55, -0.55,  2.10

# Right corner (291): mirrors (375 and 321)
1,291,  375, 321, 291,   -0.55, -0.55,  2.10
```

Tweak mls-alpha to modulate the effect—between 0.8 and 1.4 works well for a realistic smile. Tweak also alpha for the intensity parameter.

## DFM file format
Each non-comment line defines one control rule:
group, index,  t0, t1, t2,  a, b, c
	- group – Integer group id. Rows with the same id form one group (used by warp-mode).
	- index – Landmark to move.
	- t0,t1,t2 – Anchor landmark indices used to build a barycentric target point T = a·L[t0] + b·L[t1] + c·L[t2].
Weights need not sum to 1; negative weights are allowed (extrapolation), subject to your use.
	- a,b,c – Barycentric weights.

  For each rule, the destination of index is:
  ```dst = cur + alpha * (T - cur)````

where cur is the current landmark position and alpha is the element’s global intensity.

## Global vs local mode
Groups & warp behavior
	•	In global mode, all groups are merged into one control set and warped once over the full frame.
	•	In per-group-roi mode, each group is warped independently inside a tight crop (union of src/dst bounding boxes, expanded by roi-pad). This confines influence to the region and prevents far-field “banding”.

## Tips
	•	Prefer local anchors (nearby landmarks) to keep directions stable across faces.
	•	If moving exactly one point in a group, add 1–3 identity pins (e.g., p,p,p, 1,0,0) around it so MLS has ≥2 control points.
	•	If a rule references an out-of-range landmark index, that row is skipped (recommended behavior); enable strict-dfm during development to catch mistakes early.


# Plugin : mozza_mp_gpu (GPU Accelerated)
`mozza_mp_gpu` is a high-performance, GPU-accelerated drop-in replacement for `mozza_mp`. It uses **TensorRT** for two-stage face detection and landmarking, and **CUDA** kernels for the Moving Least Square (MLS) warping.

**Key advantages:**
- **Performance:** ~10ms total per frame (vs ~40ms on multitreaded CPU).
- **Precision:** Uses TensorRT FP16 inference for both detection and 478 landmarks.
- **Parallelism:** MLS warping is performed directly on the GPU using custom CUDA kernels.

## Parameters
`mozza_mp_gpu` supports the same parameters as `mozza_mp` plus GPU-specific ones:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | required | Path to `face_landmarker.task`. Also accepted as `model`. |
| `deform` | string | none | Path to deformation `.dfm` file. |
| `alpha` | float [-10..10] | 1.0 | Deformation intensity multiplier. |
| `mls-alpha` | float | 1.4 | Rigidity parameter for MLS warping. |
| `mls-grid` | int | 5 | MLS grid size in pixels (smaller = denser). |
| `warp-mode` | int | 0 | MLS warp strategy: `0`=global, `1`=per-group-roi. |
| `roi-pad` | int [0..128] | 24 | Padding around group ROI when `warp-mode=1` (pixels). |
| `no-warp` | boolean | false | Run landmark detection only, skip MLS warping. |
| `show-landmarks` | boolean | false | Draw all detected landmarks on the frame. |
| `drop` | boolean | false | Drop frame when no face is detected. |
| `max-faces` | int | 1 | Maximum number of faces to detect. |
| `gpu-id` | int | 0 | CUDA device index to use. |
| `smooth` | float [0..0.99] | 0.5 | Temporal EMA smoothing on ROI (0=off). |
| `min-cutoff` | float | 0.5 | OneEuroFilter min_cutoff: lower = more smoothing at rest (less jitter, more lag). |
| `smooth-landmarks` | boolean | true | Apply OneEuroFilter smoothing to landmarks. |
| `log-every` | uint | 60 | Emit a log message every N frames (0 disables). |
| `strict-dfm` | boolean | false | Fail to start if a deform file cannot be loaded. |
| `user-id` | string | none | Accepted but ignored; useful for uniform configs. |

**Crucial Note on Models:**
Unlike the CPU version, `mozza_mp_gpu` requires **ONNX** files. The `model_path` points to the `.task` file, but the plugin will look in that **same folder** for `face_detector.onnx` and `face_landmarks.onnx`.

## TensorRT Engine Caching
The first time you run the plugin on a new machine or GPU, TensorRT will spend **20-60 seconds** building an optimised engine.
- This only happens **once per GPU architecture**.
- The engine is cached as `<model>.engine_smXX_fp16` alongside the ONNX files.
- Subsequent runs load the cache in milliseconds.
- If you change the ONNX models, delete the `.engine_*` files to force a rebuild.

## Setup
1. Obtain the ONNX models (`face_detector.onnx`, `face_landmarks.onnx`) by converting from the `.task` file.
2. Place them alongside the `.task` file (same directory).
3. On first run TensorRT will build and cache an `.engine_smXX_fp16` file (~20-60s). Subsequent runs load the cache instantly.

## Example
```bash
gst-launch-1.0 -v \
  filesrc location=input.mp4 ! decodebin ! videoconvert ! video/x-raw,format=RGBA ! \
  mozza_mp_gpu model_path=/models/face_landmarker.task deform=/models/smile.dfm alpha=1.5 warp-mode=1 ! \
  videoconvert ! autovideosink
```

# Plugin : facelandmarks

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Path to `face_landmarker.task`. |
| `max-faces` | int | 1 | Maximum number of faces to detect. |
| `draw` | boolean | true | Overlay landmarks on the frame. |
| `radius` | int | 2 | Radius of landmark dots in pixels. |
| `color` | uint | 0x00FF00FF | Packed RGBA color for landmarks (default: green). |
| `threads` | int | 4 | Number of CPU threads for MediaPipe (0=system default). |

## Example

Usage example:
```bash
facelandmarks model=/app/plugins/face_landmarker.task threads=6
```

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

## Move the .so to DuckSoup
If you are using DuckSoup, make sure to remove old plugins and copy the new .so files to 
the correct destination so taht DuckSouop can see them. In a deploy user it looks like 
this (change with the appropriate path where you use your plugins): 
``` 
## Careful with this line : this removes old plugins from destination
##sudo rm -r /home/deploy/deploy-ducksoup/app/plugins/mp_plugins/

## Copy the new plugins to destination, as well as the face_landmarker.task files
sudo cp -r mp-out /home/deploy/deploy-ducksoup/app/plugins/mp_plugins
sudo chown -R deploy:deploy /home/deploy/deploy-ducksoup/app/plugins/mp_plugins

sudo cp dist/face_landmarker.task /home/deploy/deploy-ducksoup/app/plugins/face_landmarker.task
sudo cp face_landmarks.onnx /home/deploy/deploy-ducksoup/app/plugins/face_landmarks.onnx
sudo cp face_detector.onnx /home/deploy/deploy-ducksoup/app/plugins/face_detector.onnx
```

## ImgWarp debug logs

The underlying ImgWarp library can emit verbose diagnostics. These logs are
disabled by default. To enable them, set the `IMGWARP_DEBUG` environment
variable to a non-zero value before running any GStreamer command, for example:

```bash
IMGWARP_DEBUG=1 gst-launch-1.0 ...
```

The messages are written to standard error and can help troubleshoot warping
issues.

## Use the plugins in DuckSoup in mirror mode

Example:

mozza_mp model=/app/plugins/face_landmarker.task deform=/app/plugins/smile_corners_only.dfm alpha=1.7 threads=4 ignore-timestamps=false

If you run this in within docker and are using the CPU version, and want to improve face detection times by using multihtraeding, make sure to add this to your docker compose file:
```bash
    environment:
		.... all your other options...
      - OMP_NUM_THREADS=4
      - XNNPACK_NUM_THREADS=4
      - TFLITE_NUM_THREADS=4
```

# Testing
You can verify all plugins using the provided automated test script. This script checks for plugin availability and runs a functional test on a still image.

**Inside Docker:**
```bash
docker run --rm --gpus all -v "$PWD:/work" \
  -e GST_PLUGIN_PATH=/work/mp-out/plugins \
  -e LD_LIBRARY_PATH=/work/mp-out/lib:/opt/gstreamer/lib/x86_64-linux-gnu \
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
docker run --rm \
  -v "$PWD:/work" -v "$PWD/env:/models" \
  mp_plugins:latest \
  gst-launch-1.0 -q \
    filesrc location=/work/assets/video_example.mp4 ! decodebin ! videoconvert ! \
    video/x-raw,format=RGBA ! \
    mozza_mp model=/models/face_landmarker.task deform=/work/smile.dfm alpha=1.5 \
      warp-mode=per-group-roi ! \
    videoconvert ! x264enc tune=zerolatency ! mp4mux ! \
    filesink location=/work/output_cpu.mp4
```

## Process a video with GPU (deformation)
```bash
docker run --rm --gpus all \
  -v "$PWD:/work" -v "$PWD/env:/models" \
  mp_plugins:latest \
  gst-launch-1.0 -q \
    filesrc location=/work/assets/video_example.mp4 ! decodebin ! videoconvert ! \
    video/x-raw,format=RGBA ! \
    mozza_mp_gpu model_path=/models/face_landmarker.task deform=/work/smile.dfm alpha=1.5 \
      warp-mode=1 ! \
    videoconvert ! x264enc tune=zerolatency ! mp4mux ! \
    filesink location=/work/output_gpu.mp4
```

# Internals & notes
	•	Base class: GstVideoFilter with transform_frame_ip is the idiomatic in-place frame hook for video filters. [refs]
	•	Plugin discovery: install to /usr/lib/x86_64-linux-gnu/gstreamer-1.0/ or set GST_PLUGIN_PATH/--gst-plugin-path. [refs]
	•	MediaPipe: we pass frames as ImageFrame(SRGBA) and call DetectForVideo(image, timestamp_ms). The model is a .task bundle downloaded from the official guide. [refs]
	•	Performance: start with 640×480; increase as needed.
	•	GPU: `mozza_mp_gpu` provides native NVIDIA GPU acceleration via TensorRT and CUDA. The CPU plugins (`facelandmarks`, `mozza_mp`) are strictly CPU-based.


# Update MediaPipe version
Change `ARG MEDIAPIPE_TAG=v0.10.26` at the top of `Dockerfile` then rebuild:
```bash
DOCKER_BUILDKIT=1 docker build --no-cache -t mp_plugins:latest .
```

# Python Wrapper: mozza_process.py
We provide a Python wrapper that simplifies processing images and videos by automatically managing Docker mounts and GStreamer pipelines.

### Prerequisites
- Python 3
- Docker (with NVIDIA Container Toolkit for GPU mode).
- Get the .task model if you haven't:
```bash
chmod +x download_face_landmarker_model.sh
./download_face_landmarker_model.sh
```

- **For GPU:** You also need the ONNX models (`face_detector.onnx`, `face_landmarks.onnx`) in the same directory as the `.task` file.

### Basic Usage
```bash
# Process a video using GPU (deformation, alpha=2.0, no landmark overlay)
python3 mozza_process.py --input assets/video_example.mp4 --output output/gpu_smile.mp4 \
  --mode gpu --deform smile.dfm --alpha 2.0 --warp-mode per-group-roi --show-landmarks false

# Process a video using CPU (deformation)
python3 mozza_process.py --input assets/video_example.mp4 --output output/cpu_smile.mp4 \
  --mode cpu --deform smile.dfm --alpha 2.0 --warp-mode per-group-roi --show-landmarks false

# Show landmark tracking only (no deformation)
python3 mozza_process.py --input assets/video_example.mp4 --output output/landmarks.mp4 \
  --mode gpu --no-warp true --show-landmarks true
```

### Key Parameters
| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Input image or video file path. |
| `--output` | required | Output file path. |
| `--mode` | `gpu` | `gpu`, `cpu`, or `landmarks`. |
| `--docker-image` | `mp_plugins:latest` | Docker image to use. |
| `--deform` | none | Path to `.dfm` deformation file. |
| `--alpha` | 1.0 | Deformation intensity multiplier. |
| `--warp-mode` | `global` | `global` or `per-group-roi` (recommended for parity). |
| `--show-landmarks` | `true` | `true` or `false` — draw landmark dots on output. |
| `--no-warp` | `false` | `true` to skip MLS warping (landmark tracking only). |
| `--model-path` | `face_landmarker.task` | Path to `.task` model. |
| `--smooth` | 0.5 | Temporal smoothing on ROI (GPU only, 0=off). |
| `--verbose` | false | Print the full Docker and GStreamer commands. |

Run `python3 mozza_process.py --help` for the full list of supported parameters.

# Tutorial
For a step-by-step guide on how to use these plugins from scratch using Python, check out our [Tutorial Notebook](tutorial/tutorial.ipynb).

# References
• MediaPipe Face Landmarker task & models (C++/Tasks, .task bundle, running modes).
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker  ← [official]

• GStreamer GstVideoFilter API (video filter base class).
https://gstreamer.freedesktop.org/documentation/video/gstvideofilter.html

• GStreamer plugin discovery (GST_PLUGIN_PATH, --gst-plugin-path).
https://gstreamer.freedesktop.org/documentation/tools/gst-launch.html
https://gstreamer.freedesktop.org/documentation/plugin-development/basics/testapp.html

• Bazelisk (recommended Bazel launcher).
https://bazel.build/install/bazelisk


# Developer Tips & Lessons Learned
### Testing with GStreamer 1.28.0
Always test and run using the base image specified in the Dockerfiles:
`ducksouplab/debian-gstreamer:deb12-with-plugins-cuda12.2-gst1.28.0`

Run an interactive debugging shell:
```bash
docker run --rm -it --gpus all \
  -v "$PWD:/work" -w /work \
  -e GST_PLUGIN_PATH=/work/mp-out/plugins \
  -e LD_LIBRARY_PATH=/work/mp-out/lib:/opt/gstreamer/lib/x86_64-linux-gnu \
  ducksouplab/debian-gstreamer:deb12-with-plugins-cuda12.2-gst1.28.0 bash
```

### Plugin Naming & Symbols
GStreamer's dynamic loader requires the `GST_PLUGIN_DEFINE` macro name to match the exported symbol in the `.so` file.
- **Filename:** `libgstmozzamp_gpu.so`
- **Plugin Name:** `mozzamp_gpu` (must match the filename suffix)
- **Verified via:** `nm -D libgstmozzamp_gpu.so | grep gst_plugin_`

### Dependency Management
The GPU plugin requires `libcudart.so.12` and TensorRT libraries. When running tests outside the final production image, ensure `LD_LIBRARY_PATH` includes:
1. `/work/mp-out/lib` (for `libmp_runtime.so`)
2. `/usr/local/cuda/lib64` (for CUDA)
3. `/opt/gstreamer/lib/x86_64-linux-gnu` (for the correct GStreamer 1.28.0 libraries)

### Performance & Cache
TensorRT engines are cached as `.engine_smXX_fp16` files. If you modify the ONNX models or tensor layouts, **delete these files** from your host directory before the next run to force a clean reconstruction by the TensorRT builder.


