# Mediapipe plugins in GStreamer

This repository contains three lean GStreamer video filters (`facelandmarks`, `mozza_mp`, and `mozza_mp_gpu`) which run face landmark detection and optional deformations.

- **facelandmarks** (CPU): MediaPipe Face Landmarker (C++ Tasks) on CPU.
- **mozza_mp** (CPU): MediaPipe Face Landmarker + OpenCV MLS warping on CPU.
- **mozza_mp_gpu** (GPU): TensorRT Face Landmarker + CUDA MLS warping on GPU.

We currently recommend running with `ignore-timestamps=false`.
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
The parameters are identical to `mozza_mp`, but the main model flag is renamed for clarity:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path`| string | required | Path to `face_landmarker.task`. |
| `model`     | string | alias    | Alias for `model_path` (for compatibility). |
| `gpu-id`    | int    | 0        | CUDA device index to use. |

**Crucial Note on Models:**
Unlike the CPU version, `mozza_mp_gpu` requires **ONNX** files. The `model_path` points to the `.task` file, but the plugin will look in that **same folder** for `face_detector.onnx` and `face_landmarks.onnx`.

## TensorRT Engine Caching
The first time you run the plugin on a new machine or a different GPU, TensorRT will spend **20-60 seconds** "optimizing" the model for your specific hardware.
- This only happens **once**.
- The result is saved as an `.engine_smXX_fp16` file in the same folder.
- Subsequent runs will load this engine file in milliseconds.

## Setup
1. Extract and convert models:
   ```bash
   python3 convert_models.py face_landmarker.task
   ```
2. Place the generated `.onnx` files alongside the `.task` file.

## Example
```bash
gst-launch-1.0 -v \
  filesrc location=input.mp4 ! decodebin ! videoconvert ! video/x-raw,format=RGBA ! \
  mozza_mp_gpu model_path=/models/face_landmarker.task deform=/models/smile.dfm alpha=1.5 ! \
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
## Build
```bash
docker build -t ducksouplab/mozzamp:latest .
```

This will:
	1.	Install build deps + GStreamer dev headers (to compile a plugin).
	2.	Install Bazelisk (recommended Bazel launcher).
	3.	Clone MediaPipe and build the Face Landmarker C++ target.
	4.	Export headers/libs into third_party/mediapipe-export.
	5.	Build libgstfacelandmarks.so with CMake and install it to the system plugin path.
	6.	Download face_landmarker.task into /opt/models.

## Docker push or pull
Push or pull docker image:

Push — you need writes to ducksouplab to do this:
```bash
docker push ducksouplab/mozzamp:latest
```

Pull :
```bash
docker pull ducksouplab/mozzamp:latest
```

## Verify plugin
```
docker run --rm -it mozzamp:latest \
  bash -lc 'gst-inspect-1.0 mozzamp'
```
You should see properties: model, max-faces, draw, radius, color, delegate, threads.

## Get the .task model

If you haven't:
```
chmod +x download_face_landmarker_model.sh
./download_face_landmarker_model.sh
```

## Get the .so files
```
chmod +x get_so_file.sh
./get_so_file.sh ducksouplab/mozzamp:latest
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
sudo cp dist/face_landmarker.task /home/deploy/deploy-ducksoup/app/plugins/face_landmarker.task
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
You can verify all plugins using the provided automated test script. This script checks for plugin availability and runs a functional test on `test_face.jpg`.

**Inside Docker:**
```bash
docker run --rm --gpus all -v "$PWD:/models" \
  -e GST_PLUGIN_PATH=/models/mp-out/plugins \
  mp_plugins_test:latest \
  bash -c "cd /models && ./test_plugins.sh"
```

The script will generate:
- `test_out_landmarks.png`: Landmarks overlay (CPU)
- `test_out_mozza_cpu.png`: Deformation (CPU)
- `test_out_mozza_gpu.png`: Deformation (GPU)

# Quick runs

## Check gst-inspect-1.0
gst-inspect-1.0 mozzamp

Or with the .so directly:
gst-inspect-1.0 libgstmozzamp.so

## Synthetic input (videotestsrc → mp4 file on host)
```
mkdir -p out
docker run --rm -it -v "$PWD/out:/out" mozzamp:latest bash -lc '
  gst-launch-1.0 -v \
    videotestsrc num-buffers=300 ! video/x-raw,width=640,height=480,framerate=30/1 ! \
    videoconvert ! video/x-raw,format=RGBA ! \
    mozzamp model=/opt/models/face_landmarker.task max-faces=1 threads=4 draw=true landmark-radius=2 landmark-color=0x00FF00FF ! \
    videoconvert ! x264enc tune=zerolatency ! mp4mux ! filesink location=/out/landmarked.mp4
'
```

## Process a host video file
docker run --rm -it -v "$PWD:/work" mozzamp:latest bash -lc '
  gst-launch-1.0 -v \
    filesrc location=/work/input.mp4 ! decodebin ! videoconvert ! video/x-raw,format=RGBA ! \
    mozzamp model=/opt/models/face_landmarker.task max-faces=1 threads=4 ! \
    videoconvert ! x264enc ! mp4mux ! filesink location=/work/output_landmarked.mp4
'

## Element usage
The plugin expects RGBA input; negotiate with videoconvert if needed. It can also
negotiate GPU buffers via `video/x-raw(memory:GLMemory)` and will fall back to CPU
copies when such memory types are unsupported:

```
gst-launch-1.0 -v \
  videotestsrc ! video/x-raw,width=640,height=480 ! \
  videoconvert ! video/x-raw,format=RGBA ! \
  mozzamp model=/opt/models/face_landmarker.task max-faces=1 threads=4 draw=true landmark-radius=2 landmark-color=0x00FF00FF ! \
  fakesink
```

# Internals & notes
	•	Base class: GstVideoFilter with transform_frame_ip is the idiomatic in-place frame hook for video filters. [refs]
	•	Plugin discovery: install to /usr/lib/x86_64-linux-gnu/gstreamer-1.0/ or set GST_PLUGIN_PATH/--gst-plugin-path. [refs]
	•	MediaPipe: we pass frames as ImageFrame(SRGBA) and call DetectForVideo(image, timestamp_ms). The model is a .task bundle downloaded from the official guide. [refs]
	•	Performance: start with 640×480; increase as needed.
	•	GPU: `mozza_mp_gpu` provides native NVIDIA GPU acceleration via TensorRT and CUDA. The CPU plugins (`facelandmarks`, `mozza_mp`) are strictly CPU-based.


# Update MediaPipe version
```
docker build --build-arg MP_REF=v0.10.xx -t mozzamp:latest .
```

# Python Wrapper: mozza_process.py
We provide a Python wrapper that simplifies processing images and videos by automatically managing Docker mounts and GStreamer pipelines.

### Prerequisites
- Python 3
- Docker (with NVIDIA Container Toolkit for GPU mode), prefer CPU if no need of Real Time processing—this will be easier to execute.
- Get the .task model if you haven't
```
chmod +x download_face_landmarker_model.sh
./download_face_landmarker_model.sh
```

- For GPU, see convert models, below.

### Basic Usage
```bash
# Process a video using GPU
python3 mozza_process.py --input assets/video_example.mp4 --output assets/output.mp4 --mode gpu --deform smile.dfm --alpha 2.0

# Process an image using CPU
python3 mozza_process.py --input assets/test_face.jpg --output assets/output.png --mode cpu --deform smile.dfm --alpha 1.5

# Extract landmarks only (green dots)
python3 mozza_process.py --input assets/test_face.jpg --output assets/landmarks.png --mode landmarks --show-landmarks
```

### All Parameters
| Flag | Description |
|------|-------------|
| `--input` | Input image or video file path. |
| `--output`| Output file path (automatically adds extension if missing). |
| `--mode`  | `gpu`, `cpu`, or `landmarks`. |
| `--deform`| Path to your `.dfm` file. |
| `--alpha` | Intensity of the transformation (default 1.0). |
| `--model-path`| Path to `.task` model (default `face_landmarker.task`). |
| `--show-landmarks`| Draw landmarks on the output. |
| `--verbose`| Print the full Docker and GStreamer commands. |

Run `python3 mozza_process.py --help` for the full list of supported parameters.

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


