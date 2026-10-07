# Mediapipe GStreamer Plugins

High-performance GStreamer plugins for real-time face landmark detection and facial geometry transformation (warping). Built for both edge devices (CPU) and high-density GPU servers.

---

## 🚀 Quick Start
If you are new to this project, start with our **[Tutorial Notebook](tutorial/tutorial.ipynb)**. It guides you through using our pre-built Docker image to transform images and videos with just a few lines of Python.

---

## What is this?
This repository provides four GStreamer filters:

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
| `warp-mode` | int | 0 | MLS warp strategy: `0`=global, `1`=per-group-roi (recommended). |
| `roi-pad` | int | 24 | Padding around facial groups in ROI mode. |
| `smooth` | float | 0.5 | High-level temporal smoothing factor. |
| `min-cutoff`| float | 2.0 | OneEuroFilter min_cutoff (lower = less jitter). |
| `beta` | float | 0.05 | OneEuroFilter beta (higher = less lag). |
| `show-landmarks`| boolean | false | Draw landmarks over the deformed image. |
| `gpu-id` | int | 0 | CUDA device index. |

### 4. `mozza_mesh` (CPU)
A transformer that deforms the face with **displacement fields** (one per transformation) and a **mesh warp**, instead of `.dfm` rules and MLS. It ships with:
- **Facial action units:** AU1, AU2, AU4, AU5, AU7, AU12, AU15, AU20, AU43 (`au_basis_v1.json`).
- **Face morphology:** perceived trustworthiness, dominance and threat, from Oosterhof & Todorov (2008) (`trait_basis_v1.json`).

Each transformation has its own amplitude. Amplitudes can be changed live from DuckSoup (`controlFx`), and any number of them combine. `mozza_mp` and `mozza_mp_gpu` are unchanged and independent of it.

**How it differs from `mozza_mp`:**
| | `mozza_mp` / `mozza_mp_gpu` | `mozza_mesh` |
|---|---|---|
| A deformation is defined by | a `.dfm` rule file (landmarks move towards combinations of other landmarks) | a basis file: how every landmark moves at amplitude 1, for each transformation |
| Warp | MLS (moving least squares) | piecewise-affine on MediaPipe's 898-triangle face mesh: every landmark lands exactly where the field puts it |
| Live control | one `alpha` | one amplitude per transformation (`au12`, `dom`, ...), combinable |
| Eyes and brows | MLS can ripple or pinch | clean |
| Background | moves in `global` mode | untouched beyond a ring around the face |

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | string | required | Path to the `.task` model file. |
| `basis` | string | none | Comma-separated basis JSON files (e.g. `au_basis_v1.json,trait_basis_v1.json`). Later files override fields with the same name. Can be changed live. |
| `au1` `au2` `au4` `au5` `au7` `au12` `au15` `au20` `au43` | float | 0 | Amplitude of each action unit. 1 = a clear, plausible expression; negative values reverse it (`au43=-1` widens the eyes). Range -10..10, sensible within ±1.5. |
| `trust` `dom` `threat` | float | 0 | Amplitude of `TRUST_o`, `DOM_o` (an orthogonal pair, recommended) and `THREAT`, in SDs of Oosterhof & Todorov's models. Sensible within ±3. |
| `amplitudes` | string | "" | All amplitudes at once, for any field in the loaded bases: `"AU12=0.5,DOM_o=2,TRUST=-1"`. Unlisted fields are set to 0. |
| `fold-guard` | float | 5 | If a combination would fold the skin over itself (area in px²), the deformation is scaled down and recovers gradually (~0.7 s). Negative = off. |
| `smooth-landmarks` | boolean | true | OneEuro temporal smoothing of the landmarks (less jitter in video). |
| `min-cutoff` | float | 2.0 | OneEuroFilter min_cutoff (lower = less jitter). |
| `beta` | float | 0.05 | OneEuroFilter beta (higher = less lag). |
| `show-landmarks` | boolean | false | Draw the detected landmarks. |
| `landmark-radius` / `landmark-color` | int / uint | 2 / 0x00FF00FF | Landmark dot size and RGBA color. |
| `no-warp` | boolean | false | Detect only, don't deform. |
| `drop` | boolean | false | Drop frames without a face. |
| `ignore-timestamps` | boolean | false | Use the frame count as detector timestamps. |
| `threads` | int | 4 | Number of CPU threads for MediaPipe. |
| `max-faces` | int | 1 | Faces detected; only the first is transformed. |
| `log-every` | uint | 60 | Timing log every N frames (`GST_DEBUG=mozza_mesh:4`). |
| `user-id` | string | | Accepted for DuckSoup configs; unused. |

The named properties and `amplitudes` write to the same amplitudes: `au12=0.8` is the same as `amplitudes="AU12=0.8"`. All amplitudes at 0 leave the frame unchanged.

**Example pipeline:**
```bash
gst-launch-1.0 filesrc location=face.jpg ! jpegdec ! videoconvert ! video/x-raw,format=RGBA ! \
  mozza_mesh model=face_landmarker.task \
             basis=gstmozzamesh/bases/au_basis_v1.json,gstmozzamesh/bases/trait_basis_v1.json \
             au12=0.8 dom=2 ! \
  videoconvert ! pngenc ! filesink location=out.png
```

**From Python** (runs the plugin in Docker, like `mozza_process.py`):
```python
from mesh_process import transform_image, transform_video, side_by_side_video
transform_image("assets/test_image.jpg", "smile.png", {"AU12": 0.8})
keys = [(0, {"DOM_o": -3}), (2.6, {"DOM_o": 3})]  # dominance sweeps from -3 to +3 SD
transform_video("assets/video_example.mp4", "out.mp4", keyframes=keys)
side_by_side_video("assets/video_example.mp4", "out.mp4", "compare.mp4", keyframes=keys)  # original | transformed, labelled
```

**Basis files** (`gstmozzamesh/bases/`): each file is a set of named fields. A field gives, for each of MediaPipe's 468 face-mesh landmarks, its displacement `[dx, dy]` at amplitude 1, in face units (relative to the line between the outer eye corners), so it works for any face size, position and in-plane rotation. Every frame, the plugin computes `displacement = Σ amplitude × field`, converts it to pixels and warps. New or improved transformations are new basis files (v2, ...), with no change to the plugin code. How the v1 fields were built and validated is documented in [face-transforms](https://github.com/Pablo-Arias/face-transforms).

> **Attribution:** the bases are for non-commercial research. Cite RAVDESS (Livingstone & Russo, 2018) for the AU basis and Oosterhof & Todorov (2008) for the trait basis (full references in [gstmozzamesh/bases/README.md](gstmozzamesh/bases/README.md)). The files contain only averaged landmark displacements, no images or video from those databases.

**More:** [gstmozzamesh/README.md](gstmozzamesh/README.md) (design, development build, tests) · **Tutorial:** [tutorial/mozza_mesh_tutorial.ipynb](tutorial/mozza_mesh_tutorial.ipynb)

**Not yet available:** a GPU version (`mozza_mesh_gpu`).

## How it works
The project uses a two-stage pipeline:
- **Stage 1 (Detection)**: Uses a BlazeFace SSD model to locate the face and primary keypoints.
- **Stage 2 (Landmarking)**: Crops the face and runs a high-resolution regressor to find all 478 landmarks.
- **Transformation**:
  - `mozza_mp` / `mozza_mp_gpu`: rule-based `.dfm` files map source landmarks to target destinations, creating effects like smiles, frowns, or morphology changes via MLS warping.
  - `mozza_mesh`: displacement fields from basis files, scaled by their amplitudes and summed, move every landmark; the image follows with a piecewise-affine warp on the face mesh.

---

## DFM file format
Each non-comment line in a `.dfm` file defines one control rule:
`group, index, t0, t1, t2, a, b, c`

- **`group`**: Integer group ID. Rows with the same ID form one group (used by `warp-mode=per-group-roi`).
- **`index`**: The landmark index to move (0-477).
- **`t0, t1, t2`**: Anchor landmark indices used to build a reference target point.
- **`a, b, c`**: Weights for the anchors.

The destination for the landmark is calculated as:
`Target = a*L[t0] + b*L[t1] + c*L[t2]`
`Final_Destination = Current + alpha * (Target - Current)`

### Example: `smile.dfm`
```text
# Left corner (61): use two upper-lip/cheek points near-above it (146 and 91)
0, 61,   146,  91,  61,   -0.55, -0.55,  2.10

# Right corner (291): mirror points (375 and 321)
1, 291,  375, 321, 291,   -0.55, -0.55,  2.10
```

---

## Global vs Local ROI Mode
- **Global (`warp-mode=global`)**: All deformation rules are merged and applied to the entire frame at once. This is simple but can cause background "bending" if landmarks are near the image edge.
- **Local (`warp-mode=per-group-roi`)**: Each group of rules is processed independently inside a small, tight crop (ROI) around the affected landmarks. This ensures that the deformation **only** affects the face and keeps the rest of the image perfectly still. **Recommended for production.**

---

## 📊 Performance & Latency
These plugins are highly optimized for real-time usage. Below are typical latencies measured on a modern workstation (NVIDIA RTX A5000 + Intel Xeon):

| Mode | Total Latency | Key Steps | Max FPS |
|------|---------------|-----------|---------|
| **GPU (`mozza_mp_gpu`)** | **~1.6 ms** | Detect: 1.3ms, Warp: <0.1ms | ~600+ |
| **CPU (`mozza_mp`)** | **~35.0 ms** | Detect: 35ms, Warp: <0.1ms | ~28 |

> **Note:** GPU performance includes TensorRT inference and custom CUDA kernels. CPU performance is limited by the MediaPipe TFLite inference speed on a single core (by default).

### How to measure latency
You can see live timing statistics by enabling `GST_INFO` and setting a `log-every` interval:
```bash
# For GPU
GST_DEBUG=mozza_mp_gpu:4 python3 mozza_process.py --input assets/video_example.mp4 --output /dev/null --mode gpu --log-every 60

# For CPU
GST_DEBUG=mozza_mp:4 python3 mozza_process.py --input assets/video_example.mp4 --output /dev/null --mode cpu --log-every 60
```

---
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

# For mozza_mesh, copy the basis files
sudo cp gstmozzamesh/bases/*.json /home/deploy/deploy-ducksoup/app/plugins/

#Copy shared library
sudo cp -r mp-out/lib /home/deploy/deploy-ducksoup/app/plugins/mp_plugins/lib
```

Now you can use the plugin within ducksoup using the following arguments:
```bash
mozza_mp_gpu deform=/app/plugins/smile_mp.dfm alpha=2 model=/app/plugins/face_landmarker.task warp-mode=1
```

For `mozza_mesh`, name the effect and control its amplitudes with the player API:
```js
videoFx: "mozza_mesh model=/app/plugins/face_landmarker.task " +
         "basis=/app/plugins/au_basis_v1.json,/app/plugins/trait_basis_v1.json name=fx"

ds.controlFx("fx", "au12", 0.8, 500);   // smile appears over 500 ms
ds.controlFx("fx", "dom", 2.0);         // dominant face, immediately
ds.controlFx("fx", "au12", 0, 300);     // smile fades out
ds.polyControlFx("fx", "amplitudes", "string", "AU12=0.5,TRUST_o=1.5");  // any combination at once
```
`controlFx` only takes float properties (`au12`, `dom`, ...) and can interpolate over a duration; the `amplitudes` string reaches every field but changes immediately.

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

For `mozza_mesh`, a live-control test changes amplitudes while a video plays (as DuckSoup does) and checks that frames are identical to the input when all amplitudes are 0 and change as expected otherwise. See [gstmozzamesh/README.md](gstmozzamesh/README.md#tests).

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

## Process an image or video with mozza_mesh
```bash
python3 -c 'from mesh_process import transform_image; transform_image("assets/test_image.jpg", "output/mesh_smile.png", {"AU12": 0.8, "DOM_o": 2})'
python3 -c 'from mesh_process import transform_video; transform_video("assets/video_example.mp4", "output/mesh_smile.mp4", {"AU12": 0.8})'
```

# References & Citation
If you use this work in your, please cite:

```text
Arias, P., Soladie, C., Bouafif, O., Roebel, A., Seguier, R., & Aucouturier, J. J. (2018). Realistic transformation of facial and vocal smiles in real-time audiovisual streams. IEEE Transactions on Affective Computing, 11(3), 507-518.

Arias-Sarah, P., Denis, G., Hall, L., Aucouturier, J. J., Schyns, P. G., Jack, R. E., & Johansson, P. DuckSoup: a videoconference experimental platform to transform participants’ voice and face in real-time during social interactions.

Retrieved from https://github.com/ducksouplab/mp_plugins
```

If you use the `mozza_mesh` basis files, also cite:

```text
Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE, 13(5), e0196391. (AU basis)

Oosterhof, N. N., & Todorov, A. (2008). The functional basis of face evaluation. PNAS, 105(32), 11087-11092. (trait basis)
```

- **MediaPipe**: [Face Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- **GStreamer**: [GstVideoFilter API](https://gstreamer.freedesktop.org/documentation/video/gstvideofilter.html)
