# mozza_mp_gpu — GPU-Accelerated Facial Deformation

Drop-in GPU replacement for `mozza_mp` using **TensorRT + CUDA** instead of MediaPipe + OpenCV.

## Architecture

```
                        CPU (mozza_mp) ~40ms/frame
GstBuffer(RGBA) → MediaPipe CPU (35ms) → OpenCV MLS warp (5ms) → out

                        GPU (mozza_mp_gpu) ~3-5ms/frame
GstBuffer(RGBA) → cudaMemcpy H2D → TensorRT inference (1-2ms) → CUDA MLS warp (0.5ms) → cudaMemcpy D2H → out
```

**What changed:**
| Component | CPU (mozza_mp) | GPU (mozza_mp_gpu) |
|-----------|---------------|-------------------|
| Face detection | MediaPipe FaceLandmarker (XNNPACK, 4 threads) | TensorRT BlazeFace (FP16) |
| Landmark regression | MediaPipe FaceLandmarker | TensorRT Face Landmarks (FP16) |
| MLS Warp | OpenCV `cv::Mat` CPU loops | CUDA kernels (parallel per-pixel) |
| Image preprocessing | CPU copy + ImageFrame | CUDA resize + RGBA→RGB kernel |

## Setup

### 1. Convert models (one-time)

```bash
pip install tf2onnx tensorflow
python3 convert_models.py face_landmarker.task
```

This creates `face_detector.onnx` and `face_landmarks.onnx` alongside the `.task` file.

### 2. Build

```bash
docker build -f Dockerfile.gpu -t mp_plugins_gpu .
```

### 3. Use in GStreamer pipeline

```bash
# Same properties as mozza_mp — drop-in replacement
gst-launch-1.0 videotestsrc ! video/x-raw,format=RGBA ! \
  mozza_mp_gpu model=/path/to/face_landmarker.task \
               deform=/path/to/smile.dfm \
               alpha=1.7 \
               mls-alpha=1.4 \
               mls-grid=5 \
               gpu-id=0 ! \
  autovideosink
```

## Properties

Same as `mozza_mp`, plus:

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `gpu-id` | int | 0 | CUDA device index |

**Note:** `delegate` and `threads` properties are removed (always GPU, no CPU threads needed).

## Performance

On an NVIDIA A100/V100/RTX server:
- Face detection + landmarks: ~1-2ms (vs 35ms CPU)
- MLS warp: ~0.5ms (vs 5ms CPU)
- Memory transfer (H2D + D2H): ~1-2ms at 640x480
- **Total: ~3-5ms/frame → 200-300+ FPS**

First run takes ~30-60s to build TensorRT engines (cached to disk for subsequent runs).

## Files

```
gstmozzamp_gpu/
├── gstmozzamp_gpu.cpp         # GStreamer plugin (transform_frame_ip)
├── trt_face_landmarker.h/cpp  # TensorRT two-stage inference
├── cuda_mls_warp.h/cu         # CUDA MLS rigid warp kernels
├── cuda_preprocess.h/cu       # CUDA image preprocessing kernels
├── task_model_extractor.h/cpp # Extract .tflite from .task ZIP
├── BUILD                      # Bazel build file
└── README.md                  # This file
```
