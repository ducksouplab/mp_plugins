# facelandmarks (GStreamer + MediaPipe in Docker)

Two lean GStreamer video filters (`facelandmarks` and `mozza_mp`) which run **MediaPipe Face Landmarker (C++ Tasks)** on CPU and overlay the landmarks on RGBA frames. The project is still under active development: the GPU delegate path (`delegate=gpu`) is not functional yet and requires further work. Depends on OpenCV (see build files for details). Set `delegate=gpu` to enable the MediaPipe GPU delegate when available once implemented. We currently recommend running with `ignore-timestamps=false`.

- Base image: `ducksouplab/debian-gstreamer:deb12-cuda12.2-plugins-gst1.24.10`
- GStreamer plugin base class: **GstVideoFilter** (`transform_frame_ip`).  
  See GStreamer docs for GstVideoFilter and plugin discovery. [refs]  
- MediaPipe Face Landmarker uses a `.task` model and `VIDEO` mode (ms timestamps). [refs]

## Plugin parameters

Each plugin has its own set of parameters. The GPU delegate (`delegate=gpu`) is
not yet functional and requires more work. We currently recommend keeping
`ignore-timestamps=false` for typical scenarios.

### mozza_mp

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Path to `face_landmarker.task`. |
| `deform` | string | none | Path to deformation `.dfm` file. |
| `dfm` | string | none | Alias for `deform`. |
| `alpha` | float [-10..10] | 1.0 | Smile intensity multiplier (negative values frown). |
| `overlay` | boolean | false | Draw source/destination control points and vectors. |
| `drop` | boolean | false | Drop frame when no face is detected. |
| `show-landmarks` | boolean | false | Draw all detected landmarks even without a DFM. |
| `strict-dfm` | boolean | false | Fail to start if a deform file is given but cannot be loaded. |
| `force-rgb` | boolean | false | No-op; pads already require RGBA. |
| `ignore-timestamps` | boolean | false | Pass `0` as timestamp to the detector; recommended to keep `false`. |
| `log-every` | uint | 60 | Emit a log message every N frames (0 disables). |
| `user-id` | string | none | Accepted but ignored; useful for uniform configs. |
| `delegate` | string | cpu | Runtime execution delegate (`cpu`, `gpu`, `xnnpack`). `delegate=gpu` is not working and needs further development. |

Recommended invocation:

`mozza_mp model=/app/plugins/face_landmarker.task deform=/app/plugins/smile_corners_only.dfm alpha=1.7 delegate=cpu ignore-timestamps=false`

### facelandmarks

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Path to `face_landmarker.task`. |
| `max-faces` | int | 1 | Maximum number of faces to detect. |
| `draw` | boolean | true | Overlay landmarks on the frame. |
| `radius` | int | 2 | Radius of landmark dots in pixels. |
| `color` | uint | 0x00FF00FF | Packed RGBA color for landmarks. |
| `delegate` | string | cpu | Runtime execution delegate (`cpu`, `gpu`, `xnnpack`). `delegate=gpu` is not working and needs further development. |

## Build

```bash
docker build -t mozzamp:latest .
```

This will:
	1.	Install build deps + GStreamer dev headers (to compile a plugin).
	2.	Install Bazelisk (recommended Bazel launcher).
	3.	Clone MediaPipe and build the Face Landmarker C++ target.
	4.	Export headers/libs into third_party/mediapipe-export.
	5.	Build libgstfacelandmarks.so with CMake and install it to the system plugin path.
	6.	Download face_landmarker.task into /opt/models.

# Verify plugin
```
docker run --rm -it mozzamp:latest \
  bash -lc 'gst-inspect-1.0 mozzamp'
```
You should see properties: model, max-faces, draw, radius, color, delegate.

## Get the .task model
```
chmod +x download_face_landmarker_model.sh
./download_face_landmarker_model.sh
```

## Get the .so files
```
chmod +x get_so_file.sh
./get_so_file.sh mozzamp:latest

## Careful with this line : remove old plugins from destination
##sudo rm -r /home/deploy/deploy-ducksoup/app/plugins/mp_plugins/

!!!! -- Carefull, at the moment Ducksoup won't load any of the plugins if both of them are passed in the folder, because some weird dependency issue, in which both of them load the library and it crashes. I haven't corrected this yet... Maybe will have to do it in the future...

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

## Use it in DuckSoup mirror mode

Example:

mozza_mp model=/app/plugins/face_landmarker.task deform=/app/plugins/smile_corners_only.dfm alpha=1.7 delegate=cpu ignore-timestamps=false

# Quick runs

## Check gst-inspect-1.0
gst-inspect-1.0 mozzamp

Or with the .so directly:
gst-inspect-1.0 libgstmozzamp.so

## 1) Synthetic input (videotestsrc → mp4 file on host)
```
mkdir -p out
docker run --rm -it -v "$PWD/out:/out" mozzamp:latest bash -lc '
  gst-launch-1.0 -v \
    videotestsrc num-buffers=300 ! video/x-raw,width=640,height=480,framerate=30/1 ! \
    videoconvert ! video/x-raw,format=RGBA ! \
    mozzamp model=/opt/models/face_landmarker.task max-faces=1 draw=true radius=2 color=0x00FF00FF ! \
    videoconvert ! x264enc tune=zerolatency ! mp4mux ! filesink location=/out/landmarked.mp4
'
```

## 2) Process a host video file
docker run --rm -it -v "$PWD:/work" mozzamp:latest bash -lc '
  gst-launch-1.0 -v \
    filesrc location=/work/input.mp4 ! decodebin ! videoconvert ! video/x-raw,format=RGBA ! \
    mozzamp model=/opt/models/face_landmarker.task max-faces=1 ! \
    videoconvert ! x264enc ! mp4mux ! filesink location=/work/output_landmarked.mp4
'

# Element usage
The plugin expects RGBA input; negotiate with videoconvert if needed. It can also
negotiate GPU buffers via `video/x-raw(memory:GLMemory)` and will fall back to CPU
copies when such memory types are unsupported:

````
gst-launch-1.0 -v \
  videotestsrc ! video/x-raw,width=640,height=480 ! \
  videoconvert ! video/x-raw,format=RGBA ! \
  mozzamp model=/opt/models/face_landmarker.task max-faces=1 draw=true radius=2 color=0x00FF00FF ! \
  fakesink
```

# Internals & notes
	•	Base class: GstVideoFilter with transform_frame_ip is the idiomatic in-place frame hook for video filters. [refs]
	•	Plugin discovery: install to /usr/lib/x86_64-linux-gnu/gstreamer-1.0/ or set GST_PLUGIN_PATH/--gst-plugin-path. [refs]
	•	MediaPipe: we pass frames as ImageFrame(SRGBA) and call DetectForVideo(image, timestamp_ms). The model is a .task bundle downloaded from the official guide. [refs]
	•	Performance: start with 640×480; increase as needed.
	•	GPU: this example runs on CPU (simpler, portable). MediaPipe GPU graphs require a different setup.


# Update MediaPipe version
```
docker build --build-arg MP_REF=v0.10.xx -t mozzamp:latest .
```

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

