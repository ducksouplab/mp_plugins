# mozza_mesh

A GStreamer element that deforms faces with **validated displacement fields**
(facial action units, and morphology such as dominance and trustworthiness)
and a **mesh warp**. It is a separate plugin: `mozza_mp` and `mozza_mp_gpu`
(`.dfm` rules + MLS warp) are unchanged.

| | `mozza_mp` | `mozza_mesh` |
|---|---|---|
| What defines a deformation | `.dfm` rules: each landmark moves towards a combination of other landmarks | basis files: for each transformation, how every landmark moves at amplitude 1 |
| Warp | MLS (moving least squares) | piecewise-affine on MediaPipe's face mesh: every landmark lands exactly where the field puts it |
| Changing the deformation live | `alpha` | one amplitude per transformation (`au12`, `dom`, ...), any number combined |
| Eyelids, brows | MLS ripples/pinches around the eyes | clean |

The transformations come from [face-transforms](https://github.com/Pablo-Arias/face-transforms)
(AU basis v1 learned from real faces, RAVDESS; trait basis v1 extracted from
Oosterhof & Todorov 2008), where they are documented and validated. The frozen
files are copied in [`bases/`](bases/).

## Quick example

```bash
gst-launch-1.0 filesrc location=face.jpg ! jpegdec ! videoconvert ! video/x-raw,format=RGBA ! \
  mozza_mesh model=models/face_landmarker.task \
             basis=au_basis_v1.json,trait_basis_v1.json \
             au12=0.8 dom=2 ! \
  videoconvert ! pngenc ! filesink location=out.png
```

From Python (runs the plugin in Docker), see `../mesh_process.py` and the
tutorial `../tutorial/mozza_mesh_tutorial.ipynb`:

```python
from mesh_process import transform_image, transform_video
transform_image("media/inputs/test_image.jpg", "smile.png", {"AU12": 0.8})
transform_video("media/inputs/video_example.mp4", "out.mp4", keyframes=[(0, {"AU12": 0}), (1, {"AU12": 1})])
```

## Properties

| Property | Type | Default | Live | Description |
|---|---|---|---|---|
| `model` | string | required | | Path to `face_landmarker.task` |
| `basis` | string | none | yes | Comma-separated basis JSON files. Later files override fields with the same name. Changing it reloads on the next frame. |
| `au1` `au2` `au4` `au5` `au7` `au12` `au15` `au20` `au43` | float | 0 | yes | Amplitude of AU1 ... AU43. 1 = a clear, plausible expression; negative values reverse it (e.g. `au43=-1` widens the eyes). Range -10..10, sensible within +-1.5. |
| `trust`, `dom`, `threat` | float | 0 | yes | Amplitude of `TRUST_o`, `DOM_o` (orthogonal pair) and `THREAT`, in SD of Oosterhof & Todorov's models; sensible within +-3. |
| `amplitudes` | string | "" | yes | All amplitudes at once, for any field name in the loaded bases: `"AU12=0.5,DOM_o=2,TRUST=-1"`. Unlisted fields are set to 0. Reading it returns the current non-zero amplitudes. |
| `fold-guard` | float | 5 | yes | If a combination would fold the skin over itself inside the face by more than this area (px²), the deformation is scaled down; the scale then recovers gradually (~0.7 s at 30 fps), so the expression never jumps. Triangles on the face outline are ignored: when the head turns they become slivers that flip invisibly. Negative = off. |
| `smooth-landmarks` | bool | true | yes | OneEuro filter on the landmarks (reduces jitter of the deformation in video). |
| `min-cutoff`, `beta` | float | 2.0, 0.05 | yes | OneEuro parameters, as `mozza_mp_gpu`. |
| `show-landmarks` | bool | false | yes | Draw the detected landmarks. |
| `landmark-radius`, `landmark-color` | int, uint | 2, 0x00FF00FF | | Landmark dots. |
| `no-warp` | bool | false | yes | Detect only. |
| `drop` | bool | false | | Drop frames without a face. |
| `ignore-timestamps` | bool | false | | Use the frame count as detector timestamps. |
| `threads` | int | 4 | | MediaPipe CPU threads. |
| `max-faces` | int | 1 | | Faces detected; only the first is transformed. |
| `log-every` | uint | 60 | | Timing log every N frames (`GST_DEBUG=mozza_mesh:4`). |
| `user-id` | string | | | Accepted for DuckSoup configs; unused. |

The named float properties and `amplitudes` write to the same set of
amplitudes: `au12=0.8` is the same as `amplitudes="AU12=0.8"`.

## Basis files

```
{"meta": {"version": "v1", "units": ..., "fields": {...}},
 "fields": {"AU12": [[dx, dy], ... 468 rows], "DOM_o": [...], ...}}
```

For each of MediaPipe's 468 face-mesh landmarks, the displacement at
amplitude 1, in face units: `x` along the line from the outer eye corner 33 to
263, `y` perpendicular to it (down), 1 unit = the distance between those two
landmarks. So the same file works for any face size, position and in-plane
rotation. Each frame, the plugin computes
`displacement = sum(amplitude * field)`, converts it to pixels with the
detected face's eye corners, and warps.

Shipped in `bases/` (copies of face-transforms' frozen files):

| File | Fields |
|---|---|
| `au_basis_v1.json` | AU1, AU2, AU4, AU5, AU7, AU12, AU15, AU20, AU43 |
| `trait_basis_v1.json` | TRUST_o, DOM_o (recommended orthogonal pair), TRUST, DOM, THREAT |

New or improved transformations are new basis files (v2, ...): no change to
the plugin.

**Attribution:** the bases are for non-commercial research. If you use them,
cite RAVDESS (Livingstone & Russo 2018) for the AU basis and Oosterhof &
Todorov (2008) for the trait basis; full references in
[bases/README.md](bases/README.md). The files contain only averaged landmark
displacements, no material from those databases.

## In DuckSoup

Name the effect and control it with the player API:

```js
videoFx: "mozza_mesh model=/app/plugins/face_landmarker.task " +
         "basis=/app/plugins/au_basis_v1.json,/app/plugins/trait_basis_v1.json name=fx"

ds.controlFx("fx", "au12", 0.8, 500);   // smile appears over 500 ms
ds.controlFx("fx", "dom", 2.0);         // dominant face, immediately
ds.controlFx("fx", "au12", 0, 300);     // smile fades out
ds.polyControlFx("fx", "amplitudes", "string", "AU12=0.5,TRUST_o=1.5");  // any combination at once
```

`controlFx` only takes float properties (hence `au12`, `dom`, ...);
`polyControlFx` with the `amplitudes` string reaches every field but isn't
interpolated. Deployment: copy `libgstmozzamesh.so` with the other plugins
and the basis files next to `face_landmarker.task` (see the repository
README).

## Building

**Official build**: part of the repository Dockerfile (Bazel target
`//gstmozzamesh:libgstmozzamesh.so`, installed with the other plugins).

**Development build** (seconds, no MediaPipe build): the plugin only needs
GStreamer, OpenCV and json-glib from the base image, and loads MediaPipe at
runtime through `libmp_runtime.so`.

```bash
docker build --platform linux/amd64 -t mozza-mesh-dev gstmozzamesh/dev
docker run --rm --platform linux/amd64 -v "$PWD":/src -w /src mozza-mesh-dev gstmozzamesh/dev/build.sh
# -> gstmozzamesh/out/libgstmozzamesh.so, gstmozzamesh/out/facewarp_cli
# use it with the existing plugins image:
docker run --rm --platform linux/amd64 -v "$PWD":/src -e GST_PLUGIN_PATH=/src/gstmozzamesh/out:/usr/local/lib/gstreamer-1.0:/opt/gstreamer/lib/x86_64-linux-gnu/gstreamer-1.0 \
  --entrypoint "" mp_plugins:latest gst-inspect-1.0 mozza_mesh
```

## Tests

- `tests/test_live.py`: changes amplitudes while a video plays (as DuckSoup
  does) and checks frames: identical to the unprocessed reference when all
  amplitudes are 0, growing difference during an `au12` ramp, effect of an
  `amplitudes` string, and back to identical after clearing it.
- `tools/facewarp_cli`: the warp core on one image with given landmarks;
  used to check the C++ output against the Python reference
  (face-transforms `facekit/meshwarp.py`, `topology="strip"`): at most 3/255
  difference in at most ~90 pixels per image, over 18 face/amplitude cases.

## Design

- `facewarp.hpp/.cpp`: the portable core (basis loading with json-glib, face
  frame, mesh, fold check, warp). No GStreamer dependency; a GPU version
  (`mozza_mesh_gpu`) can reuse it for everything but the per-pixel warp.
- Mesh: MediaPipe's 898 face triangles, plus two rings of 36 points around the
  face outline (scaled 1.35x and 1.8x about its centroid) joined by fixed
  triangle strips. Rings don't move; pixels beyond the outer ring are untouched.
  Degenerate triangles (closed lip seam) are dropped per frame.
- Warp: triangle id per output pixel (rasterised), affine map per triangle,
  `cv::remap` (bilinear) on the face's bounding box only.
- `face_mesh_data.hpp` is generated from MediaPipe's canonical face model.

Not yet done: `mozza_mesh_gpu` (CUDA), so far CPU only.
