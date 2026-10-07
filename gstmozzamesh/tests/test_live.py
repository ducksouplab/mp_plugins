"""Live-control test for mozza_mesh: change amplitudes while a video plays.

Mimics DuckSoup: properties are set from outside while the pipeline runs (a
pad probe sets them just before each frame, so the schedule is exact). Runs
the same video twice through mozza_mesh: once with no amplitudes (reference)
and once with this schedule:

  frames  0-19  nothing                     -> must equal the reference
  frames 20-39  au12 ramps 0 -> 1           -> difference grows (like controlFx with a duration)
  frames 40-49  amplitudes="DOM_o=3,AU4=1"  -> differs
  frames 50-    amplitudes=""               -> must equal the reference again

Also saves a contact sheet of some output frames and reports per-frame time.

  docker run --rm --platform linux/amd64 -v "$PWD":/src -w /src \\
    -e LD_LIBRARY_PATH=/src/mp-out/lib:/usr/local/lib:/opt/gstreamer/lib/x86_64-linux-gnu \\
    mozza-mesh-dev python3 gstmozzamesh/tests/test_live.py media/inputs/video_example.mp4 \\
      --basis gstmozzamesh/bases/au_basis_v1.json,gstmozzamesh/bases/trait_basis_v1.json
"""
import argparse
import os
import sys
import time

import numpy as np
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["GST_PLUGIN_PATH"] = os.path.join(HERE, "..", "out") + ":" + os.environ.get("GST_PLUGIN_PATH", "")


def schedule(fx, k):
    if k < 20:
        return "none"
    if k < 40:
        fx.set_property("au12", (k - 20) / 19)
        return f"au12={(k - 20) / 19:.2f}"
    if k == 40:
        fx.set_property("amplitudes", "DOM_o=3,AU4=1")
    if k < 50:
        return "DOM_o=3,AU4=1"
    if k == 50:
        fx.set_property("amplitudes", "")
    return "reset"


def run(video, model, basis, live):
    pipe = Gst.parse_launch(
        f"filesrc location={video} ! qtdemux ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=RGBA ! "
        f"mozza_mesh name=fx model={model} basis={basis} ! videoconvert ! video/x-raw,format=RGB ! "
        "appsink name=sink emit-signals=true sync=false")
    fx, sink = pipe.get_by_name("fx"), pipe.get_by_name("sink")
    frames, labels, count = [], [], [0]

    def probe(pad, info):
        labels.append(schedule(fx, count[0]) if live else "reference")
        count[0] += 1
        return Gst.PadProbeReturn.OK

    fx.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe)

    def on_sample(s):
        sample = s.emit("pull-sample")
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value("width"), caps.get_value("height")
        buf = sample.get_buffer()
        ok, m = buf.map(Gst.MapFlags.READ)
        frames.append(np.frombuffer(m.data, np.uint8).reshape(h, -1)[:, :w * 3].reshape(h, w, 3).copy())
        buf.unmap(m)
        return Gst.FlowReturn.OK

    sink.connect("new-sample", on_sample)
    t0 = time.time()
    pipe.set_state(Gst.State.PLAYING)
    msg = pipe.get_bus().timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    if msg.type == Gst.MessageType.ERROR:
        raise RuntimeError(msg.parse_error())
    pipe.set_state(Gst.State.NULL)
    return frames, labels, (time.time() - t0) / max(1, len(frames))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="models/face_landmarker.task")
    ap.add_argument("--basis", required=True)
    ap.add_argument("--sheet", default="gstmozzamesh/tests/live_sheet.ppm")
    a = ap.parse_args()
    Gst.init(None)
    ref, _, t_ref = run(a.video, a.model, a.basis, live=False)
    out, labels, t_live = run(a.video, a.model, a.basis, live=True)
    assert len(ref) == len(out), (len(ref), len(out))
    diffs = [float(np.abs(r.astype(int) - o.astype(int)).mean()) for r, o in zip(ref, out)]
    for k, (lab, d) in enumerate(zip(labels, diffs)):
        if k % 5 == 0 or k in (19, 20, 39, 40, 49, 50):
            print(f"frame {k:3d}  {lab:16s}  mean |out - ref| = {d:.3f}")
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)
    check(all(d == 0 for d in diffs[:20]), "frames 0-19 identical to reference")
    ramp = diffs[20:40]
    check(ramp[0] == 0 and ramp[-1] > 0 and np.corrcoef(np.arange(20), ramp)[0, 1] > 0.9, "au12 ramp: difference grows with amplitude")
    check(all(d > 0 for d in diffs[40:50]), "frames 40-49 (DOM_o=3,AU4=1) differ")
    check(all(d == 0 for d in diffs[50:]), "after amplitudes=\"\" identical to reference again")
    print(f"{len(out)} frames of {out[0].shape[1]}x{out[0].shape[0]}; {t_ref * 1000:.0f} ms/frame (reference), "
          f"{t_live * 1000:.0f} ms/frame (live) - under x86 emulation, not representative of the server")

    # Contact sheet of some output frames (binary PPM: no imaging library needed)
    picks = [10, 30, 39, 45, 55]
    sheet = np.concatenate([out[k][::2, ::2] for k in picks], 1)
    with open(a.sheet, "wb") as f:
        f.write(b"P6 %d %d 255\n" % (sheet.shape[1], sheet.shape[0]) + sheet.tobytes())
    print("sheet:", a.sheet, "frames", picks)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
