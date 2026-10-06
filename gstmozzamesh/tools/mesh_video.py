"""Run mozza_mesh on a video with amplitudes that change over time (inside the image).

Keyframes are linearly interpolated per frame from its timestamp, like
DuckSoup's controlFx with a duration. Used by mesh_process.transform_video;
runs in the plugins image, which has GStreamer's Python bindings.

  python3 mesh_video.py in.mp4 out.mp4 --model m.task --basis a.json,b.json \\
      --keyframes '[[0, {"AU12": 0}], [1.5, {"AU12": 1}], [3, {"AU12": 0, "DOM_o": 2}]]'

Each keyframe is [time in seconds, {field: amplitude}]; a field missing from a
keyframe is 0 there. Extra element properties: --prop name=value (repeatable).
"""
import argparse
import json

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


def amplitudes_at(keyframes, t):
    """Linear interpolation of the keyframe amplitudes at time t (seconds)."""
    names = sorted({k for _, a in keyframes for k in a})
    if t <= keyframes[0][0]:
        a = keyframes[0][1]
        return {n: float(a.get(n, 0.0)) for n in names}
    for (t0, a0), (t1, a1) in zip(keyframes, keyframes[1:]):
        if t <= t1:
            w = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return {n: (1 - w) * a0.get(n, 0.0) + w * a1.get(n, 0.0) for n in names}
    a = keyframes[-1][1]
    return {n: float(a.get(n, 0.0)) for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--model", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--keyframes", required=True, help="JSON list of [seconds, {field: amplitude}]")
    ap.add_argument("--prop", action="append", default=[], help="extra mozza_mesh property name=value")
    a = ap.parse_args()
    keyframes = sorted(json.loads(a.keyframes), key=lambda k: k[0])

    Gst.init(None)
    extra = " ".join(a.prop)
    pipe = Gst.parse_launch(
        f"filesrc location={a.input} ! qtdemux ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=RGBA ! "
        f"mozza_mesh name=fx model={a.model} basis={a.basis} {extra} ! videoconvert ! "
        f"x264enc speed-preset=medium ! mp4mux ! filesink location={a.output}")
    fx = pipe.get_by_name("fx")

    def probe(pad, info):
        buf = info.get_buffer()
        t = buf.pts / Gst.SECOND if buf.pts != Gst.CLOCK_TIME_NONE else 0.0
        amps = amplitudes_at(keyframes, t)
        fx.set_property("amplitudes", ",".join(f"{k}={v:.5f}" for k, v in amps.items()))
        return Gst.PadProbeReturn.OK

    fx.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe)
    pipe.set_state(Gst.State.PLAYING)
    msg = pipe.get_bus().timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    pipe.set_state(Gst.State.NULL)
    if msg.type == Gst.MessageType.ERROR:
        raise SystemExit(f"error: {msg.parse_error()}")


if __name__ == "__main__":
    main()
