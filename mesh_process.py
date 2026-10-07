#!/usr/bin/env python3
"""Python wrapper for the mozza_mesh plugin (runs it in Docker), like mozza_process.py.

    from mesh_process import transform_image, transform_video

    transform_image("assets/test_image.jpg", "smile.png", {"AU12": 0.8})
    transform_image("assets/test_image.jpg", "dominant.png", {"DOM_o": 2.5})
    transform_video("assets/video_example.mp4", "out.mp4", {"AU12": 0.6})          # constant
    transform_video("assets/video_example.mp4", "out.mp4",                          # changing over time
                    keyframes=[(0, {"AU12": 0}), (1.0, {"AU12": 1}), (2.0, {"AU12": 0, "DOM_o": 2})])

Amplitudes are a dict {field: value}: AU fields (AU1, AU2, AU4, AU5, AU7,
AU12, AU15, AU20, AU43; 1 = a clear expression) and trait fields (TRUST_o,
DOM_o, TRUST, DOM, THREAT; in SD, up to +-3) from the basis files in
gstmozzamesh/bases/. See gstmozzamesh/README.md.

Until the plugins image is rebuilt with mozza_mesh, the plugin built by
gstmozzamesh/dev/build.sh (gstmozzamesh/out/) is loaded into the image;
pass plugin_dir=None once the image contains it.
"""
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASIS = ("gstmozzamesh/bases/au_basis_v1.json", "gstmozzamesh/bases/trait_basis_v1.json")
GST_PATHS = "/usr/local/lib/gstreamer-1.0:/opt/gstreamer/lib/x86_64-linux-gnu/gstreamer-1.0"


def _docker(cmd, input_path, output_path, plugin_dir, image, env=(), verbose=False, show=None):
    """Run `cmd` in the image with /in, /out and the repo (/repo) mounted.

    show: if given, print only the output lines containing this string (e.g. "TIMING").
    """
    in_dir, out_dir = os.path.dirname(os.path.abspath(input_path)), os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    plugin_path = GST_PATHS if plugin_dir is None else f"/repo/{plugin_dir}:{GST_PATHS}"
    args = ["docker", "run", "--rm", "--platform", "linux/amd64", "--entrypoint", "",
            "-u", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{in_dir}:/in:ro", "-v", f"{out_dir}:/out", "-v", f"{ROOT}:/repo:ro",
            "-e", f"GST_PLUGIN_PATH={plugin_path}", "-e", "HOME=/tmp"]
    for e in env:
        args += ["-e", e]
    args += [image, "bash", "-c", cmd]
    if verbose:
        print(" ".join(args))
    r = subprocess.run(args, capture_output=not verbose, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"mozza_mesh failed:\n{(r.stderr or '')[-3000:]}")
    if show and not verbose:
        for line in ((r.stdout or "") + (r.stderr or "")).splitlines():
            if show in line:
                print(re.sub(r"\x1b\[[0-9;]*m", "", line[line.index(show):]))


def _props(amplitudes, basis, model, show_landmarks, fold_guard, smooth):
    amps = ",".join(f"{k}={v}" for k, v in (amplitudes or {}).items())
    basis = ",".join(f"/repo/{b}" for b in basis)
    return (f"model=/repo/{model} basis={basis} amplitudes=\"{amps}\" fold-guard={fold_guard} "
            f"show-landmarks={'true' if show_landmarks else 'false'} smooth-landmarks={'true' if smooth else 'false'}")


def transform_image(input_path, output_path, amplitudes=None, basis=DEFAULT_BASIS, model="face_landmarker.task",
                    show_landmarks=False, fold_guard=5.0, image="mp_plugins:latest",
                    plugin_dir="gstmozzamesh/out", verbose=False):
    """Transform one image (.jpg or .png) and write a .png."""
    dec = "pngdec" if input_path.lower().endswith(".png") else "jpegdec"
    props = _props(amplitudes, basis, model, show_landmarks, fold_guard, smooth=False)
    cmd = (f"gst-launch-1.0 -q filesrc location=/in/{os.path.basename(input_path)} ! {dec} ! videoconvert ! "
           f"video/x-raw,format=RGBA ! mozza_mesh {props} ! videoconvert ! pngenc ! "
           f"filesink location=/out/{os.path.basename(output_path)}")
    _docker(cmd, input_path, output_path, plugin_dir, image, verbose=verbose)
    return output_path


def transform_video(input_path, output_path, amplitudes=None, keyframes=None, basis=DEFAULT_BASIS,
                    model="face_landmarker.task", show_landmarks=False, fold_guard=5.0, smooth=True,
                    image="mp_plugins:latest", plugin_dir="gstmozzamesh/out", log_every=0, verbose=False):
    """Transform an .mp4 (H.264) video; writes .mp4 (video only).

    amplitudes: constant {field: value}; keyframes: [(seconds, {field: value}), ...]
    interpolated linearly per frame (as DuckSoup's controlFx). log_every > 0
    prints the plugin's timing every N frames.
    """
    env = [f"GST_DEBUG=mozza_mesh:4"] if log_every else []
    if keyframes is None:
        props = _props(amplitudes, basis, model, show_landmarks, fold_guard, smooth) + f" log-every={log_every}"
        cmd = (f"gst-launch-1.0 -q filesrc location=/in/{os.path.basename(input_path)} ! qtdemux ! h264parse ! "
               f"avdec_h264 ! videoconvert ! video/x-raw,format=RGBA ! mozza_mesh {props} ! videoconvert ! "
               f"x264enc speed-preset=medium ! mp4mux ! filesink location=/out/{os.path.basename(output_path)}")
    else:
        kf = json.dumps([[float(t), {k: float(v) for k, v in a.items()}] for t, a in keyframes])
        extra = [f"fold-guard={fold_guard}", f"show-landmarks={'true' if show_landmarks else 'false'}",
                 f"smooth-landmarks={'true' if smooth else 'false'}", f"log-every={log_every}"]
        cmd = ("python3 /repo/gstmozzamesh/tools/mesh_video.py "
               f"/in/{os.path.basename(input_path)} /out/{os.path.basename(output_path)} "
               f"--model /repo/{model} --basis {','.join('/repo/' + b for b in basis)} "
               f"--keyframes '{kf}' " + " ".join(f"--prop {p}" for p in extra))
    _docker(cmd, input_path, output_path, plugin_dir, image, env=env, verbose=verbose,
            show="TIMING" if log_every else None)
    return output_path
