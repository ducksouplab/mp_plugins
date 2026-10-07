#!/usr/bin/env python3
"""Python wrapper for the mozza_mesh plugin (runs it in Docker), like mozza_process.py.

    from mesh_process import transform_image, transform_video

    transform_image("media/inputs/test_image.jpg", "smile.png", {"AU12": 0.8})
    transform_image("media/inputs/test_image.jpg", "dominant.png", {"DOM_o": 2.5})
    transform_video("media/inputs/video_example.mp4", "out.mp4", {"AU12": 0.6})          # constant
    transform_video("media/inputs/video_example.mp4", "out.mp4",                          # changing over time
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


def transform_image(input_path, output_path, amplitudes=None, basis=DEFAULT_BASIS, model="models/face_landmarker.task",
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
                    model="models/face_landmarker.task", show_landmarks=False, fold_guard=5.0, smooth=True,
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


def amplitudes_at(keyframes, t):
    """Amplitudes at time t (seconds), linearly interpolated between keyframes (as transform_video)."""
    names = sorted({k for _, a in keyframes for k in a})
    if t <= keyframes[0][0]:
        a = keyframes[0][1]
        return {k: a.get(k, 0.0) for k in names}
    for (t0, a0), (t1, a1) in zip(keyframes, keyframes[1:]):
        if t <= t1:
            w = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return {k: (1 - w) * a0.get(k, 0.0) + w * a1.get(k, 0.0) for k in names}
    a = keyframes[-1][1]
    return {k: a.get(k, 0.0) for k in names}


def side_by_side_video(original_path, transformed_path, output_path, keyframes=None, amplitudes=None,
                       height=360, image="mp_plugins:latest"):
    """Original | transformed, side by side, with the current amplitudes written on each frame.

    Needs OpenCV (pip install opencv-python). The H.264 encoding runs in the plugins image,
    so the .mp4 plays in browsers and notebooks whatever OpenCV build is installed.
    """
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # silence decoder warnings on damaged frames
    import cv2

    a, b = cv2.VideoCapture(original_path), cv2.VideoCapture(transformed_path)
    fps = a.get(cv2.CAP_PROP_FPS) or 25.0
    tmp = os.path.splitext(os.path.abspath(output_path))[0] + "_tmp.avi"
    writer, k = None, 0
    while True:
        ok1, f1 = a.read()
        ok2, f2 = b.read()
        if not (ok1 and ok2):
            break
        w = int(round(f1.shape[1] * height / f1.shape[0])) // 2 * 2
        f1, f2 = cv2.resize(f1, (w, height)), cv2.resize(f2, (w, height))
        t = k / fps
        amps = amplitudes_at(keyframes, t) if keyframes else (amplitudes or {})
        label = "  ".join(f"{n}={v:+.1f}" for n, v in amps.items() if abs(v) >= 0.05) or "no transformation"
        for f, text in ((f1, "original"), (f2, label)):
            cv2.putText(f, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(f, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        frame = cv2.hconcat([f1, f2])
        if writer is None:
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        k += 1
    a.release()
    b.release()
    if writer is None:
        raise RuntimeError("could not read the videos")
    writer.release()
    cmd = (f"gst-launch-1.0 -q filesrc location=/out/{os.path.basename(tmp)} ! avidemux ! jpegdec ! videoconvert ! "
           f"x264enc speed-preset=medium ! video/x-h264,profile=main ! mp4mux ! "
           f"filesink location=/out/{os.path.basename(output_path)}")
    try:
        _docker(cmd, tmp, output_path, None, image)
    finally:
        os.remove(tmp)
    return output_path
