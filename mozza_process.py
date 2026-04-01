#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

def run_command(cmd, verbose=False):
    if verbose:
        print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        raise e

def transform_file(
    input_path,
    output_path,
    mode="gpu",
    model_path="face_landmarker.task",
    deform=None,
    alpha=1.0,
    mls_alpha=1.4,
    mls_grid=5,
    warp_mode="global",
    roi_pad=24,
    overlay=False,
    drop=False,
    show_landmarks=True,
    landmark_radius=2,
    landmark_color="0x0066CCFF",
    max_faces=1,
    threads=4,
    gpu_id=0,
    log_every=60,
    no_warp=False,
    smooth=0.5,
    docker_image="mp_plugins:latest",
    verbose=False
):
    """
    Programmatic interface to run the MediaPipe GStreamer plugins via Docker.
    """
    # Paths setup
    input_abs = os.path.abspath(input_path)
    output_abs = os.path.abspath(output_path)
    input_dir = os.path.dirname(input_abs)
    input_file = os.path.basename(input_abs)
    output_file = os.path.basename(output_abs)
    
    # We mount the directory containing the input file to /work in docker
    docker_workdir = "/work"
    
    # Prepare plugin parameters
    plugin_props = []
    
    model_abs = os.path.abspath(model_path)
    model_dir = os.path.dirname(model_abs)
    model_file = os.path.basename(model_abs)
    docker_model_path = f"/models/{model_file}"
    
    if deform:
        deform_abs = os.path.abspath(deform)
        deform_dir = os.path.dirname(deform_abs)
        deform_file = os.path.basename(deform_abs)
        docker_deform_path = f"/deform/{deform_file}"
    
    # Convert boolean-like strings/bools to GStreamer 'true'/'false'
    def to_gst_bool(val):
        if isinstance(val, str):
            return "true" if val.lower() == "true" else "false"
        return "true" if val else "false"

    if mode == "landmarks":
        element = "facelandmarks"
        plugin_props.append(f"model={docker_model_path}")
        plugin_props.append(f"max-faces={max_faces}")
        plugin_props.append(f"draw={to_gst_bool(show_landmarks)}")
        plugin_props.append(f"radius={landmark_radius}")
        plugin_props.append(f"color={landmark_color}")
        plugin_props.append(f"threads={threads}")
    elif mode == "gpu":
        element = "mozza_mp_gpu"
        plugin_props.append(f"model_path={docker_model_path}")
        if deform: plugin_props.append(f"deform={docker_deform_path}")
        plugin_props.append(f"alpha={alpha}")
        plugin_props.append(f"mls-alpha={mls_alpha}")
        plugin_props.append(f"mls-grid={mls_grid}")
        plugin_props.append(f"drop={to_gst_bool(drop)}")
        plugin_props.append(f"max-faces={max_faces}")
        plugin_props.append(f"gpu-id={gpu_id}")
        plugin_props.append(f"log-every={log_every}")
        plugin_props.append(f"show-landmarks={to_gst_bool(show_landmarks)}")
        plugin_props.append(f"no-warp={to_gst_bool(no_warp)}")
        plugin_props.append(f"smooth={smooth}")
        gpu_warp_mode = 1 if warp_mode == "per-group-roi" else 0
        plugin_props.append(f"warp-mode={gpu_warp_mode}")
        plugin_props.append(f"roi-pad={roi_pad}")
    else: # cpu
        element = "mozza_mp"
        plugin_props.append(f"model={docker_model_path}")
        if deform: plugin_props.append(f"deform={docker_deform_path}")
        plugin_props.append(f"alpha={alpha}")
        plugin_props.append(f"mls-alpha={mls_alpha}")
        plugin_props.append(f"mls-grid={mls_grid}")
        plugin_props.append(f"warp-mode={warp_mode}")
        plugin_props.append(f"roi-pad={roi_pad}")
        plugin_props.append(f"overlay={to_gst_bool(overlay)}")
        plugin_props.append(f"drop={to_gst_bool(drop)}")
        plugin_props.append(f"show-landmarks={to_gst_bool(show_landmarks)}")
        plugin_props.append(f"no-warp={to_gst_bool(no_warp)}")
        plugin_props.append(f"landmark-radius={landmark_radius}")
        plugin_props.append(f"landmark-color={landmark_color}")
        plugin_props.append(f"max-faces={max_faces}")
        plugin_props.append(f"threads={threads}")
        plugin_props.append(f"log-every={log_every}")

    props_str = " ".join(plugin_props)

    # Determine input type and GStreamer pipeline
    input_filename = os.path.basename(input_abs)
    output_filename = os.path.basename(output_abs)
    output_dir = os.path.dirname(output_abs)

    ext = os.path.splitext(input_filename)[1].lower()
    is_video = ext in [".mp4", ".avi", ".mov", ".mkv"]
    
    out_ext = os.path.splitext(output_filename)[1].lower()
    
    # We mount input_dir to /work
    # If output_dir is the same as input_dir, we can use /work directly.
    # Otherwise, we need another mount or we move it after.
    # To keep it simple and robust, we always mount input_dir to /work 
    # and if output_dir is different, we write to /work/tmp_out and then move it.
    
    docker_output_path = f"{docker_workdir}/{output_filename}"

    if is_video:
        if not out_ext: output_filename += ".mp4"; docker_output_path += ".mp4"
        source = f"filesrc location={docker_workdir}/{input_filename} ! decodebin ! videoconvert"
        sink = f"videoconvert ! x264enc tune=zerolatency ! mp4mux ! filesink location={docker_output_path}"
    else: # Assume image
        if not out_ext: output_filename += ".png"; docker_output_path += ".png"
        source = f"filesrc location={docker_workdir}/{input_filename} ! jpegdec ! videoconvert"
        sink = f"videoconvert ! pngenc ! filesink location={docker_output_path}"

    pipeline = f"{source} ! video/x-raw,format=RGBA ! {element} {props_str} ! {sink}"

    # Docker command
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{input_dir}:{docker_workdir}",
        "-v", f"{model_dir}:/models",
    ]
    if os.getenv("GST_DEBUG"):
        docker_cmd.extend(["-e", f"GST_DEBUG={os.getenv('GST_DEBUG')}"])
    
    # If output_dir is different from input_dir, we need to handle it.
    # A simple way is to mount output_dir to /output_dir in docker.
    if output_dir != input_dir:
        docker_cmd.extend(["-v", f"{output_dir}:/output_dir"])
        # Update pipeline to write to /output_dir
        pipeline = pipeline.replace(f"location={docker_output_path}", f"location=/output_dir/{output_filename}")

    if deform:
        docker_cmd.extend(["-v", f"{deform_dir}:/deform"])
    
    if mode == "gpu":
        docker_cmd.append("--gpus")
        docker_cmd.append("all")
        
    docker_cmd.extend([
        "--entrypoint", "/opt/gstreamer/bin/gst-launch-1.0",
        docker_image,
        "-q",
    ])
    
    docker_cmd.extend(pipeline.split())

    run_command(docker_cmd, verbose)
    return True

def main():
    parser = argparse.ArgumentParser(description="Wrapper for Mozza MP GStreamer plugins via Docker")
    
    # Global options
    parser.add_argument("--input", required=True, help="Input file path (image or video)")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--mode", choices=["gpu", "cpu", "landmarks"], default="gpu", help="Processing mode")
    parser.add_argument("--docker-image", default="mp_plugins:latest", help="Docker image to use")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    # Plugin properties
    parser.add_argument("--model-path", default="face_landmarker.task", help="Path to .task model")
    parser.add_argument("--deform", help="Path to .dfm deformation file")
    parser.add_argument("--alpha", type=float, default=1.0, help="Deformation intensity")
    parser.add_argument("--mls-alpha", type=float, default=1.4, help="MLS rigidity")
    parser.add_argument("--mls-grid", type=int, default=5, help="MLS grid size")
    parser.add_argument("--warp-mode", choices=["global", "per-group-roi"], default="global", help="Warp strategy")
    parser.add_argument("--roi-pad", type=int, default=24, help="ROI padding")
    parser.add_argument("--overlay", action="store_true", help="Draw debug overlay")
    parser.add_argument("--drop", action="store_true", help="Drop frames without faces")
    parser.add_argument("--show-landmarks", type=str, default="true", help="Draw landmarks (true/false)")
    parser.add_argument("--landmark-radius", type=int, default=2, help="Landmark dot radius")
    parser.add_argument("--landmark-color", default="0x0066CCFF", help="Landmark color (hex RGBA)")
    parser.add_argument("--max-faces", type=int, default=1, help="Max faces to detect")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device index")
    parser.add_argument("--log-every", type=int, default=60, help="Logging interval")
    parser.add_argument("--no-warp", type=str, default="false", help="Disable warping (true/false)")
    parser.add_argument("--smooth", type=float, default=0.5, help="Temporal smoothing (0 to 0.99)")

    args = parser.parse_args()

    try:
        transform_file(
            input_path=args.input,
            output_path=args.output,
            mode=args.mode,
            model_path=args.model_path,
            deform=args.deform,
            alpha=args.alpha,
            mls_alpha=args.mls_alpha,
            mls_grid=args.mls_grid,
            warp_mode=args.warp_mode,
            roi_pad=args.roi_pad,
            overlay=args.overlay,
            drop=args.drop,
            show_landmarks=args.show_landmarks,
            landmark_radius=args.landmark_radius,
            landmark_color=args.landmark_color,
            max_faces=args.max_faces,
            threads=args.threads,
            gpu_id=args.gpu_id,
            log_every=args.log_every,
            no_warp=args.no_warp,
            smooth=args.smooth,
            docker_image=args.docker_image,
            verbose=args.verbose
        )
        print(f"Success! Output saved to {args.output}")
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()
