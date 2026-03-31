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
        sys.exit(1)

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
    parser.add_argument("--warp-mode", choices=["global", "per-group-roi"], default="global", help="Warp strategy (CPU only)")
    parser.add_argument("--roi-pad", type=int, default=24, help="ROI padding (CPU only)")
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

    # Paths setup
    input_abs = os.path.abspath(args.input)
    output_abs = os.path.abspath(args.output)
    input_dir = os.path.dirname(input_abs)
    input_file = os.path.basename(input_abs)
    output_file = os.path.basename(output_abs)
    
    # We mount the directory containing the input file to /work in docker
    # and use /work for both input and output.
    docker_workdir = "/work"
    
    # Prepare plugin parameters
    plugin_props = []
    
    # Mount model and deform files if they exist outside input_dir
    model_abs = os.path.abspath(args.model_path)
    model_dir = os.path.dirname(model_abs)
    model_file = os.path.basename(model_abs)
    docker_model_path = f"/models/{model_file}"
    
    deform_arg = ""
    if args.deform:
        deform_abs = os.path.abspath(args.deform)
        deform_dir = os.path.dirname(deform_abs)
        deform_file = os.path.basename(deform_abs)
        docker_deform_path = f"/deform/{deform_file}"
    
    if args.mode == "landmarks":
        element = "facelandmarks"
        plugin_props.append(f"model={docker_model_path}")
        plugin_props.append(f"max-faces={args.max_faces}")
        show_lm = "true" if args.show_landmarks.lower() == "true" else "false"
        plugin_props.append(f"draw={show_lm}")
        plugin_props.append(f"radius={args.landmark_radius}")
        plugin_props.append(f"color={args.landmark_color}")
        plugin_props.append(f"threads={args.threads}")
    elif args.mode == "gpu":
        element = "mozza_mp_gpu"
        plugin_props.append(f"model_path={docker_model_path}")
        if args.deform: plugin_props.append(f"deform={docker_deform_path}")
        plugin_props.append(f"alpha={args.alpha}")
        plugin_props.append(f"mls-alpha={args.mls_alpha}")
        plugin_props.append(f"mls-grid={args.mls_grid}")
        plugin_props.append(f"drop={'true' if args.drop else 'false'}")
        plugin_props.append(f"max-faces={args.max_faces}")
        plugin_props.append(f"gpu-id={args.gpu_id}")
        plugin_props.append(f"log-every={args.log_every}")
        show_lm = "true" if args.show_landmarks.lower() == "true" else "false"
        plugin_props.append(f"show-landmarks={show_lm}")
        no_warp = "true" if args.no_warp.lower() == "true" else "false"
        plugin_props.append(f"no-warp={no_warp}")
        plugin_props.append(f"smooth={args.smooth}")
        gpu_warp_mode = 1 if args.warp_mode == "per-group-roi" else 0
        plugin_props.append(f"warp-mode={gpu_warp_mode}")
        plugin_props.append(f"roi-pad={args.roi_pad}")
    else: # cpu
        element = "mozza_mp"
        plugin_props.append(f"model={docker_model_path}")
        if args.deform: plugin_props.append(f"deform={docker_deform_path}")
        plugin_props.append(f"alpha={args.alpha}")
        plugin_props.append(f"mls-alpha={args.mls_alpha}")
        plugin_props.append(f"mls-grid={args.mls_grid}")
        plugin_props.append(f"warp-mode={args.warp_mode}")
        plugin_props.append(f"roi-pad={args.roi_pad}")
        plugin_props.append(f"overlay={'true' if args.overlay else 'false'}")
        plugin_props.append(f"drop={'true' if args.drop else 'false'}")
        show_lm = "true" if args.show_landmarks.lower() == "true" else "false"
        plugin_props.append(f"show-landmarks={show_lm}")
        no_warp = "true" if args.no_warp.lower() == "true" else "false"
        plugin_props.append(f"no-warp={no_warp}")
        plugin_props.append(f"landmark-radius={args.landmark_radius}")
        plugin_props.append(f"landmark-color={args.landmark_color}")
        plugin_props.append(f"max-faces={args.max_faces}")
        plugin_props.append(f"threads={args.threads}")
        plugin_props.append(f"log-every={args.log_every}")

    props_str = " ".join(plugin_props)

    # Determine input type and GStreamer pipeline
    ext = os.path.splitext(input_file)[1].lower()
    is_video = ext in [".mp4", ".avi", ".mov", ".mkv"]
    
    out_ext = os.path.splitext(output_file)[1].lower()
    if is_video and not out_ext:
        output_file += ".mp4"
        output_abs += ".mp4"

    if is_video:
        source = f"filesrc location={docker_workdir}/{input_file} ! decodebin ! videoconvert"
        sink = f"videoconvert ! x264enc tune=zerolatency ! mp4mux ! filesink location={docker_workdir}/{output_file}"
    else: # Assume image
        if not out_ext:
            output_file += ".png"
            output_abs += ".png"
        source = f"filesrc location={docker_workdir}/{input_file} ! jpegdec ! videoconvert"
        sink = f"videoconvert ! pngenc ! filesink location={docker_workdir}/{output_file}"

    pipeline = f"{source} ! video/x-raw,format=RGBA ! {element} {props_str} ! {sink}"

    # Docker command
    docker_cmd = [
        "docker", "run", "--rm",
        "-e", "LANDMARK_OUTPUT_FILE=/work/dyn_cpu.txt",
        "-v", f"{input_dir}:{docker_workdir}",
        "-v", f"{model_dir}:/models",
        "-v", f"{os.getcwd()}/mp-out/plugins:/plugins",
        "-e", f"GST_PLUGIN_PATH=/plugins:/usr/local/lib/gstreamer-1.0:/opt/gstreamer/lib/x86_64-linux-gnu/gstreamer-1.0",
    ]
    if args.deform:
        docker_cmd.extend(["-v", f"{deform_dir}:/deform"])
    
    if args.mode == "gpu":
        docker_cmd.append("--gpus")
        docker_cmd.append("all")
        
    docker_cmd.extend([
        args.docker_image,
        "gst-launch-1.0", "-q",
    ])
    
    # Split pipeline into components for subprocess
    docker_cmd.extend(pipeline.split())

    run_command(docker_cmd, args.verbose)
    print(f"Success! Output saved to {args.output}")

if __name__ == "__main__":
    main()
