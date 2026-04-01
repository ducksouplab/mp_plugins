#!/bin/bash
# generate_all_outputs.sh - Runs all plugins on all assets and saves to output/

mkdir -p output

# List of assets to process
ASSETS=("assets/dynamic_video.mp4" "assets/test_image.jpg" "assets/video_example.mp4")

echo "Starting batch transformation of all assets..."

for asset in "${ASSETS[@]}"; do
    if [ ! -f "$asset" ]; then
        echo "Skipping missing asset: $asset"
        continue
    fi
    
    filename=$(basename "$asset")
    name="${filename%.*}"
    ext="${filename##*.}"
    
    # Mode 1: Just Landmarks (for tracking verification)
    echo "--------------------------------------------------------"
    mode="landmarks"
    target_name="${name}_${mode}.mp4"
    if [[ "$ext" =~ ^(jpg|jpeg)$ ]]; then target_name="${name}_${mode}.png"; fi
    echo "Mode: $mode | Input: $asset -> output/$target_name"
    python3 mozza_process.py --input "$asset" --output "output/$target_name" --mode "landmarks" --model-path face_landmarker.task

    # Mode 2: CPU Smile (Comparison)
    echo "--------------------------------------------------------"
    mode="cpu_smile"
    target_name="${name}_${mode}.mp4"
    if [[ "$ext" =~ ^(jpg|jpeg)$ ]]; then target_name="${name}_${mode}.png"; fi
    echo "Mode: $mode | Input: $asset -> output/$target_name"
    python3 mozza_process.py --input "$asset" --output "output/$target_name" --mode "cpu" --deform smile.dfm --show-landmarks false --model-path face_landmarker.task --warp-mode per-group-roi --alpha 2.0

    # Mode 3: GPU Smile (Comparison)
    echo "--------------------------------------------------------"
    mode="gpu_smile"
    target_name="${name}_${mode}.mp4"
    if [[ "$ext" =~ ^(jpg|jpeg)$ ]]; then target_name="${name}_${mode}.png"; fi
    echo "Mode: $mode | Input: $asset -> output/$target_name"
    python3 mozza_process.py --input "$asset" --output "output/$target_name" --mode "gpu" --deform smile.dfm --show-landmarks false --model-path face_landmarker.task --warp-mode per-group-roi --alpha 2.0

done

echo "--------------------------------------------------------"
echo "All transformations complete. Files available in 'output/'"
ls -F output/
