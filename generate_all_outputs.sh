#!/bin/bash
# generate_all_outputs.sh - Runs all plugins on all assets and saves to output/

mkdir -p output

# List of assets to process
ASSETS=("assets/dynamic_video.mp4" "assets/test_image.jpg" "assets/video_example.mp4")
MODES=("landmarks" "cpu" "gpu")

echo "Starting batch transformation of all assets..."

for asset in "${ASSETS[@]}"; do
    if [ ! -f "$asset" ]; then
        echo "Skipping missing asset: $asset"
        continue
    fi
    
    filename=$(basename "$asset")
    name="${filename%.*}"
    ext="${filename##*.}"
    
    for mode in "${MODES[@]}"; do
        # Determine output extension (images to PNG)
        out_ext="$ext"
        if [[ "$ext" =~ ^(jpg|jpeg)$ ]]; then
             out_ext="png"
        fi
        
        target_name="${name}_${mode}.${out_ext}"
        output_path="output/${target_name}"
        
        echo "--------------------------------------------------------"
        echo "Mode: $mode | Input: $asset -> $output_path"
        
        # Run the wrapper
        # Note: mozza_process.py might save to 'assets/' due to its mount logic
        python3 mozza_process.py --input "$asset" --output "$target_name" --mode "$mode" --deform smile.dfm --model-path face_landmarker.task
        
        # Cleanup move if it landed in the wrong spot
        if [ -f "assets/${target_name}" ]; then
            mv -f "assets/${target_name}" "$output_path"
        elif [ -f "${target_name}" ]; then
            mv -f "${target_name}" "$output_path"
        fi
    done
done

echo "--------------------------------------------------------"
echo "All transformations complete. Files available in 'output/'"
ls -F output/
