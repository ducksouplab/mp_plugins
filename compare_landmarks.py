import numpy as np

def parse_landmarks(filename):
    frames = []
    current_frame = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('Frame'):
                if current_frame:
                    frames.append(np.array(current_frame))
                current_frame = []
            else:
                try:
                    parts = line.strip().split(',')
                    if len(parts) == 3:
                        current_frame.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    if current_frame:
        frames.append(np.array(current_frame))
    return frames

cpu_frames = parse_landmarks('landmarks_cpu.txt')
gpu_frames = parse_landmarks('landmarks_gpu.txt')

print(f"Loaded {len(cpu_frames)} CPU frames and {len(gpu_frames)} GPU frames")

min_frames = min(len(cpu_frames), len(gpu_frames))

for f in range(min_frames):
    cpu = cpu_frames[f]
    gpu = gpu_frames[f]
    
    diff = gpu - cpu
    rmse = np.sqrt(np.mean(diff**2))
    max_diff = np.max(np.abs(diff), axis=0)
    mean_diff = np.mean(diff, axis=0)
    
    dist = np.sqrt(np.sum(diff**2, axis=1))
    worst_indices = np.argsort(dist)[::-1][:10]
    
    print(f"\nFrame {f}:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Max Error: {np.max(dist):.6f}")
    print(f"  Worst Landmarks (Indices): {worst_indices}")
    print(f"  Worst Landmark Delta: {dist[worst_indices[0]]:.6f}")
    
    if f == 0:
        print("\nTop 5 Landmarks Comparison (X, Y):")
        for i in range(5):
            print(f"  L[{i}]: CPU({cpu[i,0]:.4f}, {cpu[i,1]:.4f}) vs GPU({gpu[i,0]:.4f}, {gpu[i,1]:.4f})")
