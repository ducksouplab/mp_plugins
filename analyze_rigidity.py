import sys

try:
    import numpy as np
except ImportError:
    print("Numpy missing, script needs to be run inside docker.")
    sys.exit(1)

cpu = open('assets/dyn_cpu.txt').read().strip().split('Frame')
for f in cpu:
    if not f.strip(): continue
    lines = f.split('\n')[1:]
    pts = []
    for l in lines:
        if ',' in l:
            pts.append([float(x) for x in l.split(',')[:2]])
    if len(pts) > 263:
        p33 = np.array(pts[33])
        p263 = np.array(pts[263])
        p1 = np.array(pts[1])
        inter_eye = np.linalg.norm(p33 - p263)
        print(f'Eye_Dist: {inter_eye:.5f} Nose: {p1[0]:.4f},{p1[1]:.4f}')
