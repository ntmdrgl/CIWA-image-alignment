import json
import numpy as np

def calculate_derived_intrinsics(H, W, HFOV, VFOV):
    fx = (W / 2) / np.tan(HFOV / 2)
    fy = (H / 2) / np.tan(VFOV / 2)
    cx = W / 2
    cy = H / 2
    return fx, fy, cx, cy

def main():
    with open('camera_intrinsics.json', 'r') as f:
        intrinsics = json.load(f)

    for camera in intrinsics:
        intrinsics[camera]['derived_intrinsics'] = calculate_derived_intrinsics(
            intrinsics[camera]['H'], 
            intrinsics[camera]['W'], 
            intrinsics[camera]['HFOV'], 
            intrinsics[camera]['VFOV']
        )

    with open('camera_intrinsics.json', 'w') as f:
        json.dump(intrinsics, f, indent=4)