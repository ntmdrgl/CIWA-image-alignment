import json
import numpy as np

def calculate_derived_intrinsics(H, W, HFOV, VFOV):
    fx = (W / 2) / np.tan(HFOV / 2).item()
    fy = (H / 2) / np.tan(VFOV / 2).item()
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
        print(f"Saved derived intrinsics for {camera} to camera_intrinsics.json:")
        print(f"    fx: {intrinsics[camera]['derived_intrinsics'][0]}")
        print(f"    fy: {intrinsics[camera]['derived_intrinsics'][1]}")
        print(f"    cx: {intrinsics[camera]['derived_intrinsics'][2]}")
        print(f"    cy: {intrinsics[camera]['derived_intrinsics'][3]}")

    with open('camera_intrinsics.json', 'w') as f:
        json.dump(intrinsics, f, indent=4)

if __name__ == "__main__":
    main()