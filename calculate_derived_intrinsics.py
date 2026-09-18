import json
import numpy as np

def calculate_derived_intrinsics(H, W, HFOV, VFOV):
    fx = (W / 2) / np.tan(np.deg2rad(HFOV / 2)).item()
    fy = (H / 2) / np.tan(np.deg2rad(VFOV / 2)).item()
    cx = W / 2
    cy = H / 2
    return fx, fy, cx, cy

def main():
    with open('camera_intrinsics.json', 'r') as f:
        intrinsics = json.load(f)

    for camera in intrinsics:
        fx, fy, cx, cy = calculate_derived_intrinsics(
            intrinsics[camera]['H'], 
            intrinsics[camera]['W'], 
            intrinsics[camera]['HFOV'], 
            intrinsics[camera]['VFOV']
        )

        intrinsics[camera]['fx'] = fx
        intrinsics[camera]['fy'] = fy
        intrinsics[camera]['cx'] = cx
        intrinsics[camera]['cy'] = cy

        print(f"Saved derived intrinsics for {camera} to camera_intrinsics.json:")
        print(f"    fx: {intrinsics[camera]['fx']}")
        print(f"    fy: {intrinsics[camera]['fy']}")
        print(f"    cx: {intrinsics[camera]['cx']}")
        print(f"    cy: {intrinsics[camera]['cy']}")

    with open('camera_intrinsics.json', 'w') as f:
        json.dump(intrinsics, f, indent=4)

if __name__ == "__main__":
    main()