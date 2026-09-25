import numpy as np
import cv2
import argparse
import pathlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import flir_image_extractor

from utils.config import load_config, store_config
from utils.optimization import run_optimization, create_dataset_objective_function
from utils.metrics import image_mutual_information
from utils.camera_intrinsics import get_camera_matrix_from_camera

parser = argparse.ArgumentParser()

parser.add_argument(
    "--store",
    type=str,
    required=True,
    help=(
        "Name of the config file in configs/. "
        "Example: --store my_config or --store my_config.yaml"
    ),
)

args, config_arguments = parser.parse_known_args()

save_config_name = args.store

# Only configuration-related arguments are passed onward.
cfg = load_config(config_arguments)

checkboard = cv2.imread("checkerboard_small.png")

src_camera = cfg.visible_camera
tgt_camera = cfg.thermal_camera

src_images = []
tgt_images = []

plant = "citrus"
num_images = 20

thermal_dir = pathlib.Path("data\\" + plant + "\\train_thermal")

flir = flir_image_extractor.FlirImageExtractor()

image_paths = sorted(
    img
    for img in thermal_dir.iterdir()
    if img.is_file() and img.suffix.lower() == ".jpg"
)

# Process at most num_images.
image_paths = image_paths[:num_images]

for img in tqdm(
    image_paths,
    desc="Extracting FLIR images",
    unit="image",
    dynamic_ncols=True,
):
    flir.process_image(img)

    src_image = flir.get_rgb_np()
    tgt_image = flir.get_thermal_np()

    tgt_H, tgt_W = tgt_image.shape[:2]

    # Resize the visible source to the thermal dimensions.
    src_image = cv2.resize(
        src_image,
        (tgt_W, tgt_H),
        interpolation=cv2.INTER_AREA,
    )

    # Convert visible RGB image to grayscale.
    src_gray = cv2.cvtColor(
        src_image,
        cv2.COLOR_RGB2GRAY,
    )

    src_images.append(src_gray)
    tgt_images.append(tgt_image)

print(f"Finished extracting {len(src_images)} image pairs.")

# --- Particle Swarm Optimization ---

parameter_names = [
    "angle",
    "hx", "hy",
    "sx", "sy",
    "tx", "ty",
    "k1", "k2", "k3",
    "p1", "p2",
]

lower_bounds = (
    0, # -1.0,                   # angle
    0, 0, # -0.05, -0.05,           # hx, hy
     1.0,  1.0,             # sx, sy
    -10.0, -10.0,           # tx, ty
    -0.50, -0.30, -0.15,    # k1, k2, k3
    -0.01, -0.01,           # p1, p2
)

upper_bounds = (
    0, # 1.0,                   # angle
    0, 0, # 0.05,  0.05,           # hx, hy
     1.40, 1.40,           # sx, sy
     10.0, 10.0,            # tx, ty
     0.50, 0.30, 0.15,      # k1, k2, k3
     0.01, 0.01,            # p1, p2
)

camera_matrix = get_camera_matrix_from_camera(tgt_camera)

objective_function = create_dataset_objective_function(
    src_images=src_images,
    tgt_images=tgt_images,
    camera_matrix=camera_matrix,
    bins=256,
)

best_parameters, best_mi = run_optimization(
    objective_function,
    lower_bounds,
    upper_bounds,
    parameter_names,
    epoch=20,
    pop_size=50,
    c1=1.8,
    c2=1.2,
    w=0.9,
    seed=42
)

for parameter_name, parameter_value in best_parameters.items():
    setattr(cfg, parameter_name, float(parameter_value))

print("Best parameters:")
for name, value in best_parameters.items():
    print(f"    {name}: {value}")

print("Best mutual information:", best_mi, "\n")

saved_path = store_config(
    **best_parameters,
    config_name=save_config_name,
)

print(f"Saved configuration to: {saved_path}")