import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.config import load_config, store_config

from utils.optimization import run_optimization, create_objective_function
from utils.metrics import image_mutual_information

parser = argparse.ArgumentParser()
parser.add_argument("--store", type=str, required=True, help="Name of config file for parameters to be stored in congigs/ ex. my_config or my_config.yaml")
args = parser.parse_args()

save_config_name = args.store # <--- save parameters to config file

cfg = load_config()

checkboard = cv2.imread("checkerboard_small.png")

src_camera = cfg.visible_camera
tgt_camera = cfg.thermal_camera

src_images = []
tgt_images = []

# src_image = np.load('visible.npy')
# src_H, src_W = src_image.shape[:2]

# tgt_image = np.load('thermal.npy').astype(np.float32)
# tgt_H, tgt_W = tgt_image.shape[:2]

# # resize source image to target dimensions
# src_image =  cv2.resize(src_image, (tgt_W, tgt_H), interpolation=cv2.INTER_AREA)

# # grayscale source image
# src_gray = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)

# print("source:", src_gray.shape, np.min(src_gray), np.max(src_gray), src_gray.dtype)
# print("target:", tgt_image.shape, np.min(tgt_image), np.max(tgt_image), tgt_image.dtype,  "\n")

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

objective_function = create_objective_function(
    src_image=src_gray,
    tgt_image=tgt_image,
    camera_matrix=camera_matrix,
    bins=256,
)

best_parameters, best_mi = run_optimization(
    objective_function,
    lower_bounds,
    upper_bounds,
    parameter_names,
    epoch=20,
    pop_size=100,
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