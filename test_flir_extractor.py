import flir_image_extractor

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--plant", type=str, required=True, help="Name of the plant dataset in data directory")
parser.add_argument("--num_images", type=int, required=False, help="The number of images", default=1)
parser.add_argument("--start_img", type=int, required=False, help="The starting image index", default=0)
args = parser.parse_args()

thermal_dir = pathlib.Path("data\\" + args.plant + "\\train_thermal")

flir = flir_image_extractor.FlirImageExtractor()

for index, img in enumerate(thermal_dir.iterdir()):
    if not img.is_file() or img.suffix != ".jpg":
        continue

    if index >= args.num_images:
        print("Finished extracting images")
        break

    flir.process_image(img)
    
    thermal_img = flir.get_thermal_np()
    print(f"thermal_img: shape: {thermal_img.shape}, dtype: {thermal_img.dtype}")
    
    embedded_img = flir.get_rgb_np()
    print(f"embedded_img: shape: {embedded_img.shape}, dtype: {embedded_img.dtype}")

    flir.plot()