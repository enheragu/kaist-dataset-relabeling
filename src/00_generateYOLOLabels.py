#!/usr/bin/env python3
# encoding: utf-8

import os
import yaml
from pathlib import Path
import multiprocessing as mp

from ultralytics import YOLO

from constants import kaist_images_path, dataset_config_path, yolo_labels_path, image_size

MODEL_PATHS='/home/arvc/eeha/yolo_test_utils/runs/detect/no_equalization_sameseed_corrected/'

day_visible_folder_yaml=f"{dataset_config_path}/val_kaist_day_visible.yaml"
night_lwir_folder_yaml=f"{dataset_config_path}/val_kaist_night_lwir.yaml"
day_visible_model=f"{MODEL_PATHS}/day_visible_4090/weights/best.pt"
night_lwir_model=f"{MODEL_PATHS}/night_lwir_4090/weights/best.pt"

# Configure to run :)
model_pt = day_visible_model
yaml_file = day_visible_folder_yaml


def process_image_path(path):
    try:
        model = YOLO(model_pt)
        full_path = Path(kaist_images_path) / path
        if not full_path.exists():
            print(f"Path does not exist: {full_path}")
            return

        output_path = Path(yolo_labels_path) / path
        results = model.predict(
            source=str(full_path),
            save=True,
            save_txt=True,
            save_conf=True,
            imgsz=(image_size[1], image_size[0]), # first vertical coord then horizontal
            project=str(output_path.parent),
            name=output_path.name
        )
        print(f"Predicciones completadas para {path}")
    except Exception as e:
        print(f"Error processing {path}: {e}")


if __name__ == '__main__':

    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    image_paths = config['image_paths']
    os.makedirs(yolo_labels_path, exist_ok=True)

    try:
        with mp.Pool(processes=3) as pool:
            pool.map(process_image_path, image_paths)

    except Exception as e:
        print(f"Error during model loading or prediction: {e}")