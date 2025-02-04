#!/usr/bin/env python3
# encoding: utf-8

"""
    Yolo uses a specific labeling format when it is executed as a validation tool for a given dataset,
    such as:
        [class] [x_center] [y_center] [width] [height] [confidence]
    This script parses all the labels data and store it in a dict format with all labels and associated
    file.
"""

import os
import pickle

from utils import computeIoU
from constants import image_size, yolo_labels_path, yolo_lb_cache_file, yolo_to_kaist_labels, yolo_labels_cache_folder


if __name__ == '__main__':

    os.makedirs(yolo_labels_cache_folder, exist_ok=True)
    labels_txt_files = [f for f in os.listdir(yolo_labels_path) if f.endswith('.txt')]
    
    labels = {}
    for label_file in labels_txt_files:
        # file format name is: set03_V000_I00000.txt
        
        with open(os.path.join(yolo_labels_path, label_file), 'r') as f:
            lines = f.readlines()

        objects = []
        for line in lines:
            class_id, x_center, y_center, width, height, confidence = line.split()
            corner_x = (float(x_center) - float(width) / 2)*image_size[0]
            corner_y = (float(y_center) - float(height) / 2)*image_size[1]
            width = float(width)*image_size[0]
            height = float(height)*image_size[1]
            
            class_id = int(class_id)
            if class_id not in yolo_to_kaist_labels:
                # print(f"Ignoring class {class_id}; not in configured labels dict: {yolo_to_kaist_labels}")
                continue

            objects.append({
                'class': yolo_to_kaist_labels[class_id],
                'corner_x': float(corner_x),
                'corner_y': float(corner_y),
                'width': float(width),
                'height': float(height),
                'confidence': float(confidence)
            })
            
        labels[label_file] = objects
        
        
    with open(yolo_lb_cache_file, 'wb') as f:
        pickle.dump(labels, f)

    objects_labeled = sum([len(objects) for objects in labels.values()])
    print(f"Found {objects_labeled} objects labeled by YOLO in {len(labels.keys())} images.")
    