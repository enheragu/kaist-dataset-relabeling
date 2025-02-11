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
from pathlib import Path

from utils.parser import countImgLabels
from constants import image_size, yolo_labels_path, yolo_lb_cache_file, yolo_to_kaist_labels, yolo_labels_cache_folder

def processLabelsFolder(folder_path):
    labels_path = os.path.join(folder_path,'labels')
    labels_txt_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    
    labels = {}
    for label_file in labels_txt_files:
        # file format name is: set03_V000_I00000.txt
        
        with open(os.path.join(labels_path, label_file), 'r') as f:
            lines = f.readlines()

        objects = []
        for line in lines:
            class_id, x_center_n, y_center_n, width_n, height_n, confidence = line.split()
            center_x = float(x_center_n) * image_size[0]
            center_y = float(y_center_n) * image_size[1]
            width = float(width_n) * image_size[0]
            height = float(height_n) * image_size[1]

            corner_x = center_x - width / 2
            corner_y = center_y - height / 2
            
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
        
        tag_name = os.path.join(folder_path,label_file).replace(yolo_labels_path,'').replace('/visible','').replace('/lwir','')
        labels[tag_name] = objects
        
    return labels
        
def findLabelsPaths(base_path):
    label_paths = []
    base_path = Path(base_path)

    for root, dirs, files in os.walk(base_path):
        if 'labels' in dirs:
            # Encontramos un directorio 'labels'
            label_path = Path(root)
            # Guardamos la ruta sin incluir 'labels' al final
            label_paths.append(str(label_path))

    return label_paths
if __name__ == '__main__':

    os.makedirs(yolo_labels_cache_folder, exist_ok=True)
    label_directories = findLabelsPaths(yolo_labels_path)
    print(f"Analyze the following paths:" + '\n\t·'.join(label_directories))


    labels = {}
    for path in label_directories:
        labels.update(processLabelsFolder(path))
        
    with open(yolo_lb_cache_file, 'wb') as f:
        pickle.dump(labels, f)

    objects = sum([len(fp) for fp in labels.values()])
    print(f"Processed {len(labels.values())} image files. Found {objects} FP.")
    