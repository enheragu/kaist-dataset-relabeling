#!/usr/bin/env python3
# encoding: utf-8


import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from constants import IoU_threshold, confidence_threshold


def computeIoU(label1, label2):

    x1 = max(label1["corner_x"], label2["corner_x"])
    y1 = max(label1["corner_y"], label2["corner_y"])
    x2 = min(label1["corner_x"]+ label1["width"], label2["corner_x"]+label2["width"])
    y2 = min(label1["corner_y"]+ label1["height"], label2["corner_y"]+label2["height"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    arealb1 = label1["width"] * label1["height"]
    arealb2 = label2["width"] * label2["height"]
    union = float(arealb1 + arealb2 - intersection)

    return (intersection / union) if union > 0 else 0


def filterNMS(objects, iou_threshold=IoU_threshold, confidence_threshold=confidence_threshold):
    sorted_objects = sorted(objects, key=lambda x: x['confidence'], reverse=True)
    sorted_objects = [obj for obj in sorted_objects if obj['confidence'] > confidence_threshold]
    
    kept_objects = []
    while sorted_objects:
        current = sorted_objects.pop(0)
        kept_objects.append(current)
        
        sorted_objects = [
            obj for obj in sorted_objects
            if computeIoU(current, obj) < iou_threshold or current['class'] != obj['class']
        ]
    
    return kept_objects


"""
    Compares whether two labels are from the same instance.
    Have epsilon as a threshold in pixels to consider the labels as equal.
"""
def labelsEqual(label1, label2, epsilon = 10):
    same_class = label1["class"] == label2["class"] 
    same_x = abs(label1["corner_x"] - label2["corner_x"]) < epsilon 
    same_y = abs(label1["corner_y"] - label2["corner_y"]) < epsilon 
    same_w = abs(label1["width"] - label2["width"]) < epsilon
    same_h = abs(label1["height"] - label2["height"]) < epsilon

    return same_class and same_x and same_y and same_w and same_h

"""
    Is label1 inside label2?
"""
def isInside(label1, label2, epsilon=1e-5):
    l1_min = np.array([label1["corner_x"], label1["corner_y"]])
    l1_max = l1_min + np.array([label1["width"], label1["height"]])

    l2_min = np.array([label2["corner_x"], label2["corner_y"]])
    l2_max = l2_min + np.array([label2["width"], label2["height"]])

    inside = np.all(l1_min >= l2_min - epsilon) and np.all(l1_max <= l2_max + epsilon)
    return inside


"""
    Compares whether two labels are from the same instance.
    True if class label matches and IoU between bot is above the threshold.
"""
def labelsEqualIoU(label1, label2, IoU = IoU_threshold):
    global kaist_to_yolo_equivalencies
    if label1["class"] in kaist_to_yolo_equivalencies:
        label1["class"] = kaist_to_yolo_equivalencies[label1["class"]]
    if label2["class"] in kaist_to_yolo_equivalencies:
        label2["class"] = kaist_to_yolo_equivalencies[label2["class"]]

    same_class = label1["class"] == label2["class"] 
    iou = computeIoU(label1, label2)
    l1Insidel2 = isInside(label1, label2)
    l2Insidel1 = isInside(label2, label1)
    
    return (iou > IoU or l1Insidel2 or l2Insidel1) and same_class



def processLabels(yolo_lb_cache_file, cache_path, process_file_func, regenerate_cache_file = True):
    with open(yolo_lb_cache_file, 'rb') as f:
        labels_data = pickle.load(f)
    labels_data_n = sum([len(fp) for fp in labels_data.values()])
    tqdm.write(f"Loaded data from {len(labels_data)} image files. Found {labels_data_n} labels.")

    if os.path.exists(cache_path) and not regenerate_cache_file:
        with open(cache_path, 'rb') as f:
            processed_labels = pickle.load(f)
    else:
        processed_labels = {}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_func, filename, data) for filename, data in labels_data.items()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Image files"):
            result = future.result()
            if result is not None:
                file, fp = result
                processed_labels[file] = fp

    with open(cache_path, 'wb') as f:
        pickle.dump(processed_labels, f)

    return processed_labels