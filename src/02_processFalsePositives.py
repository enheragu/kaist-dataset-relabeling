#!/usr/bin/env python3
# encoding: utf-8

"""
    This script takes as input the labels in XML compatible format from yolo2kaist_format.py and 
    compares them with the labels from the original dataset in order to find false positives.
    For each false positive (with a configurable threshold) the user is asked whether the label 
    should be added to the dataset or just ignored. This way missing labels can be added easily.
"""

import os
import pickle
import untangle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from utils import computeIoU
from constants import yolo_lb_cache_file, kaist_annotations, fp_cache_path, IoU_threshold, regenerate_fpcache_file, kaist_to_yolo_equivalencies

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

def getLabelFromXML(object):
    return {
        "class": object.name.cdata,
        "corner_x": float(object.bndbox.x.cdata),
        "corner_y": float(object.bndbox.y.cdata),
        "width": float(object.bndbox.w.cdata),
        "height": float(object.bndbox.h.cdata)
    }

def getLabelsFromFile(xml_file_path):
    labels = []
    with open(xml_file_path) as xml:
        doc = untangle.parse(xml)
        if hasattr(doc.annotation, "object"):
            for object in doc.annotation.object:
                labels.append(getLabelFromXML(object))
    return labels

"""
    Given an XML path and a list of labels, it compares whether the data exists or not.
"""
def checkFP(filename, labels_data):

    xml_file_name = filename.replace('txt', 'xml')
    kaist_annotation_file = f"{kaist_annotations}/{xml_file_name}"
    yolo_labels = labels_data
    kaist_labels = getLabelsFromFile(kaist_annotation_file)
    # tqdm.write(f"Get Kaist labels from {kaist_annotation_file}")
    # tqdm.write(f"File contains: {kaist_labels}")

    fp_labels_list = []
    for yolo_label in yolo_labels:
        found = False
        for kaist_label in kaist_labels:
            if labelsEqualIoU(yolo_label, kaist_label):
                found = True
                break
        if not found:
            fp_labels_list.append(yolo_label)
    
    # tqdm.write(f"{fp_labels_list} false positives found in {xml_file_name}")
    return fp_labels_list

def process_file(filename, labels_data):
    return filename, checkFP(filename, labels_data)


def gatherFalsePositives(yolo_lb_cache_file):
    with open(yolo_lb_cache_file, 'rb') as f:
        labels_data = pickle.load(f)
    labels_data_n = sum([len(fp) for fp in labels_data.values()])
    tqdm.write(f"Loaded data from {len(labels_data)} image files. Found {labels_data_n} FP.")

    if os.path.exists(fp_cache_path) and not regenerate_fpcache_file:
        with open(fp_cache_path, 'rb') as f:
            false_positives = pickle.load(f)
    else:
        false_positives = {}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, filename, data) for filename, data in labels_data.items()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Image files"):
            result = future.result()
            if result is not None:
                file, fp = result
                false_positives[file] = fp

    with open(fp_cache_path, 'wb') as f:
        pickle.dump(false_positives, f)

    return false_positives


if __name__ == '__main__':
    false_positives = gatherFalsePositives(yolo_lb_cache_file)
    false_positives_n = sum([len(fp) for fp in false_positives.values()])
    tqdm.write(f"Processed {len(false_positives)} image files. Found {false_positives_n} FP.")

