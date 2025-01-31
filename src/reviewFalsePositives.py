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
from concurrent.futures import ProcessPoolExecutor

from constants import yolo_labels_path_xml, kaist_annotations, fp_cache_path

"""
    Compares whether two labels are from the same instance.
    Have epsilon as a threshold in pixels to consider the 
    labels as equal.
"""
def labelsEqual(label1, label2, epsilon = 10):
    same_class = label1["class"] == label2["class"] 
    same_x = abs(label1["x"] - label2["x"]) < epsilon 
    same_y = abs(label1["y"] - label2["y"]) < epsilon 
    same_w = abs(label1["width"] - label2["width"]) < epsilon
    same_h = abs(label1["height"] - label2["height"]) < epsilon

    return same_class and same_x and same_y and same_w and same_h

def getLabelFromXML(object):
    label = {
        "class": object.name.cdata,
        "x": int(object.bndbox.x.cdata),
        "y": int(object.bndbox.y.cdata),
        "width": int(object.bndbox.w.cdata),
        "height": int(object.bndbox.h.cdata)
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
    Given an XML path and a list of labels, it compares whether the data exists or not
"""
def checkFP(xml_file_path):

    xml_file_name = os.path.basename(xml_file_path)
    kaist_annotation_file = os.paht.join(kaist_annotations,(xml_file_name.split("_")))
    yolo_labels = getLabelsFromFile(xml_file_path)
    kaist_labels = getLabelsFromFile(kaist_annotation_file)
    
    fp_labels_list = []
    for yolo_label in yolo_labels:
        found = False
        for kaist_label in kaist_labels:
            if labelsEqual(yolo_label, kaist_label):
                found = True
                break
        if not found:
            fp_labels_list.append(yolo_label)
    
    return fp_labels_list

def gatherFalsePositives(labels_path):
    def process_file(file):
        if file.endswith(".xml"):
            return file, checkFP(os.path.join(labels_path, file))
        return None
    
    if os.path.exists(fp_cache_path):
        with open(fp_cache_path, 'rb') as f:
            false_positives = pickle.load(f)
    else:
        false_positives = {}

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_file, os.listdir(labels_path))
        for result in results:
            if result is not None:
                file, fp = result
                false_positives[file] = fp

    with open(fp_cache_path, 'wb') as f:
        pickle.dump(false_positives, f)

    return false_positives


if __name__ == '__main__':
    false_positives = gatherFalsePositives(yolo_labels_path_xml)

