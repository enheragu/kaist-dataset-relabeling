#!/usr/bin/env python3
# encoding: utf-8

"""
    This script takes as input the labels from 01_parseYoloLabels.py stored as a pkl file and 
    compares them with the labels from the original dataset in order to find false negatives
    in the detection. This way poorly labeled items in the dataset can be removed.
"""

from utils.parser import getLabelsFromFile
from utils.labels_compare import labelsEqualIoU, processLabels
from constants import yolo_lb_cache_file, kaist_annotations, fn_cache_path, regenerate_fncache_file


"""
    Given an XML path and a list of labels, it compares whether the label from XML
    original annotation has been detected or not.
"""
def checkFN(filename, labels_data):

    xml_file_name = filename.replace('txt', 'xml')
    kaist_annotation_file = f"{kaist_annotations}/{xml_file_name}"
    yolo_labels = labels_data
    kaist_labels = getLabelsFromFile(kaist_annotation_file)

    fn_labels_list = []
    for kaist_label in kaist_labels:
        found = False
        for yolo_label in yolo_labels:
            if labelsEqualIoU(yolo_label, kaist_label):
                found = True
                break
        if not found:
            fn_labels_list.append(kaist_label)
    
    return fn_labels_list

def process_file(filename, labels_data):
    return filename, checkFN(filename, labels_data)

def gatherFalseNegatives(yolo_lb_cache_file):
    return processLabels(yolo_lb_cache_file, cache_path=fn_cache_path, process_file_func=process_file, regenerate_cache_file=regenerate_fncache_file)


if __name__ == '__main__':
    false_negatives = gatherFalseNegatives(yolo_lb_cache_file)
    false_negatives_n = sum([len(fp) for fp in false_negatives.values()])
    print(f"Processed {len(false_negatives)} image files. Found {false_negatives_n} FN.")

