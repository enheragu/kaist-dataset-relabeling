#!/usr/bin/env python3
# encoding: utf-8

"""
    This script takes as input the labels from 01_parseYoloLabels.py stored as a pkl file and 
    compares them with the labels from the original dataset in order to find false positives
    in the detection. This way unlabeled items can be added to the dataset.
"""

from utils.parser import getLabelsFromFile
from utils.labels_compare import labelsEqualIoU, processLabels
from constants import yolo_lb_cache_file, kaist_annotations, fp_cache_path, regenerate_fpcache_file


"""
    Given an XML path and a list of labels, it compares whether the data exists or not.
"""
def checkFP(filename, labels_data):

    xml_file_name = filename.replace('txt', 'xml')
    kaist_annotation_file = f"{kaist_annotations}/{xml_file_name}"
    yolo_labels = labels_data
    kaist_labels = getLabelsFromFile(kaist_annotation_file)

    fp_labels_list = []
    for yolo_label in yolo_labels:
        found = False
        for kaist_label in kaist_labels:
            if labelsEqualIoU(yolo_label, kaist_label):
                found = True
                break
        if not found:
            fp_labels_list.append(yolo_label)
    
    return fp_labels_list

def process_file(filename, labels_data):
    return filename, checkFP(filename, labels_data)


def gatherFalsePositives(yolo_lb_cache_file):
    return processLabels(yolo_lb_cache_file, cache_path=fp_cache_path, process_file_func=process_file, regenerate_cache_file=regenerate_fpcache_file)

if __name__ == '__main__':
    false_positives = gatherFalsePositives(yolo_lb_cache_file)
    false_positives_n = sum([len(fp) for fp in false_positives.values()])
    print(f"Processed {len(false_positives)} image files. Found {false_positives_n} FP.")

