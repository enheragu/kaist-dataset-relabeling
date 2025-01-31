#!/usr/bin/env python3
# encoding: utf-8

"""
    Just a python file with some constants to be used in the project and in
    its different scripts.
"""
import os

# Paths to labels exported by YOLO and its intermediate XML representation
yolo_labels_path = '../yolo_labels'
yolo_labels_path_xml = '../yolo_labels_kaist_format'

# Paths to images and labels from the dataset
kaist_images_path = '../images'
kaist_annotations = '../annotations-xml-new'

# Cache file with the false positives detected so to not chech every time
# the script is run
fp_cache_path = os.path.join(yolo_labels_path_xml,'false_positives_cache.pkl')
