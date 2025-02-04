#!/usr/bin/env python3
# encoding: utf-8

"""
    Just a python file with some constants to be used in the project and in
    its different scripts.
"""
import os

# YOLO labels are normalized to image size
image_size = (640, 512)

# Paths to labels exported by YOLO and its intermediate XML representation
yolo_labels_path = 'yolo_labels'
yolo_labels_cache_folder = 'yolo_labels_format'
yolo_lb_cache_file = os.path.join(yolo_labels_cache_folder,'labels_cache.pkl')
# Mapping from YOLO labels to KAIST labels. If label is not in the mapping
# it will be ignored
yolo_to_kaist_labels = {0: 'person'}
class_color = {  'person': (114,196,83), 'person?': (70,133,46), 'cyclist': (26,209,226), 'people': (229,29,46) }
label_color = (239,184,16) # Different color for label proposaln

# Paths to images and labels from the dataset
kaist_images_path = 'kaist-cvpr15/images'
kaist_annotations = 'annotations-xml-new'


# Cache file with the false positives detected so to not chech every time
# the script is run
fp_cache_path = os.path.join(yolo_labels_cache_folder,'false_positives_cache.pkl')
# Ignore previoues cached file and regenerate it
regenerate_fpcache_file = True

# Threshold to consider True/False Positives
IoU_threshold = 0.6
confidence_threshold = 0.8