#!/usr/bin/env python3
# encoding: utf-8

from constants import IoU_threshold, confidence_threshold

def convert_to_xml(objects):
    xml_objects = []
    for obj in objects:
        xml_object = f"""
        <object>
            <name>{obj['class']}</name>
            <bndbox>
                <x>{obj['corner_x']}</x>
                <y>{obj['corner_y']}</y>
                <w>{obj['width']}</w>
                <h>{obj['height']}</h>
            </bndbox>
            <lablVisible>{obj['lablVisible']}</lablVisible>
        </object>
        """
        xml_objects.append(xml_object)
    return "\n".join(xml_objects)


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

def countImgLabels(labels):
    img_n = 0
    obj_n = 0
    for set_v in labels.values():
        for img in set_v.values():
            img_n += 1
            obj_n += sum([len(objects) for objects in img])

    return img_n, obj_n