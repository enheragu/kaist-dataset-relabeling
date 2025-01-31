#!/usr/bin/env python3
# encoding: utf-8

"""
    Yolo uses a specific labeling format when it is executed as a validation tool for a given dataset,
    such as:
        [class] [x_center] [y_center] [width] [height] [confidence]
    This script takes a given path from output labels with such format and translates them to the XML
    standard compatible with Kaist, something that includes each label as:
        <object>
            <name>[class]</name>
            <bndbox>
                <x>[corner_x]</x>
                <y>[corner_y]</y>
                <w>[width]</w>
                <h>[height]</h>
            </bndbox>
        </object>
    This is a simple translation so no other labels are assigned to each object.
"""

import os

from constants import yolo_labels_path, yolo_labels_path_xml, yolo_to_kaist_labels

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
        </object>
        """
        xml_objects.append(xml_object)
    return "\n".join(xml_objects)

if __name__ == '__main__':

    os.makedirs(yolo_labels_path_xml, exist_ok=True)

    labels_txt_files = [f for f in os.listdir(yolo_labels_path) if f.endswith('.txt')]
    for label_file in labels_txt_files:
        # file format name is: set03_V000_I00000.txt
        
        with open(os.path.join(yolo_labels_path, label_file), 'r') as f:
            lines = f.readlines()

        objects = []
        for line in lines:
            class_id, x_center, y_center, width, height, confidence = line.split()
            corner_x = float(x_center) - float(width) / 2
            corner_y = float(y_center) - float(height) / 2
            
            if class_id not in yolo_to_kaist_labels:
                continue

            objects.append({
                'class': yolo_to_kaist_labels[class_id],
                'corner_x': corner_x,
                'corner_y': corner_y,
                'width': width,
                'height': height
            })
        
        xml_text = convert_to_xml(objects)
        with open(os.path.join(yolo_labels_path_xml, label_file.replace('.txt', '.xml')), 'w') as f:
            f.write(xml_text)