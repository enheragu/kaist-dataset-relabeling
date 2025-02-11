#!/usr/bin/env python3
# encoding: utf-8

"""
    With a curated FP list file (processed with 03_checkFalsePositives.py) it
    integrates all the new labels into the XML format of KAIST dataset.
"""

import os
import pickle
import xml.etree.ElementTree as ET

from constants import fp_cache_path, kaist_images_path, label_color, class_color, confidence_threshold, kaist_annotations


def labelIntoXML(file_path, label_list):
    xml_complete_path = f"{kaist_annotations}/{file_path}".replace('txt','xml')
    
    if not os.path.exists(xml_complete_path):
        print(f"[ERROR] XML file does not exist: {xml_complete_path}")
        return
    
    try:
        with open(xml_complete_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"[ERROR] Failed to read XML file {xml_complete_path}: {str(e)}")
        return

    modified = False
    insert_index = next(i for i in range(len(lines)-1, -1, -1) if '</annotation>' in lines[i])

    for label in label_list:
        if not 'stored' in label or not label['stored']:
            continue

        new_object = f"""  <object>
    <name>{label['class']}</name>
    <bndbox>
      <x>{int(label['corner_x'])}</x>
      <y>{int(label['corner_y'])}</y>
      <w>{int(label['width'])}</w>
      <h>{int(label['height'])}</h>
    </bndbox>
    <pose>unknown</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <occlusion>0</occlusion>
    <label>{"visible" if label['lablVisible'] else "lwir"}</label>
  </object>
"""
        lines.insert(insert_index, new_object)
        modified = True
    
    if modified:
        try:
            with open(xml_complete_path, 'w', encoding='utf-8') as file:
                file.writelines(lines)
            print(f"[INFO] Successfully updated {xml_complete_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write to XML file {xml_complete_path}: {str(e)}")
    # else:
    #     print(f"[INFO] No changes made to {xml_complete_path}")

if __name__ == '__main__':
    with open(fp_cache_path, 'rb') as f:
        false_positives = pickle.load(f)

    processed_fps = []
    for file, fps in false_positives.items():
        # try:
            labelIntoXML(file, fps)
            processed_fps.append((file, fps))
        # except Exception as e:
        #     print(f"Error procesando {file}: {str(e)}")
    
    false_positives = {k: v for k, v in false_positives.items() if (k, v) not in processed_fps}
    with open(fp_cache_path, 'wb') as f:
        pickle.dump(false_positives, f)