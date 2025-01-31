#!/usr/bin/env python3
# encoding: utf-8

import os
import cv2 as cv2

from reviewFalsePositives import gatherFalsePositives
from constants import yolo_labels_path_xml, kaist_annotations, kaist_images_path, class_color

label_image_margin = 30 # in pixels
def getLabelCrop(image_path, label):
    image = cv2.imread(image_path)
    x, y, w, h = label["x"], label["y"], label["width"], label["height"]
    label_crop = image[y-label_image_margin:y+h+label_image_margin, x-label_image_margin:x+w+label_image_margin]
    
    ## Add class label to the image for an easyer debugging :)   
    start_point = (int(x), int(y))
    end_point = (int(x + w/2), int(y + h/2))
    cv2.rectangle(label_crop, start_point, end_point, color=class_color[label["class"]], thickness=1)
    
    label_str = f"{label["class"]}"
    (w, h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Prints the text.    
    label_crop = cv2.rectangle(label_crop, (int(start_point[0]), int(start_point[1]-h-8)), (int(start_point[0]+w+4), int(start_point[1])), class_color[obj_name], -1)
    label_crop = cv2.putText(label_crop, label_str, (int(start_point[0]+4), int(start_point[1]-h//2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
 
    return label_crop

if __name__ == '__main__':
    false_positives = gatherFalsePositives(yolo_labels_path_xml)

    for xml_file_name, fp_data in false_positives.items():
        image_name = os.paht.join(kaist_annotations,(xml_file_name.split("_").replace('xml','jpg')))
        label_crop = getLabelCrop(image_name, fp_data)