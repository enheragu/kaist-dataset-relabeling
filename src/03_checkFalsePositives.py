#!/usr/bin/env python3
# encoding: utf-8

"""
    This script reviewvs false positives and displays each of them for a
    user to accept or reject the new label inclusion.
"""


import os
import pickle
import cv2 as cv2

from utils.ploting import getLabelCrop
from utils.labels_compare import filterNMS
from constants import fp_cache_path, confidence_threshold


def storeStatus(false_positives):
    with open(fp_cache_path, 'wb') as f:
        pickle.dump(false_positives, f)

def finishExecution(false_positives):
    storeStatus(false_positives)

    cv2.destroyAllWindows()
    print(f"[INFO] False positives stored in {fp_cache_path}:")
    stored_items = []
    for image in false_positives.keys():
        for fp in false_positives[image]:
            if 'stored' in fp:
                stored_items.append(fp['stored'])
    positives_stored = sum(stored_items)
    positives_discarded = sum([not x for x in stored_items])

    print(f"\t· {positives_stored} false positives stored.")
    print(f"\t· {positives_discarded} false positives discarded.")


def get_labeling_mode():
    while True:
        mode = input("¿Labels tagged are to be in LWIR or in Visible spectrum? (visible/lwir): ").lower()
        if mode == "visible":
            return True
        elif mode == "lwir":
            return False
        else:
            print("Invalid opction. Pleas introduce either: 'visible' or 'lwir'.")

if __name__ == '__main__':
    
    with open(fp_cache_path, 'rb') as f:
        false_positives = pickle.load(f)

    print(f"Confidence threshold: {confidence_threshold}")
    images_fp_filtered = {}

    lablVisible = get_labeling_mode()
    # Just for logging
    for file_name in false_positives.keys():
        fp_data = false_positives[file_name]
        nms_filtered = filterNMS(fp_data, confidence_threshold=confidence_threshold)
        if nms_filtered:
            images_fp_filtered[file_name] = nms_filtered

    nms_false_positives_n = sum([len(fp) for fp in images_fp_filtered.values()])
    print(f"NMS filtered to {len(images_fp_filtered)} image files with {nms_false_positives_n} FP.")
    print(f"Labeling is made with {'visible' if lablVisible else 'LWIR'} sectrum as reference.")

    index_autosave = 0
    for file_name in false_positives.keys():
        fp_data = false_positives[file_name]
        fp_data = filterNMS(fp_data, confidence_threshold=confidence_threshold)
        for i, fp in enumerate(fp_data):
            index_autosave += 1

            if index_autosave > 20:
                index_autosave = 0
                storeStatus(false_positives)
                print(f"Autosaved :)")

            if 'stored' in fp:
                continue

            image, label_crop = getLabelCrop(file_name=file_name, label=fp, label_visible=lablVisible)
            
            if image is None or image.size == 0 or \
               label_crop is None or label_crop.size == 0:
                print(f"[ERROR] Invalid image for {file_name}")
                continue

            cv2.imshow(f"FP Label", label_crop)           
            cv2.imshow(f"Image", image)           

            print(f"Press 's' to store the false positive, 'n' to discard it or 'q/Esc' to exit. Press 'c' to change labeling mode between Visible and LIWIR.")
            
            while True:
                key = cv2.waitKey(0)

                if key == ord('s') or key == ord('S'):  # Guardar puntos y pasar al siguiente par
                    print(f"- Stored FP for {file_name}.")
                    false_positives[file_name][i]['stored'] = True
                    false_positives[file_name][i]['lablVisible'] = lablVisible
                    break
                
                elif key == ord('n') or key == ord('N'):  # Ignore
                    print(f"- Discarded FP for {file_name}.")
                    false_positives[file_name][i]['stored'] = False
                    break
                    
                elif key == ord('c') or key == ord('C'):
                    lablVisible = get_labeling_mode()
                    print(f"Labeling is made with {'visible' if lablVisible else 'LWIR'} sectrum as reference.")


                elif key == 27 or key == ord('q') or key == ord('Q'):  # Esc para salir
                    finishExecution(false_positives)
                    exit()
        

    nms_false_positives_n = sum([len(fp) for fp in false_positives.values()])
    print(f"Processed {len(false_positives)} image files. Found {nms_false_positives_n} FP.")
    print(f"Finished processing all false positives found with {confidence_threshold} confidence threshold.")
    finishExecution(false_positives)