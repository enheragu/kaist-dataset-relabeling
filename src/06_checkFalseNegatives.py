#!/usr/bin/env python3
# encoding: utf-8

"""
    This script reviewvs false negatives and displays each of them for a
    user to accept or reject the new label inclusion.
"""


import os
import pickle
import cv2 as cv2

from utils.ploting import getLabelCrop
from utils.labels_compare import filterNMS
from constants import fn_cache_path, confidence_threshold


def storeStatus(false_negatives):
    with open(fn_cache_path, 'wb') as f:
        pickle.dump(false_negatives, f)

def finishExecution(false_negatives):
    storeStatus(false_negatives)

    cv2.destroyAllWindows()
    print(f"[INFO] False negatives stored in {fn_cache_path}:")
    removed_items = []
    for image in false_negatives.keys():
        for fn in false_negatives[image]:
            if 'removed' in fn:
                removed_items.append(fn['removed'])
    negatives_removed = sum(removed_items)
    negatives_kept = sum([not x for x in removed_items])

    print(f"\t· {negatives_removed} false negatives removed.")
    print(f"\t· {negatives_kept} false negatives kept.")


if __name__ == '__main__':
    
    with open(fn_cache_path, 'rb') as f:
        false_negatives = pickle.load(f)

    print(f"Confidence threshold: {confidence_threshold}")
    images_fn_filtered = {}

    # Just for logging
    for file_name in false_negatives.keys():
        fn_data = false_negatives[file_name]
        nms_filtered = filterNMS(fn_data, confidence_threshold=confidence_threshold)
        if nms_filtered:
            images_fn_filtered[file_name] = nms_filtered

    nms_false_negatives_n = sum([len(fn) for fn in images_fn_filtered.values()])
    print(f"NMS filtered to {len(images_fn_filtered)} image files with {nms_false_negatives_n} FN.")
    
    index_autosave = 0
    for file_name in false_negatives.keys():
        fn_data = false_negatives[file_name]
        fn_data = filterNMS(fn_data, confidence_threshold=confidence_threshold)
        for i, fn in enumerate(fn_data):
            index_autosave += 1

            if index_autosave > 20:
                index_autosave = 0
                storeStatus(false_negatives)
                print(f"Autosaved :)")

            if 'removed' in fn:
                continue

            image, label_crop = getLabelCrop(file_name=file_name, label=fn, label_visible=False)
            
            if image is None or image.size == 0 or \
               label_crop is None or label_crop.size == 0:
                print(f"[ERROR] Invalid image for {file_name}")
                continue

            cv2.imshow(f"fn Label", label_crop)           
            cv2.imshow(f"Image", image)           

            print(f"Press 'r' to remove the false negative, 'n' to keep it it or 'q/Esc' to exit.")
            
            while True:
                key = cv2.waitKey(0)

                if key == ord('r') or key == ord('R'):  # Guardar puntos y pasar al siguiente par
                    print(f"- Removed FN for {file_name}.")
                    false_negatives[file_name][i]['removed'] = True
                    break
                
                elif key == ord('n') or key == ord('N'):  # Ignore
                    print(f"- Kept FN for {file_name}.")
                    false_negatives[file_name][i]['removed'] = False
                    break
                    
                elif key == 27 or key == ord('q') or key == ord('Q'):  # Esc para salir
                    finishExecution(false_negatives)
                    exit()

    nms_false_negatives_n = sum([len(fn) for fn in false_negatives.values()])
    print(f"Processed {len(false_negatives)} image files. Found {nms_false_negatives_n} FN.")
    print(f"Finished processing all false negatives found with {confidence_threshold} confidence threshold.")
    finishExecution(false_negatives)