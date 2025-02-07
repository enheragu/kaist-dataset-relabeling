#!/usr/bin/env python3
# encoding: utf-8

import os
import pickle
import untangle
import cv2 as cv2

from utils import filterNMS
from constants import fp_cache_path, kaist_images_path, label_color, class_color, confidence_threshold, kaist_annotations

# alpha -> Original labels oppacity
def labelOriginalDataImage(xml_path, image_path, alpha = 0.5):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not open image {image_path}")

    with open(xml_path) as xml:
        doc = untangle.parse(xml)
        if hasattr(doc.annotation, "object"):
            for object in doc.annotation.object:
                obj_name = object.name.cdata

                if obj_name == "person?":
                    continue

                start_point = (int(object.bndbox.x.cdata), int(object.bndbox.y.cdata))
                end_point = (int(object.bndbox.x.cdata) + int(object.bndbox.w.cdata), 
                             int(object.bndbox.y.cdata) + int(object.bndbox.h.cdata))

                extra_data = "|truncated|" if object.truncated.cdata is True else ""  
                extra_data += "|difficult|" if object.difficult.cdata is True else ""  
                extra_data += "|occlusion|" if object.occlusion.cdata is True else ""
                extra_data = extra_data.replace("||","|")

                overlay = image.copy()
                cv2.rectangle(overlay, start_point, end_point, color=class_color[obj_name], thickness=1)
                label_str = f"label: {obj_name} {extra_data}"
                
                # For the text background
                # Finds space required by the text so that we can put a background with that amount of width.
                (w, h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Prints the text.    
                overlay = cv2.rectangle(overlay, (start_point[0], start_point[1]-h-8), (start_point[0]+w+4, start_point[1]), class_color[obj_name], -1)
                overlay = cv2.putText(overlay, label_str, (start_point[0]+4, int(start_point[1]-h/2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
    return image



label_image_margin = 30 # in pixels
def getLabelCrop(file_name, label):
    image_name = f"{kaist_images_path}/{file_name.replace('txt','jpg')}"
    image_file_name = os.path.basename(image_name)
    image_file_path = os.path.dirname(image_name)

    xml_original_labels_path = f"{kaist_annotations}/{file_name.replace('txt','xml')}"
    visible_image_path = os.path.join(image_file_path, 'visible', image_file_name)
    lwir_image_path = os.path.join(image_file_path, 'lwir', image_file_name)
    
    output_crops = []
    for image_path in [lwir_image_path, visible_image_path]:
        image = labelOriginalDataImage(xml_original_labels_path, image_path)
        label_x, label_y, label_w, label_h = label["corner_x"], label["corner_y"], label["width"], label["height"]
        
        ## Add class label to the image for an easyer debugging :)   
        start_point = (int(label_x), int(label_y))
        end_point = (int(label_x + label_w), int(label_y + label_h))
        cv2.rectangle(image, start_point, end_point, color=label_color, thickness=1)
        
        
        class_str = f'pred: {label["class"]}'
        conf_str =  f'       ({label["confidence"]:.3f})'
        (class_w, class_h), _ = cv2.getTextSize(class_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        (conf_w, conf_h), _ = cv2.getTextSize(conf_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        max_w = max(class_w, conf_w)
        # Prints the text.    
        image = cv2.rectangle(image, 
                            (int(start_point[0]), int(start_point[1]-class_h-conf_h-9)), 
                            (int(start_point[0]+max_w+6), int(start_point[1])), 
                            label_color, -1)
        image = cv2.putText(image, class_str, 
                            (int(start_point[0]+3), int(start_point[1]-conf_h-class_h//2-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        image = cv2.putText(image, conf_str, 
                            (int(start_point[0]+3), int(start_point[1]-conf_h//2-1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        label_crop = image[int(label_y-label_image_margin-class_h-conf_h):int(label_y+label_h+label_image_margin), 
                           int(label_x-label_image_margin):int(label_x+max_w+label_image_margin)]
        
        output_crops.append(label_crop)
        # print(f"Person labeled in {image_path} at {x,y} with {w,h}")
        # cv2.imshow("image", image)
        # key = cv2.waitKey(0)
        # if key == 27 or key == ord('q') or key == ord('Q'):  # Esc para salir
        #     cv2.destroyAllWindows()
        #     exit()

    mosaic = cv2.hconcat(output_crops)
    return image, mosaic

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

            image, label_crop = getLabelCrop(file_name, fp)
            
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
    print(f"Processed {len(nms_false_positives_n)} image files. Found {nms_false_positives_n} FP.")
    print(f"Finished processing all false positives found with {confidence_threshold} confidence threshold.")
    finishExecution(false_positives)