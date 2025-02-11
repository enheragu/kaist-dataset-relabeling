#!/usr/bin/env python3
# encoding: utf-8

import untangle


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

def getLabelFromXML(object):
    return {
        "class": object.name.cdata,
        "corner_x": float(object.bndbox.x.cdata),
        "corner_y": float(object.bndbox.y.cdata),
        "width": float(object.bndbox.w.cdata),
        "height": float(object.bndbox.h.cdata)
    }

def getLabelsFromFile(xml_file_path):
    labels = []
    with open(xml_file_path) as xml:
        doc = untangle.parse(xml)
        if hasattr(doc.annotation, "object"):
            for object in doc.annotation.object:
                labels.append(getLabelFromXML(object))
    return labels

def countImgLabels(labels):
    img_n = 0
    obj_n = 0
    for set_v in labels.values():
        for img in set_v.values():
            img_n += 1
            obj_n += sum([len(objects) for objects in img])

    return img_n, obj_n