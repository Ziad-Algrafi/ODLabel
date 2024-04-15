import os
import csv
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom


def save_yolo_format(detections, out_dir, image_file, img):
    result_txt_path = os.path.join(out_dir, f"{os.path.splitext(image_file)[0]}.txt")
    with open(result_txt_path, 'w') as txt_file:
        for cls_id, x_scaled, y_scaled, w_scaled, h_scaled, class_name, confidence in detections:
            normalized_x = x_scaled / img.shape[1]
            normalized_y = y_scaled / img.shape[0]
            normalized_w = w_scaled / img.shape[1]
            normalized_h = h_scaled / img.shape[0]
            txt_file.write(f"{cls_id} {normalized_x} {normalized_y} {normalized_w} {normalized_h}\n")


def save_coco_format(detections, image_file, img, i, coco_output_file):
    if not os.path.isfile(coco_output_file):
        coco_output = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
    else:
        with open(coco_output_file, 'r') as f:
            coco_output = json.load(f)

    coco_output["images"].append({
        "id": i,
        "file_name": image_file,
        "height": img.shape[0],
        "width": img.shape[1]
    })
    annotation_id = len(coco_output["annotations"]) + 1
    for cls_id, x_scaled, y_scaled, w_scaled, h_scaled, class_name, confidence in detections:
        x_min = int(x_scaled - w_scaled / 2)
        y_min = int(y_scaled - h_scaled / 2)
        width = int(w_scaled)
        height = int(h_scaled)
        coco_output["annotations"].append({
            "id": annotation_id,
            "image_id": i,
            "category_id": int(cls_id),
            "bbox": [x_min, y_min, width, height],
            "area": width * height,
            "iscrowd": 0
        })
        annotation_id += 1

    with open(coco_output_file, 'w') as f:
        json.dump(coco_output, f)

def save_csv_format(detections, image_file, img, file_path, headers):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if not file_exists:
            csv_writer.writerow(headers)
        for cls_id, x_scaled, y_scaled, w_scaled, h_scaled, class_name, confidence in detections:
            row_data = [
                class_name,
                float(x_scaled - w_scaled / 2),
                float(y_scaled - h_scaled / 2),
                float(w_scaled),
                float(h_scaled),
                image_file,
                int(img.shape[1]),
                int(img.shape[0])
            ]
            csv_writer.writerow(row_data)

def save_xml_format(self, detections, out_dir, image_file, images_dir, img):
    result_xml_path = os.path.join(out_dir, f"{os.path.splitext(image_file)[0]}.xml")

    root = ET.Element('annotation')

    ET.SubElement(root, 'folder').text = out_dir 
    ET.SubElement(root, 'filename').text = image_file
    ET.SubElement(root, 'path').text = os.path.join(images_dir, image_file)  

    source_element = ET.SubElement(root, 'source')
    ET.SubElement(source_element, 'database').text = 'Unspecified'

    size_element = ET.SubElement(root, 'size')
    ET.SubElement(size_element, 'width').text = str(img.shape[1])
    ET.SubElement(size_element, 'height').text = str(img.shape[0])
    ET.SubElement(size_element, 'depth').text = str(img.shape[2])

    for cls_id, x_scaled, y_scaled, w_scaled, h_scaled, class_name, confidence in detections:
        object_element = ET.SubElement(root, 'object')
        ET.SubElement(object_element, 'name').text = str(self.classes[cls_id])  
        ET.SubElement(object_element, 'pose').text = 'Unspecified'
        ET.SubElement(object_element, 'truncated').text = '0'
        ET.SubElement(object_element, 'difficult').text = '0'

        bndbox_element = ET.SubElement(object_element, 'bndbox')
        ET.SubElement(bndbox_element, 'xmin').text = str(int(x_scaled - w_scaled / 2))
        ET.SubElement(bndbox_element, 'ymin').text = str(int(y_scaled - h_scaled / 2))
        ET.SubElement(bndbox_element, 'xmax').text = str(int(x_scaled + w_scaled / 2))
        ET.SubElement(bndbox_element, 'ymax').text = str(int(y_scaled + h_scaled / 2))

    xml_str = ET.tostring(root, encoding='utf-8', method='xml')
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
    pretty_xml_str = '\n'.join(pretty_xml_str.split('\n')[1:])
    with open(result_xml_path, 'w') as f:
        f.write(pretty_xml_str)