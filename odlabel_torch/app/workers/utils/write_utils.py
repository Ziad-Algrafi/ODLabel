import os
import csv
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom


def save_yolo_format(detections, labels_dir, image_file):
    result_txt_path = os.path.join(labels_dir, f"{os.path.splitext(image_file)[0]}.txt")
    with open(result_txt_path, "w") as txt_file:
        for (
            cls_id,
            bbox_x,
            bbox_y,
            bbox_width,
            bbox_height,
            _,
            _,
            img_width,
            img_height,
        ) in detections:
            normalized_x = bbox_x / img_width
            normalized_y = bbox_y / img_height
            normalized_w = bbox_width / img_width
            normalized_h = bbox_height / img_height
            txt_file.write(f"{cls_id} {normalized_x:.6f} {normalized_y:.6f} {normalized_w:.6f} {normalized_h:.6f}\n")


def save_coco_format(detections, image_file, i, coco_output_file):
    if not os.path.isfile(coco_output_file):
        coco_output = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }
    else:
        with open(coco_output_file, "r") as f:
            coco_output = json.load(f)

    for (
        cls_id,
        bbox_x,
        bbox_y,
        bbox_width,
        bbox_height,
        _,
        _,
        img_width,
        img_height,
    ) in detections:
        coco_output["images"].append(
            {"id": i, "file_name": image_file, "height": img_height, "width": img_width}
        )
        annotation_id = len(coco_output["annotations"]) + 1

        bbox_xmin = int(bbox_x - bbox_width / 2)
        bbox_ymin = int(bbox_y - bbox_height / 2)
        coco_output["annotations"].append(
            {
                "id": annotation_id,
                "image_id": i,
                "category_id": int(cls_id),
                "bbox": [bbox_xmin, bbox_ymin, int(bbox_width), int(bbox_height)],
                "area": int(bbox_width * bbox_height),
                "iscrowd": 0,
            }
        )

    with open(coco_output_file, "w") as f:
        json.dump(coco_output, f)


def save_csv_format(detections, image_file, csv_file_path, csv_headers):
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, "a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        if not file_exists:
            csv_writer.writerow(csv_headers)
        for (
            _,
            bbox_x,
            bbox_y,
            bbox_width,
            bbox_height,
            label_name,
            _,
            img_width,
            img_height,
        ) in detections:
            row_data = [
                label_name,
                float(bbox_x - bbox_width / 2),
                float(bbox_y - bbox_height / 2),
                float(bbox_width),
                float(bbox_height),
                image_file,
                int(img_width),
                int(img_height),
            ]
            csv_writer.writerow(row_data)


def save_xml_format(detections, labels_dir, image_file):
    result_xml_path = os.path.join(labels_dir, f"{os.path.splitext(image_file)[0]}.xml")
    root = ET.Element("annotation")

    ET.SubElement(root, "folder").text = os.path.basename(labels_dir)
    ET.SubElement(root, "filename").text = image_file
    ET.SubElement(root, "path").text = os.path.join(labels_dir, image_file)

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unspecified"

    for (
        _,
        bbox_x,
        bbox_y,
        bbox_width,
        bbox_height,
        label_name,
        _,
        img_width,
        img_height,
    ) in detections:
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(img_width)
        ET.SubElement(size, "height").text = str(img_height)
        ET.SubElement(size, "depth").text = "3"

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = str(label_name)
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(bbox_x - bbox_width / 2))
        ET.SubElement(bndbox, "ymin").text = str(int(bbox_y - bbox_height / 2))
        ET.SubElement(bndbox, "xmax").text = str(int(bbox_x + bbox_width / 2))
        ET.SubElement(bndbox, "ymax").text = str(int(bbox_y + bbox_height / 2))

    xml_str = ET.tostring(root, encoding="utf-8", method="xml")
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
    pretty_xml_str = "\n".join(pretty_xml_str.split("\n")[1:])

    with open(result_xml_path, "w") as f:
        f.write(pretty_xml_str)
