from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2 as cv
import os
import shutil
import json
import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

def infer(model, img, conf_threshold, log_update, image_file):

    annotated_frame= img.copy()
    img_height, img_width = img.shape[:2]
    results = model(img, conf=conf_threshold, imgsz=640)

    detections = []
    names = model.names 
    if results[0].boxes is not None:
        log_update.emit(f"Results for {image_file}:")
        boxes = results[0].boxes.xywhn.cpu()
        clss = results[0].boxes.cls.int().cpu().tolist()
        for box, class_id in zip(boxes, clss):
            x, y, w, h = box
            cls_id = class_id
            class_name = names[cls_id]
            x_scaled = x * img_width
            y_scaled = y * img_height
            w_scaled = w * img_width
            h_scaled = h * img_height

            detections.append((cls_id, x_scaled, y_scaled, w_scaled, h_scaled, class_name))
    else:
        log_update.emit(f"No Results for {image_file}:")

    return detections, annotated_frame

def slice_and_infer(model, img, conf_threshold, log_update, image_file, slice_size):
    annotated_frame = img.copy()
    height, width = img.shape[:2]
    slices = []
    num_slices = 0

    for y in range(0, height, slice_size // 2):
        for x in range(0, width, slice_size // 2):
            slice_y = min(y + slice_size, height)
            slice_x = min(x + slice_size, width)
            slice = img[y:slice_y, x:slice_x]
            slices.append((slice, x, y, slice_x - x, slice_y - y))
            num_slices += 1

    log_update.emit(f"Number of slices: {num_slices}")
    detections = []
    names = model.names 
    for slice, x_offset, y_offset, slice_width, slice_height in slices:
        results = model(slice, conf=conf_threshold, imgsz=slice_size)[0]
        if results.boxes is not None:
            log_update.emit(f"Results for {image_file}:")
            for box in results.boxes:
                x, y, w, h = box.xywhn[0].tolist()
                cls_id = box.cls[0].int().item()
                class_name = names[cls_id]
                x_pixel = x * slice_width
                y_pixel = y * slice_height
                w_pixel = w * slice_width
                h_pixel = h * slice_height
                x_global = x_pixel + x_offset
                y_global = y_pixel + y_offset
                x_scaled = x_global
                y_scaled = y_global
                w_scaled = w_pixel
                h_scaled = h_pixel
                detections.append((cls_id, x_scaled, y_scaled, w_scaled, h_scaled, class_name))
        else:
            log_update.emit(f"No Results for {image_file}:")

    return detections, annotated_frame

def draw_boxes_on_image(annotated_frame, detections, image_path, output_path=None, log_update=None):
    if output_path is None:
        output_path = os.path.splitext(image_path)[0] + '_with_boxes.jpg'

    for cls_id, x_scaled, y_scaled, w_scaled, h_scaled, class_name in detections:
        x1 = int(x_scaled - w_scaled / 2)
        y1 = int(y_scaled - h_scaled / 2)
        x2 = int(x_scaled + w_scaled / 2)
        y2 = int(y_scaled + h_scaled / 2)
        label = class_name
        cv.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(annotated_frame, label, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.imwrite(output_path, annotated_frame)
    if log_update:
        log_update.emit(f"Saved image with bounding boxes to {output_path}")

    return annotated_frame

def save_yolo_format(detections, out_dir, image_file, img):
    result_txt_path = os.path.join(out_dir, f"{os.path.splitext(image_file)[0]}.txt")
    with open(result_txt_path, 'w') as txt_file:
        for cls_id, x_scaled, y_scaled, w_scaled, h_scaled, class_name in detections:
            normalized_x = x_scaled / img.shape[1]
            normalized_y = y_scaled / img.shape[0]
            normalized_w = w_scaled / img.shape[1]
            normalized_h = h_scaled / img.shape[0]
            txt_file.write(f"{cls_id} {normalized_x} {normalized_y} {normalized_w} {normalized_h}\n")

def save_coco_format(detections, image_file, img, i, coco_output):
    coco_output["images"].append({
        "id": i,
        "file_name": image_file,
        "height": img.shape[0],
        "width": img.shape[1]
    })
    annotation_id = len(coco_output["annotations"]) + 1
    for cls_id, x_scaled, y_scaled, w_scaled, h_scaled, class_name in detections:
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

def save_csv_format(detections, image_file, img,  csv_writer):
    for cls_id, x_scaled, y_scaled, w_scaled, h_scaled, class_name in detections:
        csv_writer.writerow([
            class_name,
            float(x_scaled - w_scaled / 2),
            float(y_scaled - h_scaled / 2),
            float(w_scaled),
            float(h_scaled),
            image_file,
            int(img.shape[1]),
            int(img.shape[0])
        ])

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

    for cls_id, x_scaled, y_scaled, w_scaled, h_scaled, class_name in detections:
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

class ObjectDetectionWorker(QThread):
    """Worker thread to run object detection without blocking the UI."""
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    log_update = pyqtSignal(str)
    output_image_signal = pyqtSignal(QPixmap, str)

    def __init__(self, conf_threshold, model_path, images_folder, output_directory, val_split, output_format, classes, device, use_sahi, slice_size):
        super().__init__()
        self.model_path = model_path
        self.images_folder = images_folder
        self.output_directory = output_directory
        self.use_sahi = use_sahi
        self.slice_size = slice_size
        self.val_split = val_split
        self.conf_threshold = conf_threshold
        self.output_format = output_format
        self.classes = classes
        self.device = device
        self._is_running = True

    def run(self):
        import resource
        resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
        select_device(self.device)
        image_files = [f for f in os.listdir(self.images_folder) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]
        if not image_files:
            self.log_update.emit("No images in the folder. Exiting.")
            self.finished.emit()
            return

        num_val_images = int(len(image_files) * self.val_split)

        if num_val_images > 0:
            val_dir = os.path.join(self.output_directory, "val")
            val_images_dir = os.path.join(val_dir, "detected")
            val_labels_dir = os.path.join(val_dir, "labels")
            val_original_dir = os.path.join(val_dir, "images")
            os.makedirs(val_images_dir, exist_ok=True)
            os.makedirs(val_labels_dir, exist_ok=True)
            os.makedirs(val_original_dir, exist_ok=True)

        if num_val_images < len(image_files):
            train_dir = os.path.join(self.output_directory, "train")
            train_images_dir = os.path.join(train_dir, "detected")
            train_labels_dir = os.path.join(train_dir, "labels")
            train_original_dir = os.path.join(train_dir, "images")
            os.makedirs(train_images_dir, exist_ok=True)
            os.makedirs(train_labels_dir, exist_ok=True)
            os.makedirs(train_original_dir, exist_ok=True)

        model = YOLO(self.model_path)
        model.set_classes(self.classes)

        train_coco_output = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": cls_name, "supercategory": cls_name} for i, cls_name in enumerate(self.classes)]
        }
        val_coco_output = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": cls_name, "supercategory": cls_name} for i, cls_name in enumerate(self.classes)]
        }

        for i, image_file in enumerate(image_files):
            if not self._is_running:
                break

            img_path = os.path.join(self.images_folder, image_file)
            img = cv.imread(img_path)

            if self.use_sahi:
                detections, annotated_frame = slice_and_infer(model, img, self.conf_threshold, self.log_update, image_file, self.slice_size)
            else:
                detections, annotated_frame = infer(model, img, self.conf_threshold, self.log_update, image_file)

            if i < num_val_images:
                output_dir = val_dir
                images_dir = val_images_dir
                labels_dir = val_labels_dir
                original_dir = val_original_dir
                coco_output = val_coco_output
                
                if self.output_format == "CSV":
                    val_csv_file_path = os.path.join(val_labels_dir, "annotations.csv")
                    val_csv_file = open(val_csv_file_path, 'a', newline='')
                    val_csv_writer = csv.writer(val_csv_file)
                    if i == 0:
                        val_csv_writer.writerow(['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'image_name', 'image_width', 'image_height'])
            else:
                output_dir = train_dir
                images_dir = train_images_dir
                labels_dir = train_labels_dir
                original_dir = train_original_dir
                coco_output = train_coco_output
                
                if self.output_format == "CSV":
                    train_csv_file_path = os.path.join(train_labels_dir, "annotations.csv")
                    train_csv_file = open(train_csv_file_path, 'a', newline='')
                    train_csv_writer = csv.writer(train_csv_file)
                    if i == num_val_images:
                        train_csv_writer.writerow(['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'image_name', 'image_width', 'image_height'])

            if detections:
                output_img = draw_boxes_on_image(annotated_frame, detections, img_path, os.path.join(images_dir, image_file))

                if self.output_format == "YOLO":
                    save_yolo_format(detections, labels_dir, image_file, img)
                elif self.output_format == "COCO":
                    save_coco_format(detections, image_file, img, i, coco_output)
                elif self.output_format == "CSV":
                    if i < num_val_images:
                        save_csv_format(detections, image_file, img, val_csv_writer)
                    else:
                        save_csv_format(detections, image_file, img, train_csv_writer)
                elif self.output_format == "XML":
                    save_xml_format(self, detections, labels_dir, image_file, images_dir, img)

                original_img_path = os.path.join(original_dir, image_file)
                shutil.copy(img_path, original_img_path)

                self.emit_output_image(output_img, image_file)

            self.progress.emit(int((i + 1) / len(image_files) * 100))            
            self.log_update.emit(f"Processed image: {image_file}")

        if self.output_format == "COCO":
            if num_val_images > 0:
                val_coco_json_path = os.path.join(val_labels_dir, "coco_output.json")
                with open(val_coco_json_path, "w") as val_coco_file:
                    json.dump(val_coco_output, val_coco_file)

            if num_val_images < len(image_files):
                train_coco_json_path = os.path.join(train_labels_dir, "coco_output.json")
                with open(train_coco_json_path, "w") as train_coco_file:
                    json.dump(train_coco_output, train_coco_file)

        if self.output_format == "CSV":
            if 'val_csv_file' in locals():
                val_csv_file.close()
            if 'train_csv_file' in locals():
                train_csv_file.close()

        self.log_update.emit("Results Finished.")
        self.finished.emit()

    def emit_output_image(self, image, image_file):
        """Emit the processed image to the GUI."""
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.output_image_signal.emit(pixmap, image_file)
