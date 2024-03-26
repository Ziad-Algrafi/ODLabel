from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
import os
import shutil
import json
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

def detection(model, img, conf_split):
    results = model(img, conf=conf_split, imgsz=640)
    img_height, img_width = img.shape[:2]
    annotated_frame = results[0].plot()

    detections = []
    if results[0].boxes is not None:
        boxes = results[0].boxes.xywhn.cpu()
        clss = results[0].boxes.cls.int().cpu().tolist()
        for box, track_cls in zip(boxes, clss):
            x, y, w, h = box
            cls = track_cls
            x_scaled = x * img_width
            y_scaled = y * img_height
            w_scaled = w * img_width
            h_scaled = h * img_height
            detections.append((cls, x_scaled, y_scaled, w_scaled, h_scaled))

    return detections, annotated_frame

def draw_boxes_on_image(image, detections, image_path, annotated_frame, output_path=None):
    if output_path is None:
        output_path = os.path.splitext(image_path)[0] + '_with_boxes.jpg'
    cv2.imwrite(output_path, annotated_frame)
    print(f"Saved image with bounding boxes to {output_path}")
    return annotated_frame

def save_yolo_format(detections, out_dir, image_file, img):
    result_txt_path = os.path.join(out_dir, f"{os.path.splitext(image_file)[0]}.txt")
    with open(result_txt_path, 'a') as txt_file:
        for cls, x, y, w, h in detections:
            normalized_x = x / img.shape[1]
            normalized_y = y / img.shape[0]
            normalized_w = w / img.shape[1]
            normalized_h = h / img.shape[0]
            txt_file.write(f"{cls} {normalized_x} {normalized_y} {normalized_w} {normalized_h}\n")

def save_coco_format(detections, out_dir, image_file, img, i, model):
    coco_output = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    for category in model.names:
        coco_output["categories"].append({
            "id": model.names.index(category),
            "name": category,
            "supercategory": category
        })
    coco_output["images"].append({
        "id": i,
        "file_name": image_file,
        "height": img.shape[0],
        "width": img.shape[1]
    })
    for cls, x, y, w, h in detections:
        coco_output["annotations"].append({
            "id": len(coco_output["annotations"]),
            "image_id": i,
            "category_id": cls,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0
        })
    coco_json_path = os.path.join(out_dir, "coco_output.json")
    with open(coco_json_path, "w") as coco_file:
        json.dump(coco_output, coco_file)


class ObjectDetectionWorker(QThread):
    """Worker thread to run object detection without blocking the UI."""
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    log_update = pyqtSignal(str)
    output_image_signal = pyqtSignal(QPixmap, str)

    def __init__(self, conf_split, model_path, images_folder, output_directory, val_split, output_format, classes, device):
        super().__init__()
        self.model_path = model_path
        self.images_folder = images_folder
        self.output_directory = output_directory
        self.val_split = val_split
        self.conf_split = conf_split
        self.output_format = output_format
        self.classes = classes
        self.device = device
        self._is_running = True

    def run(self):
        select_device(self.device)
        image_files = [f for f in os.listdir(self.images_folder) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]
        if not image_files:
            self.log_update.emit("No images in the folder. Exiting.")
            self.finished.emit()
            return

        train_dir = os.path.join(self.output_directory, "train")
        train_images_dir = os.path.join(train_dir, "detected")
        train_labels_dir = os.path.join(train_dir, "labels")
        train_original_dir = os.path.join(train_dir, "images")
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(train_original_dir, exist_ok=True)

        if self.val_split > 0:
            val_dir = os.path.join(self.output_directory, "val")
            val_images_dir = os.path.join(val_dir, "detected")
            val_labels_dir = os.path.join(val_dir, "labels")
            val_original_dir = os.path.join(val_dir, "images")
            os.makedirs(val_images_dir, exist_ok=True)
            os.makedirs(val_labels_dir, exist_ok=True)
            os.makedirs(val_original_dir, exist_ok=True)

        model = YOLO(self.model_path)
        classes_string = ', '.join(self.classes)
        model.set_classes(classes_string.split(', '))

        for i, image_file in enumerate(image_files):
            if not self._is_running:
                break

            img_path = os.path.join(self.images_folder, image_file)
            img = cv2.imread(img_path)
            detections, annotated_frame = detection(model, img, self.conf_split)

            self.log_update.emit(f"Results for {image_file}:")

            if self.val_split > 0 and i % (1 / self.val_split) == 0:
                output_dir = val_dir
                images_dir = val_images_dir
                labels_dir = val_labels_dir
                original_dir = val_original_dir
            else:
                output_dir = train_dir
                images_dir = train_images_dir
                labels_dir = train_labels_dir
                original_dir = train_original_dir

            output_img = draw_boxes_on_image(img, detections, img_path, annotated_frame, os.path.join(images_dir, image_file))
            if self.output_format == "yolo":
                save_yolo_format(detections, labels_dir, image_file, img)
            elif self.output_format == "coco":
                save_coco_format(detections, output_dir, image_file, img, i, model)

            original_img_path = os.path.join(original_dir, image_file)
            shutil.copy(img_path, original_img_path)

            self.progress.emit(int((i + 1) / len(image_files) * 100))
            self.log_update.emit(f"Processed image: {image_file}")
            self.emit_output_image(output_img, image_file)

        self.log_update.emit("Object detection completed.")
        self.finished.emit()

    def emit_output_image(self, image, image_file):
        """Emit the processed image to the GUI."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.output_image_signal.emit(pixmap, image_file)

    def stop(self):
        self._is_running = False
