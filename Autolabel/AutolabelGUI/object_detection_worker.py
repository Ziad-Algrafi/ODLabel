from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
import cv2
import shutil
import os

class_names_dict = {i: x for i, x in enumerate(["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"])}

def draw_boxes_on_image(image_path, labels_path, output_path):
    img = cv2.imread(image_path)
    with open(labels_path, 'r') as txt_file:
        lines = txt_file.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_center, y_center, width, height = map(float, parts)
        img_height, img_width, _ = img.shape
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)
        thickness = 2
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness)
        class_name = class_names_dict.get(int(class_id), "Unknown")
        font_scale = 0.7
        font_thickness = 1
        text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        rect_width = text_size[0] + 10
        rect_height = text_size[1] + 5
        img = cv2.rectangle(img, (x_min, y_min - rect_height), (x_min + rect_width, y_min), (0, 0, 255), thickness=cv2.FILLED)
        text_position = (x_min + 5, y_min - 5)
        img = cv2.putText(img, class_name , text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    if output_path is None:
        output_path = os.path.splitext(image_path)[0] + '_with_boxes.jpg'
    cv2.imwrite(output_path, img)
    print(f"Saved image with bounding boxes to {output_path}")
class ObjectDetectionWorker(QThread):
    """Worker thread to run object detection without blocking the UI."""
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    log_update = pyqtSignal(str)
    def __init__(self,conf_split, model_path, images_folder, output_directory, val_split, output_format, classes, device):
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
        model = YOLO(self.model_path)
        image_files = [f for f in os.listdir(self.images_folder) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]
        if not image_files:
            self.log_update.emit("No images in the folder. Exiting.")
            self.finished.emit()
            return
        train_dir = os.path.join(self.output_directory, "train")
        val_dir = os.path.join(self.output_directory, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        for i, image_file in enumerate(image_files):
            if not self._is_running:
                break
            img_path = os.path.join(self.images_folder, image_file)
            img = cv2.imread(img_path)
            results = model(img, conf=self.conf_split, imgsz=640)
            self.log_update.emit(f"Results for {image_file}:")
            is_val = "val" in image_file
            out_dir = val_dir if is_val else train_dir
            out_img_path = os.path.join(out_dir, image_file)
            shutil.copy(img_path, out_img_path)
            if self.output_format == "yolo":
                result_txt_path = os.path.join(out_dir, f"{os.path.splitext(image_file)[0]}.txt")
                with open(result_txt_path, 'a') as txt_file:
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            xy = box.xywhn.cpu().numpy()
                            c = box.cls.cpu().numpy()
                            class_id = int(c)
                            self.log_update.emit(f" ({class_id}): {xy}")
                            self.log_update.emit(str(class_id))
                            for box in xy:
                                txt_file.write(f"{class_id} {box[1]} {box[2]} {box[3]}\n")
                                
                # Draw bounding boxes on the image
                draw_boxes_on_image(out_img_path, result_txt_path, None)
            elif self.output_format == "coco":
                import json
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
                    "width": img.shape[                    1]
                    })
                for result in results:
                    if not self._is_running:
                        break
                    boxes = result.boxes
                    for box in boxes:
                        xy = box.xywhn.cpu().numpy()
                        c = box.cls.cpu().numpy()
                        class_id = int(c)
                        x, y, w, h = [int(val) for val in xy[:4]]
                        coco_output["annotations"].append({
                            "id": len(coco_output["annotations"]),
                            "image_id": i,
                            "category_id": class_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                coco_json_path = os.path.join(out_dir, "coco_output.json")
                with open(coco_json_path, "w") as coco_file:
                    json.dump(coco_output, coco_file)
                # Draw bounding boxes on the image
                draw_boxes_on_image(out_img_path, coco_json_path, None)
            self.progress.emit(int((i + 1) / len(image_files) * 100))
        self.log_update.emit(f"Results and images saved in {self.output_directory}")
        self.finished.emit()
    def stop(self):
        self._is_running = False
