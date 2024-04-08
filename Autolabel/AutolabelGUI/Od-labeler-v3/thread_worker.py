import threading
import queue
import cv2 as cv
import os
import shutil
from ultralytics import YOLOWorld
from ultralytics.utils.torch_utils import select_device

from detection_utils import draw_boxes_on_image
from write_utils import save_yolo_format, save_coco_format, save_csv_format, save_xml_format

class ObjectDetectionWorker(threading.Thread):
    def __init__(self, conf_threshold, model_path, images_folder, output_directory, val_split, output_format, classes, device, use_sahi, slice_size):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.images_folder = images_folder
        self.output_directory = output_directory
        self.val_split = val_split
        self.output_format = output_format
        self.classes = classes
        self.device = device
        self.use_sahi = use_sahi
        self.slice_size = slice_size
        self._terminate_event = threading.Event()

        self.log_queue = None
        self.output_image_queue = None
        self.progress_queue = None

    def run(self):
        select_device(self.device)
        image_files = [f for f in os.listdir(self.images_folder) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]
        if not image_files:
            self.log_update("No images in the folder. Exiting.")
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

        model = YOLOWorld(self.model_path)
        model.set_classes(self.classes)

        for i, image_file in enumerate(image_files):
            if self._terminate_event.is_set():
                break

            img_path = os.path.join(self.images_folder, image_file)
            img = cv.imread(img_path)

            if self.use_sahi:
                detections, annotated_frame = self.slice_and_infer(model, img, image_file)
            else:
                detections, annotated_frame = self.infer(model, img, image_file)

            if i < num_val_images:
                output_dir = val_dir
                images_dir = val_images_dir
                labels_dir = val_labels_dir
                original_dir = val_original_dir
            else:
                output_dir = train_dir
                images_dir = train_images_dir
                labels_dir = train_labels_dir
                original_dir = train_original_dir

            if detections:
                output_img = draw_boxes_on_image(annotated_frame, detections, img_path, os.path.join(images_dir, image_file))

                output_img_path = os.path.join(images_dir, image_file)
                cv.imwrite(output_img_path, output_img)

                if self.output_format == "YOLO":
                    save_yolo_format(detections, labels_dir, image_file, img)
                elif self.output_format == "COCO":
                    coco_output_file = os.path.join(labels_dir, "coco_output.json")
                    save_coco_format(detections, image_file, img, i, coco_output_file)
                elif self.output_format == "CSV":
                    csv_headers = ['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'image_name', 'image_width', 'image_height']
                    csv_file_path = os.path.join(labels_dir, "annotations.csv")
                    save_csv_format(detections, image_file, img, csv_file_path, csv_headers)
                elif self.output_format == "XML":
                    save_xml_format(self, detections, labels_dir, image_file, images_dir, img)

                original_img_path = os.path.join(original_dir, image_file)
                shutil.copy(img_path, original_img_path)

                self.output_image(output_img_path, image_file)

            self.progress(int((i + 1) / len(image_files) * 100))
            self.log_update(f"Processed image: {image_file}")

        self.log_update("Results Finished.")

    def infer(self, model, img, image_file):
        annotated_frame = img.copy()
        img_height, img_width = img.shape[:2]
        results = model(img, conf=self.conf_threshold, imgsz=640)

        detections = []
        names = model.names
        if results[0].boxes is not None:
            self.log_update(f"Results for {image_file}:")
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
            self.log_update(f"No Results for {image_file}:")

        return detections, annotated_frame

    def slice_and_infer(self, model, img, image_file):
        annotated_frame = img.copy()
        height, width = img.shape[:2]
        slices = []
        num_slices = 0

        for y in range(0, height, self.slice_size // 2):
            for x in range(0, width, self.slice_size // 2):
                slice_y = min(y + self.slice_size, height)
                slice_x = min(x + self.slice_size, width)
                slice = img[y:slice_y, x:slice_x]
                slices.append((slice, x, y, slice_x - x, slice_y - y))
                num_slices += 1

        self.log_update(f"Number of slices: {num_slices}")
        detections = []
        names = model.names
        for slice, x_offset, y_offset, slice_width, slice_height in slices:
            results = model(slice, conf=self.conf_threshold, imgsz=self.slice_size)[0]
            if results.boxes is not None:
                self.log_update(f"Results for {image_file}:")
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
                self.log_update(f"No Results for {image_file}:")

        return detections, annotated_frame

    def terminate(self):
        self._terminate_event.set()

    def log_update(self, message):
        self.log_queue.put(message)

    def output_image(self, output_img_path, image_file):
        self.output_image_queue.put((output_img_path, image_file))

    def progress(self, value):
        self.progress_queue.put(value)