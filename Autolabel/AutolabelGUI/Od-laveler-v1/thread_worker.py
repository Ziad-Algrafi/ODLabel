from PyQt6.QtCore import QThread, pyqtSignal
import cv2 as cv
import os
import shutil
from ultralytics import YOLOWorld
from ultralytics.utils.torch_utils import select_device

from detection_utils import infer, slice_and_infer, draw_boxes_on_image
from write_utils import save_yolo_format, save_coco_format, save_csv_format, save_xml_format

class ObjectDetectionWorker(QThread):
    """Worker thread to run object detection without blocking the UI."""
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    log_update = pyqtSignal(str)
    output_image_signal = pyqtSignal(str, str)

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
        select_device(self.device)
        image_files = [f for f in os.listdir(self.images_folder) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]
        if not image_files:
            self.log_update.emit("No images in the folder. Exiting.")
            self.finished.emit()
            return

        num_val_images = int(len(image_files) * self.val_split)

        # if os.path.exists(self.output_directory):
        #     shutil.rmtree(self.output_directory)

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

                self.emit_output_image(output_img_path, image_file)

            self.progress.emit(int((i + 1) / len(image_files) * 100))
            self.log_update.emit(f"Processed image: {image_file}")

        # Release memory
        del detections
        del annotated_frame
        del img
        del model

        self.log_update.emit("Results Finished.")
        self.finished.emit()

    def emit_output_image(self, image_path, image_file):
        self.output_image_signal.emit(image_path, image_file)