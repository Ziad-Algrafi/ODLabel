import threading
import cv2 as cv
import os
import shutil
from ultralytics import YOLOWorld
from ultralytics.utils.torch_utils import select_device

from detection_utils import infer, slice_and_infer, draw_boxes_on_image
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
        self.train_detections = []
        self.val_detections = []

        self.log_queue = None
        self.output_image_queue = None
        self.progress_queue = None

    def get_image_files(self):
        return [f for f in os.listdir(self.images_folder) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]

    def run(self):
        select_device(self.device, verbose=False)
        image_files = self.get_image_files()
        if not image_files:
            self.log_update("No images in the folder. Exiting.")
            return

        num_train_images = int(len(image_files) * (1 - self.val_split))
        num_val_images = len(image_files) - num_train_images
        num_total_images = len(image_files)
        self.create_output_directories(num_train_images, num_val_images, num_total_images)

        model = self.load_model()

        for i, image_file in enumerate(image_files):
            if self._terminate_event.is_set():
                break

            img = self.load_image(image_file)
            if img is None:
                continue

            detections, annotated_frame = self.perform_inference(model, img, image_file)

            output_dir, detected_image_dir, labels_dir, original_dir, detections_list = self.get_output_paths(i, num_train_images)
            detections_list.extend(detections)

            self.save_results(detections, annotated_frame, img, image_file, detected_image_dir, labels_dir, original_dir, i)

            self.progress(int((i + 1) / num_total_images * 100))

        self.finalize_detections(num_train_images, num_val_images, num_total_images)
        self.log_update("Results Finished.")

    def get_output_paths(self, i, num_train_images):
        if i < num_train_images:
            output_dir = os.path.join(self.output_directory, "train")
            detected_image_dir = os.path.join(output_dir, "detected")
            labels_dir = os.path.join(output_dir, "labels")
            original_dir = os.path.join(output_dir, "images")
            detections_list = self.train_detections
        else:
            output_dir = os.path.join(self.output_directory, "val")
            detected_image_dir = os.path.join(output_dir, "detected")
            labels_dir = os.path.join(output_dir, "labels")
            original_dir = os.path.join(output_dir, "images")
            detections_list = self.val_detections

        return output_dir, detected_image_dir, labels_dir, original_dir, detections_list

    def create_output_directories(self, num_train_images, num_val_images, num_total_images):
        if num_train_images > 0:
            self.create_train_directories(num_train_images, num_total_images)
        if num_val_images > 0:
            self.create_val_directories(num_val_images)

    def create_train_directories(self, num_train_images, num_total_images):
        train_dir = os.path.join(self.output_directory, "train")
        detected_image_dir = os.path.join(train_dir, "detected")
        train_labels_dir = os.path.join(train_dir, "labels")
        train_original_dir = os.path.join(train_dir, "images")
        os.makedirs(detected_image_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(train_original_dir, exist_ok=True)

    def create_val_directories(self, num_val_images):
        val_dir = os.path.join(self.output_directory, "val")
        detected_image_dir = os.path.join(val_dir, "detected")
        val_labels_dir = os.path.join(val_dir, "labels")
        val_original_dir = os.path.join(val_dir, "images")
        os.makedirs(detected_image_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        os.makedirs(val_original_dir, exist_ok=True)

    def load_model(self):
        model = YOLOWorld(self.model_path)
        model.set_classes(self.classes)
        return model

    def load_image(self, image_file):
        img_path = os.path.join(self.images_folder, image_file)
        return cv.imread(img_path)

    def perform_inference(self, model, img, image_file):
        if self.use_sahi:
            return slice_and_infer(model, img, self.conf_threshold, self.log_update, image_file, self.slice_size, self.terminate)
        else:
            return infer(model, img, self.conf_threshold, self.log_update, image_file)

    def save_results(self, detections, annotated_frame, img, image_file, detected_image_dir, labels_dir, original_dir, i):
        if img is None:
            return
        
        if detections:
            output_img = draw_boxes_on_image(annotated_frame, detections, img, os.path.join(detected_image_dir, image_file))

            output_img_path = os.path.join(detected_image_dir, image_file)
            cv.imwrite(output_img_path, output_img)

            self.save_format_results(detections, labels_dir, image_file, i)

            original_img_path = os.path.join(original_dir, image_file)
            cv.imwrite(original_img_path, img)

            if self.output_image_queue is not None:
                self.output_image(output_img_path, image_file)

    def save_format_results(self, detections, labels_dir, image_file, i):
        if self.output_format == "YOLO":
            save_yolo_format(detections, labels_dir, image_file)
        elif self.output_format == "COCO":
            coco_output_file = os.path.join(labels_dir, "coco_output.json")
            save_coco_format(detections, image_file, i, coco_output_file)
        elif self.output_format == "CSV":
            csv_headers = ['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'image_name', 'image_width', 'image_height']
            csv_file_path = os.path.join(labels_dir, "annotations.csv")
            save_csv_format(detections, image_file, csv_file_path, csv_headers)
        elif self.output_format == "XML":
            save_xml_format(detections, labels_dir, image_file)

    def finalize_detections(self, num_train_images, num_val_images, num_total_images):
        self.train_detections = self.train_detections if num_train_images > 0 else []
        self.val_detections = self.val_detections if num_val_images > 0 else []

    def get_detections(self):
        train_images_dir = os.path.join(self.output_directory, "train", "images")
        if os.path.exists(train_images_dir):
            num_train_images = len(os.listdir(train_images_dir))
        else:
            num_train_images = 0

        val_images_dir = os.path.join(self.output_directory, "val", "images")
        if os.path.exists(val_images_dir):
            num_val_images = len(os.listdir(val_images_dir))
        else:
            num_val_images = 0

        return self.train_detections, self.val_detections, num_train_images, num_val_images

    def terminate(self):
        self._terminate_event.set()

    def log_update(self, message):
        if self.log_queue is not None: 
            self.log_queue.put(message) 

    def output_image(self, output_img_path, image_file):
        if self.output_image_queue is not None:
            self.output_image_queue.put((output_img_path, image_file))

    def progress(self, value):
        if self.progress_queue is not None: 
            self.progress_queue.put(value)