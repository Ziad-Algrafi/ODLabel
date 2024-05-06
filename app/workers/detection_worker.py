import threading
import cv2 as cv
import os
from queue import Queue
from typing import Union
from ultralytics import YOLOWorld
from ultralytics.utils.torch_utils import select_device
from .utils.detection_utils import infer, slice_and_infer
from .utils.draw_utils import draw_boxes_on_image
from .utils.write_utils import (
    save_yolo_format,
    save_coco_format,
    save_csv_format,
    save_xml_format,
)


class ObjectDetectionWorker(threading.Thread):
    def __init__(
        self,
        conf_threshold,
        model_path,
        images_folder,
        output_directory,
        val_split,
        output_format,
        classes,
        device,
        use_sahi,
        slice_size,
        iou_threshold,
    ):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
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

        self.log_queue: Union[Queue, None] = None
        self.output_image_queue: Union[Queue, None] = None
        self.progress_queue: Union[Queue, None] = None
        self.show_error_queue: Union[Queue, None] = None

    def get_image_files(self):
        return [
            f
            for f in os.listdir(self.images_folder)
            if f.endswith((".jpg", ".jpeg", ".png", "jfif"))
        ]

    def run(self):
        select_device(self.device, verbose=False)
        image_files = self.get_image_files()
        if not image_files:
            self.log_update("No images in the folder. Exiting.")
            self.show_error("No images found in the selected folder.")
            return

        num_total_images = len(image_files)
        num_val_images = int(num_total_images * self.val_split)
        num_train_images = num_total_images - num_val_images

        self.actual_train_count = 0
        self.actual_val_count = 0

        self.create_output_directories()

        model = self.load_model()

        for i, image_file in enumerate(image_files):
            if self._terminate_event.is_set():
                break

            img = self.load_image(image_file)
            if img is None:
                continue

            detections, annotated_frame = self.perform_inference(model, img, image_file)

            has_detections = len(detections) > 0
            (
                output_dir,
                detected_image_dir,
                labels_dir,
                original_dir,
                detections_list,
            ) = self.get_output_paths(i, num_val_images, has_detections)
            detections_list.extend([(det, image_file) for det in detections])

            self.save_results(
                detections,
                annotated_frame,
                img,
                image_file,
                detected_image_dir,
                labels_dir,
                original_dir,
                i,
            )
            self.progress(int((i + 1) / num_total_images * 100))
        self.finalize_detections(num_train_images, num_val_images, num_total_images)

    def get_output_paths(self, i, num_val_images, has_detections):
        if i < num_val_images:
            output_dir = os.path.join(self.output_directory, "val")
            detected_image_dir = os.path.join(output_dir, "detected")
            labels_dir = os.path.join(output_dir, "labels")
            original_dir = os.path.join(output_dir, "images")
            detections_list = self.val_detections
            if has_detections:
                self.actual_val_count += 1
        else:
            output_dir = os.path.join(self.output_directory, "train")
            detected_image_dir = os.path.join(output_dir, "detected")
            labels_dir = os.path.join(output_dir, "labels")
            original_dir = os.path.join(output_dir, "images")
            detections_list = self.train_detections
            if has_detections:
                self.actual_train_count += 1

        return output_dir, detected_image_dir, labels_dir, original_dir, detections_list

    def create_output_directories(self):
        if self.val_split < 1:
            train_dir = os.path.join(self.output_directory, "train")
            detected_image_dir = os.path.join(train_dir, "detected")
            train_labels_dir = os.path.join(train_dir, "labels")
            train_original_dir = os.path.join(train_dir, "images")
            os.makedirs(detected_image_dir, exist_ok=True)
            os.makedirs(train_labels_dir, exist_ok=True)
            os.makedirs(train_original_dir, exist_ok=True)

        if self.val_split > 0:
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
            return slice_and_infer(
                model,
                img,
                self.conf_threshold,
                self.log_update,
                image_file,
                self.slice_size,
                self.iou_threshold,
                self.terminate,
            )
        else:
            return infer(
                model,
                img,
                self.conf_threshold,
                self.log_update,
                image_file,
                self.iou_threshold,
            )

    def save_results(
        self,
        detections,
        annotated_frame,
        img,
        image_file,
        detected_image_dir,
        labels_dir,
        original_dir,
        i,
    ):
        if img is None:
            return

        if detections:
            output_img = draw_boxes_on_image(
                annotated_frame,
                detections,
                img,
                os.path.join(detected_image_dir, image_file),
            )

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
            csv_headers = [
                "label_name",
                "bbox_x",
                "bbox_y",
                "bbox_width",
                "bbox_height",
                "image_name",
                "image_width",
                "image_height",
            ]
            csv_file_path = os.path.join(labels_dir, "annotations.csv")
            save_csv_format(detections, image_file, csv_file_path, csv_headers)
        elif self.output_format == "XML":
            save_xml_format(detections, labels_dir, image_file)

    def finalize_detections(self, num_train_images, num_val_images, num_total_images):
        expected_train_count = 0
        expected_val_count = 0 
        actual_total_images = self.actual_train_count + self.actual_val_count
        if actual_total_images == 0:
            self.log_update("No detections found in any image.")
        else:
            expected_train_count = int(actual_total_images * (1 - self.val_split))
            expected_val_count = actual_total_images - expected_train_count

            if (
                self.actual_train_count != expected_train_count
                or self.actual_val_count != expected_val_count
            ):
                self.adjust_split(expected_train_count, expected_val_count)

        self.train_detections = (
            self.train_detections if self.actual_train_count > 0 else []
        )
        self.val_detections = self.val_detections if self.actual_val_count > 0 else []

        if (
            self.actual_train_count == expected_train_count
            and self.actual_val_count == expected_val_count
        ):
            self.log_update(
                f"Split train/validation successful. Train: {self.actual_train_count}, Val: {self.actual_val_count}"
            )
        else:
            self.log_update(
                f"Split train/validation failed. Expected: train={expected_train_count}, val={expected_val_count}. Actual: train={self.actual_train_count}, val={self.actual_val_count}"
            )

        self.log_update(
            f"Results Finished. Processed {num_total_images} images: End Results is {self.actual_train_count} training images and {self.actual_val_count} validation images\n"
        )

    def adjust_split(self, expected_train_count, expected_val_count):
        train_images_dir = os.path.join(self.output_directory, "train", "images")
        val_images_dir = os.path.join(self.output_directory, "val", "images")
        train_detected_dir = os.path.join(self.output_directory, "train", "detected")
        train_labels_dir = os.path.join(self.output_directory, "train", "labels")
        val_detected_dir = os.path.join(self.output_directory, "val", "detected")
        val_labels_dir = os.path.join(self.output_directory, "val", "labels")

        while (
            self.actual_train_count != expected_train_count
            or self.actual_val_count != expected_val_count
        ):
            if self.actual_train_count < expected_train_count:
                val_images = os.listdir(val_images_dir)
                if not val_images:
                    break
                image = val_images.pop(0)
                src_image_path = os.path.join(val_images_dir, image)
                dst_image_path = os.path.join(train_images_dir, image)
                src_detected_path = os.path.join(val_detected_dir, image)
                dst_detected_path = os.path.join(train_detected_dir, image)
                src_label_path = os.path.join(
                    val_labels_dir, os.path.splitext(image)[0] + ".txt"
                )
                dst_label_path = os.path.join(
                    train_labels_dir, os.path.splitext(image)[0] + ".txt"
                )
                os.rename(src_image_path, dst_image_path)
                os.rename(src_detected_path, dst_detected_path)
                os.rename(src_label_path, dst_label_path)

                detections_to_move = [
                    det for det, img in self.val_detections if img == image
                ]
                self.train_detections.extend(
                    [(det, image) for det in detections_to_move]
                )
                self.val_detections = [
                    (det, img) for det, img in self.val_detections if img != image
                ]

                self.actual_train_count += 1
                self.actual_val_count -= 1
            elif self.actual_val_count < expected_val_count:
                train_images = os.listdir(train_images_dir)
                if not train_images:
                    break
                image = train_images.pop(0)
                src_image_path = os.path.join(train_images_dir, image)
                dst_image_path = os.path.join(val_images_dir, image)
                src_detected_path = os.path.join(train_detected_dir, image)
                dst_detected_path = os.path.join(val_detected_dir, image)
                src_label_path = os.path.join(
                    train_labels_dir, os.path.splitext(image)[0] + ".txt"
                )
                dst_label_path = os.path.join(
                    val_labels_dir, os.path.splitext(image)[0] + ".txt"
                )
                os.rename(src_image_path, dst_image_path)
                os.rename(src_detected_path, dst_detected_path)
                os.rename(src_label_path, dst_label_path)

                detections_to_move = [
                    det for det, img in self.train_detections if img == image
                ]
                self.val_detections.extend([(det, image) for det in detections_to_move])
                self.train_detections = [
                    (det, img) for det, img in self.train_detections if img != image
                ]

                self.actual_val_count += 1
                self.actual_train_count -= 1

    def get_detections(self):
        train_detections = [det for det, _ in self.train_detections]
        val_detections = [det for det, _ in self.val_detections]
        return (
            train_detections,
            val_detections,
            self.actual_train_count,
            self.actual_val_count,
        )

    def terminate(self):
        self._terminate_event.set()

    def log_update(self, log_message):
        if self.log_queue is not None:
            self.log_queue.put(log_message)

    def show_error(self, message):
        if self.show_error_queue is not None:
            self.show_error_queue.put(message)

    def output_image(self, output_img_path, image_file):
        if self.output_image_queue is not None:
            self.output_image_queue.put((output_img_path, image_file))

    def progress(self, value):
        if self.progress_queue is not None:
            self.progress_queue.put(value)
