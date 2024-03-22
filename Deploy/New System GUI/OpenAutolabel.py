import os
import sys
import cv2
import shutil
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QComboBox,
    QTextEdit,
    QSlider,
    QGridLayout,
    QScrollArea
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Open Dictionary Auto Label")
        self.setMinimumSize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left Panel
        left_panel_layout = QVBoxLayout()
        main_layout.addLayout(left_panel_layout)

        # Model Path
        model_path_layout = QHBoxLayout()
        model_path_label = QLabel("Model Path:")
        self.model_path_line_edit = QLineEdit()
        model_path_button = QPushButton("Select YOLO Model")
        model_path_button.clicked.connect(self.select_model)
        model_path_layout.addWidget(model_path_label)
        model_path_layout.addWidget(self.model_path_line_edit)
        model_path_layout.addWidget(model_path_button)
        left_panel_layout.addLayout(model_path_layout)

        # Images Folder
        images_folder_layout = QHBoxLayout()
        images_folder_label = QLabel("Images Folder:")
        self.images_folder_line_edit = QLineEdit()
        images_folder_button = QPushButton("Select Images Folder")
        images_folder_button.clicked.connect(self.select_images_folder)
        images_folder_layout.addWidget(images_folder_label)
        images_folder_layout.addWidget(self.images_folder_line_edit)
        images_folder_layout.addWidget(images_folder_button)
        left_panel_layout.addLayout(images_folder_layout)

        # Output Directory
        output_directory_layout = QHBoxLayout()
        output_directory_label = QLabel("Output Directory:")
        self.output_directory_line_edit = QLineEdit()
        output_directory_button = QPushButton("Select Output Directory")
        output_directory_button.clicked.connect(self.select_output_directory)
        output_directory_layout.addWidget(output_directory_label)
        output_directory_layout.addWidget(self.output_directory_line_edit)
        output_directory_layout.addWidget(output_directory_button)
        left_panel_layout.addLayout(output_directory_layout)

        # Device
        device_layout = QHBoxLayout()
        device_label = QLabel("Select Device:")
        self.device_combo_box = QComboBox()
        self.device_combo_box.addItems(["cpu", "0"])
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo_box)
        left_panel_layout.addLayout(device_layout)

        # Output Format
        output_format_layout = QHBoxLayout()
        output_format_label = QLabel("Output Format:")
        self.output_format_combo_box = QComboBox()
        self.output_format_combo_box.addItems(["YOLO", "COCO"])
        output_format_layout.addWidget(output_format_label)
        output_format_layout.addWidget(self.output_format_combo_box)
        left_panel_layout.addLayout(output_format_layout)

        # Train/Val Split
        split_layout = QHBoxLayout()
        split_label = QLabel("Train/Val Split:")
        self.val_slider = QSlider(Qt.Orientation.Horizontal)
        self.val_slider.setMinimum(0)
        self.val_slider.setMaximum(100)
        self.val_slider.setValue(20)
        self.val_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.val_slider.setTickInterval(10)
        self.val_label = QLabel(f"0.{self.val_slider.value()}0")
        self.val_slider.valueChanged.connect(self.update_val_label)
        split_layout.addWidget(split_label)
        split_layout.addWidget(self.val_slider)
        split_layout.addWidget(self.val_label)
        left_panel_layout.addLayout(split_layout)

        # Run/Stop Button
        self.run_stop_button = QPushButton("Run Code")
        self.run_stop_button.clicked.connect(self.run_stop_code)
        self.run_stop_button.setEnabled(False)
        left_panel_layout.addWidget(self.run_stop_button)

        # Log Text
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        left_panel_layout.addWidget(self.log_text_edit)

        # Photo Gallery
        photo_gallery_layout = QVBoxLayout()
        main_layout.addLayout(photo_gallery_layout)

        photo_gallery_label = QLabel("Photo Gallery")
        photo_gallery_label.setStyleSheet("font-weight: bold; font-size: 16px; padding: 10px;")
        photo_gallery_layout.addWidget(photo_gallery_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        photo_gallery_layout.addWidget(scroll_area)

        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)

        self.photo_grid_layout = QGridLayout()
        scroll_widget.setLayout(self.photo_grid_layout)

        self.running = False

    def select_model(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("YOLO Model (*.pt)")
        if file_dialog.exec():
            model_path = file_dialog.selectedFiles()[0]
            self.model_path_line_edit.setText(model_path)
            self.check_run_button_state()

    def select_images_folder(self):
        folder_dialog = QFileDialog()
        folder_dialog.setFileMode(QFileDialog.FileMode.Directory)
        if folder_dialog.exec():
            images_folder = folder_dialog.selectedFiles()[0]
            self.images_folder_line_edit.setText(images_folder)
            self.load_photos(images_folder)
            self.check_run_button_state()

    def load_photos(self, folder):
        self.clear_photo_grid()
        image_files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]
        row = 0
        col = 0
        for image_file in image_files:
            img_path = os.path.join(folder, image_file)
            pixmap = QPixmap(img_path).scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
            label = QLabel()
            label.setPixmap(pixmap)
            label.setStyleSheet("border: 1px solid #ccc; padding: 5px;")
            self.photo_grid_layout.addWidget(label, row, col)
            col += 1
            if col >= 4:
                row += 1
                col = 0

    def clear_photo_grid(self):
        for i in reversed(range(self.photo_grid_layout.count())):
            item = self.photo_grid_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()

    def select_output_directory(self):
        folder_dialog = QFileDialog()
        folder_dialog.setFileMode(QFileDialog.FileMode.Directory)
        if folder_dialog.exec():
            output_directory = folder_dialog.selectedFiles()[0]
            self.output_directory_line_edit.setText(output_directory)
            self.check_run_button_state()

    def update_val_label(self, value):
        self.val_label.setText(f"0.{value}0")

    def check_run_button_state(self):
        model_path = self.model_path_line_edit.text()
        images_folder = self.images_folder_line_edit.text()
        output_directory = self.output_directory_line_edit.text()
        self.run_stop_button.setEnabled(bool(model_path and images_folder and output_directory))

    def run_stop_code(self):
        if not self.running:
            self.running = True
            self.run_stop_button.setText("Stop Code")
            self.run_code()
        else:
            self.running = False
            self.run_stop_button.setText("Run Code")
            # Add code to stop the code execution here

    def run_code(self):
        select_device(self.device_combo_box.currentText())
        self.log_text_edit.clear()

        model_path = self.model_path_line_edit.text()
        images_folder = self.images_folder_line_edit.text()
        output_directory = self.output_directory_line_edit.text()
        val_split = self.val_slider.value() / 100
        output_format = self.output_format_combo_box.currentText().lower()

        model = YOLO(model_path)

        image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]

        if not image_files:
            self.log_text_edit.append("No images in the folder. Exiting.")
            self.running = False
            self.run_stop_button.setText("Run Code")
            return

        train_dir = os.path.join(output_directory, "train")
        val_dir = os.path.join(output_directory, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for i, image_file in enumerate(image_files):
            if not self.running:
                break

            img_path = os.path.join(images_folder, image_file)
            img = cv2.imread(img_path)

            results = model(img, conf=0.3, imgsz=640)
            self.log_text_edit.append(f"Results for {image_file}:")

            is_val = i < len(image_files) * val_split
            out_dir = val_dir if is_val else train_dir
            out_img_path = os.path.join(out_dir, image_file)
            shutil.copy(img_path, out_img_path)

            if output_format == "yolo":
                result_txt_path = os.path.join(out_dir, f"{os.path.splitext(image_file)[0]}.txt")
                with open(result_txt_path, 'a') as txt_file:
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            xy = box.xywhn.cpu().numpy()
                            c = box.cls.cpu().numpy()
                            class_id = int(c)
                            self.log_text_edit.append(f" ({class_id}): {xy}")
                            self.log_text_edit.append(str(class_id))

                            for box in xy:
                                txt_file.write(f"{class_id} {box[0]} {box[1]} {box[2]} {box[3]}\n")

                            QApplication.processEvents()

            elif output_format == "coco":
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
                    "width": img.shape[1]
                })

                for result in results:
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


            self.log_text_edit.append(f"Results and images saved in {output_directory}")
            self.running = False
            self.run_stop_button.setText("Run Code")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
