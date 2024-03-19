from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QComboBox, QTextEdit, QSlider, QGridLayout, QScrollArea,
    QProgressBar, QMessageBox, QFileDialog, QTabWidget
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
import os

from object_detection_worker import ObjectDetectionWorker

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

        # Run/Stop/Cancel Buttons
        self.run_stop_cancel_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Code")
        self.run_button.clicked.connect(self.run_code)
        self.run_button.setEnabled(False)
        self.stop_button = QPushButton("Stop Code")
        self.stop_button.clicked.connect(self.stop_code)
        self.stop_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_code)
        self.cancel_button.setEnabled(False)
        self.run_stop_cancel_layout.addWidget(self.run_button)
        self.run_stop_cancel_layout.addWidget(self.stop_button)
        self.run_stop_cancel_layout.addWidget(self.cancel_button)
        left_panel_layout.addLayout(self.run_stop_cancel_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_panel_layout.addWidget(self.progress_bar)

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

        self.tab_widget = QTabWidget()
        photo_gallery_layout.addWidget(self.tab_widget)

        self.input_tab = QWidget()
        self.output_tab = QWidget()
        
        self.tab_widget.addTab(self.input_tab, "Input")
        self.tab_widget.addTab(self.output_tab, "Output")
    #     self.create_photo_gallery_tabs()

    # def create_photo_gallery_tabs(self):
        # Input Tab
        input_tab_layout = QVBoxLayout()
        self.input_tab.setLayout(input_tab_layout)

        input_scroll_area = QScrollArea()
        input_scroll_area.setWidgetResizable(True)
        input_widget = QWidget()
        input_grid_layout = QGridLayout(input_widget)
        input_scroll_area.setWidget(input_widget)
        input_tab_layout.addWidget(input_scroll_area)
        self.input_photo_grid_layout = input_grid_layout

        # Output Tab
        output_tab_layout = QVBoxLayout()
        self.output_tab.setLayout(output_tab_layout)

        output_tab_widget = QTabWidget()
        output_tab_layout.addWidget(output_tab_widget)

        train_tab = QWidget()
        val_tab = QWidget()
        output_tab_widget.addTab(train_tab, "Train")
        output_tab_widget.addTab(val_tab, "Val")

        train_scroll_area = QScrollArea()
        train_scroll_area.setWidgetResizable(True)
        train_widget = QWidget()
        train_grid_layout = QGridLayout(train_widget)
        train_scroll_area.setWidget(train_widget)
        train_tab_layout = QVBoxLayout(train_tab)
        train_tab_layout.addWidget(train_scroll_area)
        self.train_photo_grid_layout = train_grid_layout

        val_scroll_area = QScrollArea()
        val_scroll_area.setWidgetResizable(True)
        val_widget = QWidget()
        val_grid_layout = QGridLayout(val_widget)
        val_scroll_area.setWidget(val_widget)
        val_tab_layout = QVBoxLayout(val_tab)
        val_tab_layout.addWidget(val_scroll_area)
        self.val_photo_grid_layout = val_grid_layout

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
            self.load_photos(images_folder, "input")
            self.check_run_button_state()

    def load_photos(self, folder, tab_type):
        grid_layout = None
        if tab_type == "input":
            grid_layout = self.input_photo_grid_layout
        elif tab_type == "train":
            grid_layout = self.train_photo_grid_layout
        elif tab_type == "val":
            grid_layout = self.val_photo_grid_layout

        if grid_layout is None:
            return

        self.clear_photo_grid(grid_layout)

        image_files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]
        row = 0
        col = 0
        for image_file in image_files:
            img_path = os.path.join(folder, image_file)
            pixmap = QPixmap(img_path).scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
            label = QLabel()
            label.setPixmap(pixmap)
            label.setStyleSheet("border: 1px solid #ccc; padding: 5px;")

            grid_layout.addWidget(label, row, col)
            col += 1
            if col >= 4:
                row += 1
                col = 0

    def clear_photo_grid(self, grid_layout):
        for i in reversed(range(grid_layout.count())):
            item = grid_layout.itemAt(i)
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
        self.run_button.setEnabled(bool(model_path and images_folder and output_directory))

    def run_code(self):
        model_path = self.model_path_line_edit.text()
        images_folder = self.images_folder_line_edit.text()
        output_directory = self.output_directory_line_edit.text()
        val_split = self.val_slider.value() / 100
        output_format = self.output_format_combo_box.currentText().lower()
        device = self.device_combo_box.currentText()

        # Create and start the worker thread
        self.worker_thread = ObjectDetectionWorker(model_path, images_folder, output_directory, val_split, output_format, device)
        self.worker_thread.finished.connect(self.on_worker_finished)
        self.worker_thread.progress.connect(self.progress_bar.setValue)
        self.worker_thread.log_update.connect(self.update_log)
        self.worker_thread.start()

        # Update UI
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)

    def stop_code(self):
        # Implement stopping mechanism here
        # For now, we just disable the stop button
        self.stop_button.setEnabled(False)

    def cancel_code(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.terminate()  # Terminate the worker thread
            self.worker_thread.wait()  # Wait for the thread to finish
            self.worker_thread = None

        # Reset UI
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text_edit.append("Object detection process canceled.")

    def on_worker_finished(self):
        # Reset UI
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.worker_thread = None

        # Load output images
        output_directory = self.output_directory_line_edit.text()
        train_dir = os.path.join(output_directory, "train")
        val_dir = os.path.join(output_directory, "val")
        self.load_photos(train_dir, "train")
        self.load_photos(val_dir, "val")

    def update_log(self, log_message):
        self.log_text_edit.append(log_message)

    def show_error_message(self, message):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Icon.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.exec()
