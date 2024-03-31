from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QComboBox, QTextEdit, QSlider, QGridLayout, QScrollArea,
    QProgressBar, QMessageBox, QFileDialog, QTabWidget, QCheckBox, QSpinBox
)
from PyQt6.QtGui import QPixmap, QPalette, QColor
from PyQt6.QtCore import Qt, QLocale
import os

from thread_worker import ObjectDetectionWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OD-Labeler")
        self.setFixedSize(1280, 900)

        # Set dark color palette
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(21, 24, 26))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(47, 49, 54))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(54, 57, 63))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(47, 49, 54))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Button, QColor(54, 57, 63))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
        self.setPalette(palette)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.button_style_ON = """
        QPushButton {
            background-color: #404040;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px;
            font-size: 14px;
            font-weight: light;
        }
        QPushButton:hover {
            background-color: #808080;
        }
        QPushButton:pressed {
            background-color: #404040;
        }
        """

        self.button_style_OFF = """
        QPushButton {
            background-color: #2F3136;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px;
            font-size: 14px;
            font-weight: light;
        }
        QPushButton:pressed {
            background-color: #2F3136;
        }
        """

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left Panel
        left_panel_layout = QVBoxLayout()
        main_layout.addLayout(left_panel_layout)

        # Model Path
        model_path_layout = QHBoxLayout()
        model_path_label = QLabel("Model Path:")
        model_path_label.setFixedWidth(120)
        self.model_path_line_edit = QLineEdit()
        self.model_path_line_edit.setFixedWidth(320)
        self.model_path_button = QPushButton("Select YOLO Model")
        self.model_path_button.setFixedWidth(190)
        self.model_path_button.setEnabled(True)
        self.model_path_button.clicked.connect(self.select_model)


        # Apply stylesheet to the button
        self.model_path_button.setStyleSheet(self.button_style_ON)

        model_path_layout.addWidget(model_path_label)
        model_path_layout.addWidget(self.model_path_line_edit)
        model_path_layout.addStretch()
        model_path_layout.addWidget(self.model_path_button)
        left_panel_layout.addLayout(model_path_layout)

        # Images Folder
        images_folder_layout = QHBoxLayout()
        images_folder_label = QLabel("Images Folder:")
        images_folder_label.setFixedWidth(120)
        self.images_folder_line_edit = QLineEdit()
        self.images_folder_line_edit.setFixedWidth(320)
        self.images_folder_button= QPushButton("Select Images Folder")
        self.images_folder_button.setFixedWidth(190)
        self.images_folder_button.setEnabled(True)

        self.images_folder_button.clicked.connect(self.select_images_folder)

        # Apply stylesheet to the button
        self.images_folder_button.setStyleSheet(self.button_style_ON)

        images_folder_layout.addWidget(images_folder_label)
        images_folder_layout.addWidget(self.images_folder_line_edit)
        images_folder_layout.addStretch()
        images_folder_layout.addWidget(self.images_folder_button)
        left_panel_layout.addLayout(images_folder_layout)

        # Output Directory
        output_directory_layout = QHBoxLayout()
        output_directory_label = QLabel("Output Directory:")
        output_directory_label.setFixedWidth(120)
        self.output_directory_line_edit = QLineEdit()
        self.output_directory_line_edit.setFixedWidth(320)
        self.output_directory_button = QPushButton("Select Output Directory")
        self.output_directory_button.setFixedWidth(190)
        self.output_directory_button.setEnabled(True)
        self.output_directory_button.clicked.connect(self.select_output_directory)

        # Apply stylesheet to the button
        self.output_directory_button.setStyleSheet(self.button_style_ON)

        output_directory_layout.addWidget(output_directory_label)
        output_directory_layout.addWidget(self.output_directory_line_edit)
        output_directory_layout.addStretch()
        output_directory_layout.addWidget(self.output_directory_button)
        left_panel_layout.addLayout(output_directory_layout)

        # Classes
        classes_layout = QHBoxLayout()
        classes_label = QLabel("Classes:")
        self.classes_line_edit = QLineEdit()
        self.classes_line_edit.textChanged.connect(self.update_classes)
        classes_layout.addWidget(classes_label)
        classes_layout.addWidget(self.classes_line_edit)
        left_panel_layout.addLayout(classes_layout)

        # SAHI Checkbox and Slice Size Input
        sahi_layout = QHBoxLayout()
        self.sahi_checkbox = QCheckBox("Slicing Adaptive Inference (SAHI)")
        self.sahi_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                padding-left: 5px;
                padding-right: 5px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
        """)
        sahi_layout.addWidget(self.sahi_checkbox)

        slice_size_layout = QHBoxLayout()
        slice_size_label = QLabel("Slice Size:")
        slice_size_label.setContentsMargins(130, 0, 0, 0)
        self.slice_size_input = QSpinBox()
        self.slice_size_input.setMinimum(128)
        self.slice_size_input.setMaximum(4032)
        self.slice_size_input.setSingleStep(32)
        self.slice_size_input.setValue(512)
        self.slice_size_input.setFixedWidth(115)
        self.slice_size_input.setEnabled(False)  
        self.slice_size_input.setLocale(QLocale('en_US')) 
        slice_size_layout.addWidget(slice_size_label)
        slice_size_layout.addWidget(self.slice_size_input)
        slice_size_layout.setContentsMargins(0, 0, 0, 0)
        self.slice_size_input.setStyleSheet(self.button_style_ON)
        sahi_layout.addSpacing(0)
        sahi_layout.addLayout(slice_size_layout)
        left_panel_layout.addLayout(sahi_layout)

        self.sahi_checkbox.stateChanged.connect(self.update_slice_size_input)


        # Device
        device_layout = QHBoxLayout()
        device_label = QLabel("Select Device:")
        self.device_combo_box = QComboBox()
        self.device_combo_box.addItems(["CPU", "GPU"])
        self.device_combo_box.setFixedWidth(190)
        self.device_combo_box.setStyleSheet(self.button_style_ON)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo_box)
        left_panel_layout.addLayout(device_layout)

        # Output Format
        output_format_layout = QHBoxLayout()
        output_format_label = QLabel("Output Format:")
        self.output_format_combo_box = QComboBox()
        self.output_format_combo_box.addItems(["YOLO", "COCO", "CSV", "XML"])
        self.output_format_combo_box.setFixedWidth(190)
        self.output_format_combo_box.setStyleSheet(self.button_style_ON)
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
        self.val_slider.setLocale(QLocale('en_US'))  
        self.val_label = QLabel(f"{self.val_slider.value() / 100:.2f}")
        self.val_slider.valueChanged.connect(self.update_val_label)
        split_layout.addWidget(split_label)
        split_layout.addWidget(self.val_slider)
        split_layout.addWidget(self.val_label)
        left_panel_layout.addLayout(split_layout)

        # Minimum Confidence
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence:")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(20) 
        self.conf_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.setLocale(QLocale('en_US'))  
        self.conf_label = QLabel(f"{self.conf_slider.value() / 100:.2f}")
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        left_panel_layout.addLayout(conf_layout)

        # Run/Cancel Buttons
        self.run_cancel_layout = QHBoxLayout()
        self.run_button = QPushButton("Start")
        self.run_button.clicked.connect(self.run_code)
        self.run_button.setEnabled(False)
        self.run_button.setStyleSheet(self.button_style_OFF)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_code)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setStyleSheet(self.button_style_OFF)
        self.run_cancel_layout.addWidget(self.run_button)
        self.run_cancel_layout.addWidget(self.cancel_button)
        left_panel_layout.addLayout(self.run_cancel_layout)


        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setLocale(QLocale('en_US'))  
        left_panel_layout.addWidget(self.progress_bar)

        # Log Text
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        left_panel_layout.addWidget(self.log_text_edit)

        # Photo Gallery
        photo_gallery_layout = QVBoxLayout()
        main_layout.addLayout(photo_gallery_layout)

        photo_gallery_label = QLabel("Images")
        photo_gallery_label.setStyleSheet("font-weight: bold; font-size: 16px; padding: 10px;")
        photo_gallery_layout.addWidget(photo_gallery_label)

        self.tab_widget = QTabWidget()
        photo_gallery_layout.addWidget(self.tab_widget)

        self.input_tab = QWidget()
        self.output_tab = QWidget()

        self.tab_widget.addTab(self.input_tab, "Input")
        self.tab_widget.addTab(self.output_tab, "Output")

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

    def update_classes(self, text):
        self.classes = [c.strip() for c in text.split(',')]

    def update_slice_size_input(self, state):
        self.slice_size_input.setEnabled(state)

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
            try:
                self.load_photos(images_folder, "input")
                self.check_run_button_state()
            except Exception as e:
                self.show_error_message(str(e))

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

        if tab_type == "input":
            image_files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]
        elif tab_type == "train":
            train_dir = os.path.join(folder, "detected")
            if not os.path.exists(train_dir):
                return
            image_files = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]
        elif tab_type == "val":
            val_dir = os.path.join(folder, "detected")
            if not os.path.exists(val_dir):
                return
            image_files = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]

        row = 0
        col = 0
        for image_file in image_files:
            if tab_type == "input":
                img_path = os.path.join(folder, image_file)
            elif tab_type == "train":
                img_path = os.path.join(train_dir, image_file)
            elif tab_type == "val":
                img_path = os.path.join(val_dir, image_file)

            try:
                pixmap = QPixmap(img_path).scaled(150, 180, Qt.AspectRatioMode.KeepAspectRatio)
                label = QLabel()
                label.setPixmap(pixmap)
                label.setStyleSheet("border: 1px solid #ccc; padding: 5px;")
                grid_layout.addWidget(label, row, col)
                col += 1
                if col >= 3:
                    row += 1
                    col = 0
            except Exception as e:
                self.show_error_message(f"Error loading image '{image_file}': {str(e)}")

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
        self.val_label.setText(f"{value / 100:.2f}")

    def update_conf_label(self, value):
        self.conf_label.setText(f"{value / 100:.2f}")

    def check_run_button_state(self):
        model_path = self.model_path_line_edit.text()
        images_folder = self.images_folder_line_edit.text()
        output_directory = self.output_directory_line_edit.text()
        self.run_button.setEnabled(bool(model_path and images_folder and output_directory))

        self.run_button.setStyleSheet(self.button_style_ON)        
        
    def run_code(self):
        model_path = self.model_path_line_edit.text()
        images_folder = self.images_folder_line_edit.text()
        output_directory = self.output_directory_line_edit.text()
        use_sahi = self.sahi_checkbox.isChecked()
        slice_size = self.slice_size_input.value() 
        val_split = self.val_slider.value() / 100
        conf_threshold = self.conf_slider.value() / 100
        output_format = self.output_format_combo_box.currentText()
        device = "cpu" if self.device_combo_box.currentText() == "CPU" else "0"

        try:
            self.clear_output()
            self.worker_thread = ObjectDetectionWorker(conf_threshold, model_path, images_folder, output_directory, val_split, output_format, self.classes, device, use_sahi, slice_size)
            self.worker_thread.finished.connect(self.on_worker_finished)
            self.worker_thread.progress.connect(self.progress_bar.setValue)
            self.worker_thread.log_update.connect(self.update_log)
            self.worker_thread.output_image_signal.connect(self.on_output_image)
            self.worker_thread.start()
            
            self.log_text_edit.append("Process Started.")
            self.model_path_button.setEnabled(False)
            self.images_folder_button.setEnabled(False)
            self.output_directory_button.setEnabled(False)
            self.run_button.setEnabled(False)
            self.classes_line_edit.setEnabled(False)
            self.model_path_line_edit.setEnabled(False)
            self.images_folder_line_edit.setEnabled(False)
            self.output_directory_line_edit.setEnabled(False)
            self.sahi_checkbox.setEnabled(False)
            self.slice_size_input.setEnabled(False)
            self.device_combo_box.setEnabled(False)
            self.output_format_combo_box.setEnabled(False)
            self.val_slider.setEnabled(False)
            self.conf_slider.setEnabled(False)
            self.cancel_button.setEnabled(True)


            self.model_path_button.setStyleSheet(self.button_style_OFF)
            self.images_folder_button.setStyleSheet(self.button_style_OFF)
            self.output_directory_button.setStyleSheet(self.button_style_OFF)
            self.run_button.setStyleSheet(self.button_style_OFF)
            self.sahi_checkbox.setStyleSheet(self.button_style_OFF)
            self.slice_size_input.setStyleSheet(self.button_style_OFF)
            self.device_combo_box.setStyleSheet(self.button_style_OFF)
            self.output_format_combo_box.setStyleSheet(self.button_style_OFF)
            self.val_slider.setStyleSheet(self.button_style_OFF)
            self.conf_slider.setStyleSheet(self.button_style_OFF)

            self.cancel_button.setStyleSheet(self.button_style_ON) 

            self.progress_bar.setValue(0)
        except Exception as e:
            self.show_error_message(str(e))

    def on_output_image(self, image_path, image_file):
        pixmap = QPixmap(image_path)
        label = QLabel()
        label.setPixmap(pixmap)
        label.setStyleSheet("border: 1px solid #ccc; padding: 5px;")

        if image_file.startswith("train"):
            self.train_photo_grid_layout.addWidget(label)
        elif image_file.startswith("val"):
            self.val_photo_grid_layout.addWidget(label)

    def clear_output(self):
        for i in reversed(range(self.train_photo_grid_layout.count())):
            label = self.train_photo_grid_layout.itemAt(i).widget()
            if isinstance(label, QLabel):
                self.train_photo_grid_layout.removeWidget(label)
                label.setParent(None)
                label.deleteLater()

        for i in reversed(range(self.val_photo_grid_layout.count())):
            label = self.val_photo_grid_layout.itemAt(i).widget()
            if isinstance(label, QLabel):
                self.val_photo_grid_layout.removeWidget(label)
                label.setParent(None)
                label.deleteLater()
    def cancel_code(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread._is_running = False
            self.worker_thread.wait()
            self.worker_thread = None
            self.clear_output()

            self.model_path_button.setEnabled(True)
            self.images_folder_button.setEnabled(True)
            self.output_directory_button.setEnabled(True)
            self.classes_line_edit.setEnabled(True)
            self.model_path_line_edit.setEnabled(True)
            self.images_folder_line_edit.setEnabled(True)
            self.output_directory_line_edit.setEnabled(True)
            self.sahi_checkbox.setEnabled(True)
            self.slice_size_input.setEnabled(True)
            self.device_combo_box.setEnabled(True)
            self.output_format_combo_box.setEnabled(True)
            self.val_slider.setEnabled(True)
            self.conf_slider.setEnabled(True)
            self.run_button.setEnabled(True)


            self.model_path_button.setStyleSheet(self.button_style_ON)
            self.images_folder_button.setStyleSheet(self.button_style_ON)
            self.output_directory_button.setStyleSheet(self.button_style_ON)
            self.run_button.setStyleSheet(self.button_style_ON)
            self.sahi_checkbox.setStyleSheet(self.button_style_ON)
            self.slice_size_input.setStyleSheet(self.button_style_ON)
            self.device_combo_box.setStyleSheet(self.button_style_ON)
            self.output_format_combo_box.setStyleSheet(self.button_style_ON)
            self.val_slider.setStyleSheet(self.button_style_ON)
            self.conf_slider.setStyleSheet(self.button_style_ON)

            self.cancel_button.setStyleSheet(self.button_style_OFF)

        self.progress_bar.setValue(0)
        self.log_text_edit.append("Process Canceled.")
        
    def on_worker_finished(self):
        self.worker_thread = None

        self.model_path_button.setEnabled(True)
        self.images_folder_button.setEnabled(True)
        self.output_directory_button.setEnabled(True)
        self.classes_line_edit.setEnabled(True)
        self.model_path_line_edit.setEnabled(True)
        self.images_folder_line_edit.setEnabled(True)
        self.output_directory_line_edit.setEnabled(True)
        self.sahi_checkbox.setEnabled(True)
        self.slice_size_input.setEnabled(True)
        self.device_combo_box.setEnabled(True)
        self.output_format_combo_box.setEnabled(True)
        self.val_slider.setEnabled(True)
        self.conf_slider.setEnabled(True)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        


        self.model_path_button.setStyleSheet(self.button_style_ON)
        self.images_folder_button.setStyleSheet(self.button_style_ON)
        self.output_directory_button.setStyleSheet(self.button_style_ON)
        self.run_button.setStyleSheet(self.button_style_ON)
        self.sahi_checkbox.setStyleSheet(self.button_style_ON)
        self.slice_size_input.setStyleSheet(self.button_style_ON)
        self.device_combo_box.setStyleSheet(self.button_style_ON)
        self.output_format_combo_box.setStyleSheet(self.button_style_ON)
        self.val_slider.setStyleSheet(self.button_style_ON)
        self.conf_slider.setStyleSheet(self.button_style_ON)

        self.cancel_button.setStyleSheet(self.button_style_OFF)
     
        output_directory = self.output_directory_line_edit.text()
        train_dir = os.path.join(output_directory, "train")
        val_dir = os.path.join(output_directory, "val")

        try:
            self.load_photos(train_dir, "train")
            self.load_photos(val_dir, "val")
        except Exception as e:
            self.show_error_message(str(e))

    def update_log(self, log_message):
        self.log_text_edit.append(log_message)

    def show_error_message(self, message):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Icon.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.exec()