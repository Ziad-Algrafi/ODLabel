import queue
from queue import Queue
import customtkinter
import torch
from tkinter import messagebox, filedialog
from .workers.chart_worker import ChartWorker
from .workers.detection_worker import ObjectDetectionWorker


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("ODLabel")
        self.root.geometry(
            "{}x{}".format(
                int(self.root.winfo_screenwidth() * 0.8),
                int(self.root.winfo_screenheight() * 0.8),
            )
        )
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=90)

        # Left Panel
        self.left_frame = customtkinter.CTkFrame(self.root)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Configure grid weights for the left frame
        for i in range(26):
            self.left_frame.grid_rowconfigure(i, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        # Model Path
        self.selected_model_path_label = customtkinter.CTkLabel(
            self.left_frame, text="", width=100
        )
        self.selected_model_path_label.grid(row=1, column=0, sticky="w")
        self.model_path_button = customtkinter.CTkButton(
            self.left_frame, text="Select YOLO-World Model", command=self.select_model
        )
        self.model_path_button.grid(row=0, column=0, sticky="s")

        # Images Folder
        self.selected_images_folder_label = customtkinter.CTkLabel(
            self.left_frame, text="", width=100
        )
        self.selected_images_folder_label.grid(row=3, column=0, sticky="w")
        self.images_folder_button = customtkinter.CTkButton(
            self.left_frame,
            text="Select Images Folder",
            command=self.select_images_folder,
        )
        self.images_folder_button.grid(row=2, column=0, sticky="s")

        # Output Directory
        self.selected_output_directory_label = customtkinter.CTkLabel(
            self.left_frame, text="", width=100
        )
        self.selected_output_directory_label.grid(row=5, column=0, sticky="w")
        self.output_directory_button = customtkinter.CTkButton(
            self.left_frame,
            text="Select Output Directory",
            command=self.select_output_directory,
        )
        self.output_directory_button.grid(row=4, column=0, sticky="s")

        # Classes
        self.classes_label = customtkinter.CTkLabel(
            self.left_frame, text="Object Categories:"
        )
        self.classes_label.grid(row=6, column=0, sticky="w")
        self.classes_entry = customtkinter.CTkEntry(
            self.left_frame,
            placeholder_text="Enter the classes to be detected, separated by comma",
        )
        self.classes_entry.grid(row=7, column=0, sticky="ew")
        self.classes_entry.bind("<KeyRelease>", self.update_classes)

        # SAHI Checkbox and Slice Size Input
        sahi_frame = customtkinter.CTkLabel(self.left_frame)
        sahi_frame.grid(row=8, column=0, pady=(0, 0), sticky="ew")

        self.sahi_var = customtkinter.BooleanVar()
        self.sahi_checkbox = customtkinter.CTkCheckBox(
            sahi_frame,
            variable=self.sahi_var,
            command=self.update_slice_size_input,
            height=20,
            text="Slicing Adaptive Inference (SAHI)",
        )
        self.sahi_checkbox.grid(row=0, column=0)

        # Slice Size Input
        self.slice_size_frame = customtkinter.CTkFrame(self.left_frame)
        self.slice_size_frame.grid(row=9, column=0, pady=(0, 10), sticky="s")
        self.slice_size_label = customtkinter.CTkLabel(
            self.slice_size_frame, text="Slice Size:"
        )
        self.slice_size_label.grid(row=0, column=0)
        self.slice_size_var = customtkinter.StringVar(value="512")
        self.slice_size_entry = customtkinter.CTkEntry(
            self.slice_size_frame, textvariable=self.slice_size_var, width=80
        )
        self.slice_size_entry.grid(row=0, column=1, padx=5)
        self.slice_size_entry.configure(state="disabled")

        # Device
        self.device_label = customtkinter.CTkLabel(self.left_frame, text="Device Type:")
        self.device_label.grid(row=10, column=0, sticky="sw")
        self.device_frame = customtkinter.CTkFrame(self.left_frame)
        self.device_frame.grid(row=10, column=0, sticky="s")
        self.device_var = customtkinter.StringVar(value="CPU")
        self.device_radiobutton_cpu = customtkinter.CTkRadioButton(
            self.device_frame, text="CPU", variable=self.device_var, value="CPU"
        )
        self.device_radiobutton_cpu.grid(row=0, column=0)
        self.device_radiobutton_gpu = customtkinter.CTkRadioButton(
            self.device_frame, text="GPU", variable=self.device_var, value="GPU"
        )
        self.device_radiobutton_gpu.grid(row=0, column=1)

        # Output Format
        self.output_format_label = customtkinter.CTkLabel(
            self.left_frame, text="Output Format:"
        )
        self.output_format_label.grid(row=12, column=0, sticky="sw")
        self.output_format_var = customtkinter.StringVar(value="YOLO")
        self.output_format_option_menu = customtkinter.CTkOptionMenu(
            self.left_frame,
            values=["YOLO", "COCO", "CSV", "XML"],
            variable=self.output_format_var,
        )
        self.output_format_option_menu.grid(row=12, column=0, sticky="s")

        # Train/Val Split
        self.split_label = customtkinter.CTkLabel(
            self.left_frame, text="Train/Val Split:"
        )
        self.split_label.grid(row=14, column=0, sticky="s")
        self.val_var = customtkinter.DoubleVar(value=0.8)
        self.val_slider = customtkinter.CTkSlider(
            self.left_frame,
            from_=0,
            to=1,
            number_of_steps=20,
            variable=self.val_var,
            command=self.update_val_label,
        )
        self.val_slider.grid(row=15, column=0, sticky="ew")
        self.val_label = customtkinter.CTkLabel(self.left_frame, text="0.80")
        self.val_label.grid(row=16, column=0)

        # Confidence
        self.conf_label = customtkinter.CTkLabel(
            self.left_frame, text="Confidence Level:"
        )
        self.conf_label.grid(row=17, column=0, sticky="s")
        self.conf_var = customtkinter.DoubleVar(value=0.2)
        self.conf_slider = customtkinter.CTkSlider(
            self.left_frame,
            from_=0,
            to=1,
            number_of_steps=100,
            variable=self.conf_var,
            command=self.update_conf_label,
        )
        self.conf_slider.grid(row=18, column=0, sticky="ew")
        self.conf_label = customtkinter.CTkLabel(self.left_frame, text="0.20")
        self.conf_label.grid(row=19, column=0)

        # Union over Intersection iou
        self.iou_label = customtkinter.CTkLabel(
            self.left_frame, text="Non Maximum Suppression:"
        )
        self.iou_label.grid(row=20, column=0, sticky="s")
        self.iou_var = customtkinter.DoubleVar(value=0.7)
        self.iou_slider = customtkinter.CTkSlider(
            self.left_frame,
            from_=0,
            to=1,
            number_of_steps=10,
            variable=self.iou_var,
            command=self.update_iou_label,
        )
        self.iou_slider.grid(row=21, column=0, sticky="ew")
        self.iou_label = customtkinter.CTkLabel(self.left_frame, text="0.70")
        self.iou_label.grid(row=22, column=0)

        # Run/Cancel Buttons
        self.run_cancel_frame = customtkinter.CTkFrame(self.left_frame)
        self.run_cancel_frame.grid(row=23, column=0, pady=10, sticky="s")
        self.run_button = customtkinter.CTkButton(
            self.run_cancel_frame, text="Start", command=self.run_code
        )
        self.run_button.grid(row=0, column=0, padx=5)
        self.cancel_button = customtkinter.CTkButton(
            self.run_cancel_frame, text="Cancel", command=self.cancel_code
        )
        self.cancel_button.grid(row=0, column=1, padx=5)
        self.cancel_button.configure(state="disabled")

        # Set the width of the output format option menu
        self.output_format_option_menu.configure(width=self.run_button.winfo_reqwidth())

        # Progress Bar
        self.progress_bar = customtkinter.CTkProgressBar(self.left_frame, width=400)
        self.progress_bar.grid(row=24, column=0, sticky="ew")
        self.progress_bar.set(0)

        # Log Text
        self.log_text = customtkinter.CTkTextbox(self.left_frame, width=400, height=200)
        self.log_text.grid(row=25, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

        # chart frame
        self.charts_frame = customtkinter.CTkFrame(self.root)
        self.charts_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.tab_control = customtkinter.CTkTabview(self.charts_frame)
        self.tab_control.grid(row=0, column=0, sticky="nsew")

        self.input_tab = self.tab_control.add("Input")
        self.output_tab = self.tab_control.add("Output")

        self.input_chart_frame = customtkinter.CTkFrame(self.input_tab)
        self.input_chart_frame.grid(row=0, column=0, sticky="nsew")

        self.output_tab_control = customtkinter.CTkTabview(self.output_tab)
        self.output_tab_control.grid(row=0, column=0, sticky="nsew")

        self.train_tab = self.output_tab_control.add("Train")
        self.val_tab = self.output_tab_control.add("Val")

        self.train_chart_frame = customtkinter.CTkFrame(self.train_tab)
        self.train_chart_frame.grid(row=0, column=0, sticky="nsew")

        self.val_chart_frame = customtkinter.CTkFrame(self.val_tab)
        self.val_chart_frame.grid(row=0, column=0, sticky="nsew")

        self.charts_frame.grid_rowconfigure(0, weight=1)
        self.charts_frame.grid_columnconfigure(0, weight=1)

        self.input_tab.grid_rowconfigure(0, weight=1)
        self.input_tab.grid_columnconfigure(0, weight=1)

        self.output_tab.grid_rowconfigure(0, weight=1)
        self.output_tab.grid_columnconfigure(0, weight=1)

        self.train_tab.grid_rowconfigure(0, weight=1)
        self.train_tab.grid_columnconfigure(0, weight=1)

        self.val_tab.grid_rowconfigure(0, weight=1)
        self.val_tab.grid_columnconfigure(0, weight=1)

        self.chart_queue = Queue()
        self.worker_thread = None
        self.classes = []
        self.input_figures = None
        self.train_figures = None
        self.val_figures = None
        self.input_chart_worker = None
        self.output_chart_worker = None
        self.show_error_queue = None
        self.log_queue = None
        self.output_image_queue = None
        self.progress_queue = None
        self.train_figures_processed = False
        self.val_figures_processed = False

        # Chart Colors
        self.chart_colors = {
            "images": "#1f77b4",
            "resolution": "#1f77b4",
            "quality": "#1f77b4",
            "format": "#1f77b4",
        }

    def update_classes(self, event):
        text = self.classes_entry.get()
        self.classes = [c.strip() for c in text.split(",")]

    def update_slice_size_input(self):
        if self.sahi_var.get():
            self.slice_size_entry.configure(state="normal")
            self.slice_size_label.configure(state="normal")
        else:
            self.slice_size_entry.configure(state="disabled")
            self.slice_size_label.configure(state="disabled")

    def validate_slice_size(self, value):
        if value.isdigit() and int(value) % 32 == 0:
            return True
        return False

    def select_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("YOLO-WORLD Model", "*.pt")])
        if file_path:
            self.selected_model_path_label.configure(text=file_path)
            self.check_run_button_state()

    def select_images_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.selected_images_folder_label.configure(text=folder_path)
            self.check_run_button_state()

            try:
                self.clear_input_chart()
                self.log_text.configure(state="normal")
                self.log_text.insert(customtkinter.END, "Processing input charts...\n")
                self.log_text.configure(state="disabled")
                self.input_chart_worker = ChartWorker(folder_path, self.chart_colors)
                self.input_chart_worker.queue = self.chart_queue
                self.input_chart_worker.start()
                self.monitor_input_chart_worker()
            except Exception as e:
                self.show_error_message(str(e))

    def select_output_directory(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.selected_output_directory_label.configure(text=folder_path)
            self.check_run_button_state()

    def clear_input_chart(self):
        if self.input_chart_worker:
            self.input_chart_worker.close_figures()
        for widget in self.input_chart_frame.winfo_children():
            widget.destroy()

    def update_val_label(self, value):
        self.val_label.configure(text=f"{float(value):.2f}")

    def update_conf_label(self, value):
        self.conf_label.configure(text=f"{float(value):.2f}")

    def update_iou_label(self, value):
        self.iou_label.configure(text=f"{float(value):.2f}")

    def check_run_button_state(self):
        model_path = self.selected_model_path_label.cget("text")
        images_folder = self.selected_images_folder_label.cget("text")
        output_directory = self.selected_output_directory_label.cget("text")
        if model_path and images_folder and output_directory:
            self.run_button.configure(state="normal")
        else:
            self.run_button.configure(state="disabled")

    def run_code(self):
        model_path = self.selected_model_path_label.cget("text")
        images_folder = self.selected_images_folder_label.cget("text")
        output_directory = self.selected_output_directory_label.cget("text")
        use_sahi = self.sahi_var.get()
        slice_size = int(self.slice_size_var.get())
        train_split = self.val_var.get()
        val_split = 1 - train_split
        conf_threshold = self.conf_var.get()
        iou_threshold = self.iou_var.get()
        output_format = self.output_format_var.get()
        device = "cpu"

        if not self.classes:
            self.show_error_message("Please enter class names.")
            return
        
        if self.device_var.get() == "GPU":
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                device = "0"
                gpu_name = torch.cuda.get_device_name(0)
                self.log_text.configure(state="normal")
                self.log_text.insert(customtkinter.END, f"Using GPU: {gpu_name}\n")
                self.log_text.configure(state="disabled")
            else:
                device = "cpu"
                self.log_text.configure(state="normal")
                self.log_text.insert(customtkinter.END, "No GPU found. Using CPU instead.\n")
                self.log_text.configure(state="disabled")       
        else:
            self.log_text.configure(state="normal")
            self.log_text.insert(customtkinter.END, "Using CPU.\n")
            self.log_text.configure(state="disabled")  

        try:
            self.clear_output()
            self.worker_thread = ObjectDetectionWorker(
                conf_threshold,
                model_path,
                images_folder,
                output_directory,
                val_split,
                output_format,
                self.classes,
                device,
                use_sahi,
                slice_size,
                iou_threshold,
            )
            self.worker_thread.log_queue = queue.Queue()
            self.worker_thread.progress_queue = queue.Queue()
            self.worker_thread.show_error_queue = queue.Queue()

            self.progress_bar.set(0)
            self.worker_thread.start()

            self.monitor_worker_thread()

            self.log_text.configure(state="normal")
            self.log_text.insert(customtkinter.END, "Process Started.\n")
            self.log_text.configure(state="disabled")

            self.update_ui_state("disabled")

        except Exception as e:
            self.show_error_message(str(e))

    def monitor_worker_thread(self):
        if self.worker_thread is None:
            return

        if self.worker_thread.is_alive():
            if (
                self.worker_thread.progress_queue is not None
                and not self.worker_thread.progress_queue.empty()
            ):
                progress_value = self.worker_thread.progress_queue.get()
                self.update_progress(progress_value)

            if (
                self.worker_thread.show_error_queue is not None
                and not self.worker_thread.show_error_queue.empty()
            ):
                error_message = self.worker_thread.show_error_queue.get()
                self.show_error_message(error_message)

            if (
                self.worker_thread.log_queue is not None
                and not self.worker_thread.log_queue.empty()
            ):
                log_message = self.worker_thread.log_queue.get()
                self.update_log(log_message)

            self.root.after(100, self.monitor_worker_thread)
        else:
            self.on_worker_finished()

    def monitor_input_chart_worker(self):
        if self.input_chart_worker is None:
            return

        if not self.input_chart_worker.is_alive():
            while not self.chart_queue.empty():
                chart_data = self.chart_queue.get()
                chart_type, data = chart_data
                if chart_type == "input_figures":
                    self.process_input_figures(data)
                    self.on_input_chart_worker_finished()
                elif chart_type == "error":
                    self.log_text.configure(state="normal")
                    self.log_text.insert(
                        customtkinter.END, f"Error generating input charts: {data}\n"
                    )
                    self.log_text.configure(state="disabled")
                    self.show_error_message("No images found in the selected folder.")

        else:
            self.root.after(100, self.monitor_input_chart_worker)

    def monitor_output_chart_worker(self):
        if self.output_chart_worker is None:
            return

        self.train_figures_processed = False
        self.val_figures_processed = False

        if not self.output_chart_worker.is_alive():
            while not self.chart_queue.empty():
                chart_data = self.chart_queue.get()
                chart_type, figures = chart_data
                if chart_type == "train_figures":
                    self.process_train_figures(figures)
                    self.train_figures_processed = True
                elif chart_type == "val_figures":
                    self.process_val_figures(figures)
                    self.val_figures_processed = True

            self.on_output_chart_worker_finished()
        else:
            self.root.after(100, self.monitor_output_chart_worker)

    def clear_output(self):
        for widget in self.train_chart_frame.winfo_children():
            widget.destroy()
        for widget in self.val_chart_frame.winfo_children():
            widget.destroy()

    def cancel_code(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.terminate()
            self.worker_thread.join()
            self.worker_thread = None
            self.progress_bar.set(0)

            self.log_text.configure(state="normal")
            self.log_text.insert(customtkinter.END, "Process Canceled.\n")
            self.log_text.configure(state="disabled")

            self.update_ui_state("normal")

        self.on_worker_finished()

    def process_input_figures(self, figures):
        self.clear_input_chart()
        if figures and self.input_chart_worker is not None:
            self.input_chart_worker.display_chart_in_frame(
                figures[0], self.input_chart_frame, row=0, col=0
            )
            self.input_chart_worker.display_chart_in_frame(
                figures[1], self.input_chart_frame, row=0, col=1
            )
            self.input_chart_worker.display_chart_in_frame(
                figures[2], self.input_chart_frame, row=1, col=0
            )
            self.input_chart_worker.display_chart_in_frame(
                figures[3], self.input_chart_frame, row=1, col=1
            )

    def process_train_figures(self, figures):
        self.clear_output_charts(self.train_chart_frame)
        if figures and self.output_chart_worker is not None:
            self.output_chart_worker.display_chart_in_frame(
                figures[0], self.train_chart_frame, row=0, col=0
            )
            self.output_chart_worker.display_chart_in_frame(
                figures[1], self.train_chart_frame, row=1, col=1
            )
            self.output_chart_worker.display_chart_in_frame(
                figures[2], self.train_chart_frame, row=0, col=1
            )
            self.output_chart_worker.display_chart_in_frame(
                figures[3], self.train_chart_frame, row=1, col=0
            )

    def process_val_figures(self, figures):
        self.clear_output_charts(self.val_chart_frame)
        if figures and self.output_chart_worker is not None:
            self.output_chart_worker.display_chart_in_frame(
                figures[0], self.val_chart_frame, row=0, col=0
            )
            self.output_chart_worker.display_chart_in_frame(
                figures[1], self.val_chart_frame, row=1, col=1
            )
            self.output_chart_worker.display_chart_in_frame(
                figures[2], self.val_chart_frame, row=0, col=1
            )
            self.output_chart_worker.display_chart_in_frame(
                figures[3], self.val_chart_frame, row=1, col=0
            )

    def on_input_chart_worker_finished(self):
        self.log_text.configure(state="normal")
        self.log_text.insert(
            customtkinter.END, "Input charts processed successfully.\n"
        )
        self.log_text.configure(state="disabled")

        self.input_chart_worker = None

    def on_output_chart_worker_finished(self):
        if self.train_figures_processed and self.val_figures_processed:
            self.log_text.configure(state="normal")
            self.log_text.insert(
                customtkinter.END, "Output charts processed successfully.\n"
            )
            self.log_text.configure(state="disabled")
        elif self.train_figures_processed and not self.val_figures_processed:
            if self.val_var.get() > 0:
                self.log_text.configure(state="normal")
                self.log_text.insert(
                    customtkinter.END, "Training charts processed successfully.\n"
                )
                self.log_text.insert(
                    customtkinter.END, "No detections found in validation images.\n"
                )
                self.log_text.configure(state="disabled")
            else:
                self.log_text.configure(state="normal")
                self.log_text.insert(
                    customtkinter.END, "Training charts processed successfully.\n"
                )
                self.log_text.configure(state="disabled")
        elif not self.train_figures_processed and self.val_figures_processed:
            if self.val_var.get() < 1:
                self.log_text.configure(state="normal")
                self.log_text.insert(
                    customtkinter.END, "No detections found in training images.\n"
                )
                self.log_text.insert(
                    customtkinter.END, "Validation charts processed successfully.\n"
                )
                self.log_text.configure(state="disabled")
            else:
                self.log_text.configure(state="normal")
                self.log_text.insert(
                    customtkinter.END, "Validation charts processed successfully.\n"
                )
                self.log_text.configure(state="disabled")
        elif not self.train_figures_processed and not self.val_figures_processed:
            if self.val_var.get() < 1:
                self.log_text.configure(state="normal")
                self.log_text.insert(
                    customtkinter.END, "No detections found in training images.\n"
                )
                self.log_text.configure(state="disabled")
            if self.val_var.get() > 0:
                self.log_text.configure(state="normal")
                self.log_text.insert(
                    customtkinter.END, "No detections found in validation images.\n"
                )
                self.log_text.configure(state="disabled")

        self.output_chart_worker = None

    def on_worker_finished(self):
        if self.worker_thread is not None:
            train_detections, val_detections, actual_train_count, actual_val_count = (
                self.worker_thread.get_detections()
            )

            if self.worker_thread.log_queue is not None:
                while not self.worker_thread.log_queue.empty():
                    log_message = self.worker_thread.log_queue.get()
                    self.update_log(log_message)

            if self.worker_thread.show_error_queue is not None:
                while not self.worker_thread.show_error_queue.empty():
                    error_message = self.worker_thread.show_error_queue.get()
                    self.show_error_message(error_message)

        else:
            train_detections, val_detections, actual_train_count, actual_val_count = (
                [],
                [],
                0,
                0,
            )

        self.worker_thread = None
        self.progress_bar.set(100)

        self.log_text.configure(state="normal")
        self.log_text.insert(customtkinter.END, "Processing output charts...\n")
        self.log_text.configure(state="disabled")

        self.update_ui_state("normal")

        if actual_train_count == 0 and actual_val_count == 0:
            self.log_text.configure(state="normal")
            self.log_text.insert(
                customtkinter.END, "No detections found in any image.\n"
            )
            self.log_text.configure(state="disabled")
            return

        try:
            self.output_chart_worker = ChartWorker(
                self.selected_images_folder_label.cget("text"),
                self.chart_colors,
                train_detections,
                val_detections,
                actual_train_count,
                actual_val_count,
            )
            self.output_chart_worker.queue = self.chart_queue
            self.output_chart_worker.start()
            self.monitor_output_chart_worker()
        except Exception as e:
            self.show_error_message(str(e))

    def clear_output_charts(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def display_chart(self, fig, frame, row, col):
        if self.input_chart_worker is not None:
            self.input_chart_worker.display_chart_in_frame(fig, frame, row, col)

    def update_log(self, log_message):
        self.log_text.configure(state="normal")
        self.log_text.insert(customtkinter.END, log_message + "\n")
        self.log_text.see(customtkinter.END)
        self.log_text.configure(state="disabled")

    def show_error_message(self, message):
        messagebox.showerror("Error", message)

    def update_progress(self, progress_value):
        self.progress_bar.set(progress_value / 100)

    def update_ui_state(self, state):
        self.run_button.configure(state=state)
        self.cancel_button.configure(
            state="normal" if state == "disabled" else "disabled"
        )
        self.model_path_button.configure(state=state)
        self.selected_model_path_label.configure(state=state)
        self.images_folder_button.configure(state=state)
        self.selected_images_folder_label.configure(state=state)
        self.output_directory_button.configure(state=state)
        self.selected_output_directory_label.configure(state=state)
        self.sahi_checkbox.configure(state=state)
        self.classes_entry.configure(state=state)
        self.classes_label.configure(state=state)
        self.slice_size_label.configure(state=state)
        self.slice_size_entry.configure(state=state)
        self.device_label.configure(state=state)
        self.device_radiobutton_cpu.configure(state=state)
        self.device_radiobutton_gpu.configure(state=state)
        self.output_format_option_menu.configure(state=state)
        self.conf_slider.configure(state=state)
        self.val_slider.configure(state=state)
        self.iou_slider.configure(state=state)

    def on_closing(self):
        if self.input_chart_worker:
            self.input_chart_worker.close_figures()
        if self.output_chart_worker:
            self.output_chart_worker.close_figures()
        self.root.quit()
        self.root.destroy()
