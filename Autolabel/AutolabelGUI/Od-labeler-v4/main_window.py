import queue
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter
from tkinter import messagebox, filedialog
from thread_worker import ObjectDetectionWorker

import matplotlib.pyplot as plt

from chart_utils import update_input_chart, update_output_charts

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("OD-Labeler")
        self.root.geometry("1280x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


        # Left Panel
        self.left_frame = customtkinter.CTkFrame(self.root)
        self.left_frame.pack(side=customtkinter.LEFT, padx=10, pady=10)

        # Model Path
        self.model_path_label = customtkinter.CTkLabel(self.left_frame, text="Model Path:")
        self.model_path_label.pack()
        self.model_path_entry = customtkinter.CTkEntry(self.left_frame, width=320)
        self.model_path_entry.pack()
        self.model_path_button = customtkinter.CTkButton(self.left_frame, text="Select YOLO Model", command=self.select_model)
        self.model_path_button.pack()

        # Images Folder
        self.images_folder_label = customtkinter.CTkLabel(self.left_frame, text="Images Folder:")
        self.images_folder_label.pack()
        self.images_folder_entry = customtkinter.CTkEntry(self.left_frame, width=320)
        self.images_folder_entry.pack()
        self.images_folder_button = customtkinter.CTkButton(self.left_frame, text="Select Images Folder", command=self.select_images_folder)
        self.images_folder_button.pack()

        # Output Directory
        self.output_directory_label = customtkinter.CTkLabel(self.left_frame, text="Output Directory:")
        self.output_directory_label.pack()
        self.output_directory_entry = customtkinter.CTkEntry(self.left_frame, width=320)
        self.output_directory_entry.pack()
        self.output_directory_button = customtkinter.CTkButton(self.left_frame, text="Select Output Directory", command=self.select_output_directory)
        self.output_directory_button.pack()

        # Classes
        self.classes_label = customtkinter.CTkLabel(self.left_frame, text="Classes:")
        self.classes_label.pack()
        self.classes_entry = customtkinter.CTkEntry(self.left_frame, width=320)
        self.classes_entry.pack()
        self.classes_entry.bind("<KeyRelease>", self.update_classes)

        # SAHI Checkbox and Slice Size Input
        self.sahi_var = customtkinter.BooleanVar()
        self.sahi_checkbox = customtkinter.CTkCheckBox(self.left_frame, text="Slicing Adaptive Inference (SAHI)", variable=self.sahi_var, command=self.update_slice_size_input)
        self.sahi_checkbox.pack()


        # Slice Size Input
        self.slice_size_label = customtkinter.CTkLabel(self.left_frame, text="Slice Size:")
        self.slice_size_label.pack()
        self.slice_size_var = customtkinter.StringVar(value="512")
        self.slice_size_entry = customtkinter.CTkEntry(self.left_frame, textvariable=self.slice_size_var, width=320)
        self.slice_size_entry.pack()
        self.slice_size_entry.configure(state="disabled")

        # Device
        self.device_label = customtkinter.CTkLabel(self.left_frame, text="Device Type:")
        self.device_label.pack()
        self.device_frame = customtkinter.CTkFrame(self.left_frame)
        self.device_frame.pack()
        self.device_var = customtkinter.StringVar(value="CPU")
        self.device_radiobutton_cpu = customtkinter.CTkRadioButton(self.device_frame, text="CPU", variable=self.device_var, value="CPU")
        self.device_radiobutton_cpu.pack(side=customtkinter.LEFT)
        self.device_radiobutton_gpu = customtkinter.CTkRadioButton(self.device_frame, text="GPU", variable=self.device_var, value="GPU")
        self.device_radiobutton_gpu.pack(side=customtkinter.LEFT)

        # Output Format
        self.output_format_label = customtkinter.CTkLabel(self.left_frame, text="Output Format:")
        self.output_format_label.pack()
        self.output_format_var = customtkinter.StringVar(value="YOLO")
        self.output_format_option_menu = customtkinter.CTkOptionMenu(self.left_frame, values=["YOLO", "COCO", "CSV", "XML"], variable=self.output_format_var)
        self.output_format_option_menu.pack()

        # Train/Val Split
        self.split_label = customtkinter.CTkLabel(self.left_frame, text="Train/Val Split:")
        self.split_label.pack()
        self.val_var = customtkinter.DoubleVar(value=0.2)
        self.val_slider = customtkinter.CTkSlider(self.left_frame, from_=0, to=1, number_of_steps=20, variable=self.val_var, command=self.update_val_label)
        self.val_slider.pack()
        self.val_label = customtkinter.CTkLabel(self.left_frame, text="0.20")
        self.val_label.pack()

        # Confidence
        self.conf_label = customtkinter.CTkLabel(self.left_frame, text="Confidence:")
        self.conf_label.pack()
        self.conf_var = customtkinter.DoubleVar(value=0.2)
        self.conf_slider = customtkinter.CTkSlider(self.left_frame, from_=0, to=1, number_of_steps=20, variable=self.conf_var, command=self.update_conf_label)
        self.conf_slider.pack()
        self.conf_label = customtkinter.CTkLabel(self.left_frame, text="0.20")
        self.conf_label.pack()

        # Run/Cancel Buttons
        self.run_cancel_frame = customtkinter.CTkFrame(self.left_frame)
        self.run_cancel_frame.pack()
        self.run_button = customtkinter.CTkButton(self.run_cancel_frame, text="Start", command=self.run_code)
        self.run_button.pack(side=customtkinter.LEFT, padx=5)
        self.cancel_button = customtkinter.CTkButton(self.run_cancel_frame, text="Cancel", command=self.cancel_code)
        self.cancel_button.pack(side=customtkinter.LEFT, padx=5)
        self.cancel_button.configure(state="disabled")

        # Progress Bar
        self.progress_bar = customtkinter.CTkProgressBar(self.left_frame, width=400)
        self.progress_bar.pack()
        self.progress_bar.set(0)

        # Log Text
        self.log_text = customtkinter.CTkTextbox(self.left_frame, width=400, height=200)
        self.log_text.pack()
        self.log_text.configure(state="disabled")

        # Photo Gallery
        self.photo_gallery_frame = customtkinter.CTkFrame(self.root)
        self.photo_gallery_frame.pack(side=customtkinter.RIGHT, padx=10, pady=10, fill=customtkinter.BOTH, expand=True)

        self.tab_control = customtkinter.CTkTabview(self.photo_gallery_frame)
        self.tab_control.pack(fill=customtkinter.BOTH, expand=True)

        self.input_tab = self.tab_control.add("Input")
        self.output_tab = self.tab_control.add("Output")

        self.input_chart_frame = customtkinter.CTkFrame(self.input_tab)
        self.input_chart_frame.pack(fill=customtkinter.BOTH, expand=True)

        self.output_tab_control = customtkinter.CTkTabview(self.output_tab)
        self.output_tab_control.pack(fill=customtkinter.BOTH, expand=True)

        self.train_tab = self.output_tab_control.add("Train")
        self.val_tab = self.output_tab_control.add("Val")

        self.train_chart_frame = customtkinter.CTkFrame(self.train_tab)
        self.train_chart_frame.pack(fill=customtkinter.BOTH, expand=True)

        self.val_chart_frame = customtkinter.CTkFrame(self.val_tab)
        self.val_chart_frame.pack(fill=customtkinter.BOTH, expand=True)

        self.worker_thread = None
        self.classes = []

        # Chart Colors
        self.chart_colors = {
            'images': '#1f77b4',
            'resolution': '#1f77b4',
            'quality': '#1f77b4',
            'format': '#1f77b4'
        }

    def update_classes(self, event):
        text = self.classes_entry.get()
        self.classes = [c.strip() for c in text.split(',')]

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
        file_path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if file_path:
            self.model_path_entry.delete(0, customtkinter.END)
            self.model_path_entry.insert(0, file_path)
            self.check_run_button_state()

    def select_images_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.images_folder_entry.delete(0, customtkinter.END)
            self.images_folder_entry.insert(0, folder_path)
            try:
                self.update_input_chart(folder_path)
                self.check_run_button_state()
            except Exception as e:
                self.show_error_message(str(e))

    def update_input_chart(self, images_folder):
        try:
            figure = update_input_chart(images_folder, self.chart_colors)
            self.clear_input_chart()
            canvas = FigureCanvasTkAgg(figure, master=self.input_chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=customtkinter.BOTH, expand=True)
        except Exception as e:
            self.show_error_message(str(e))

    def clear_input_chart(self):
        for widget in self.input_chart_frame.winfo_children():
            widget.destroy()

    def select_output_directory(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_directory_entry.delete(0, customtkinter.END)
            self.output_directory_entry.insert(0, folder_path)
            self.check_run_button_state()

    def update_val_label(self, value):
        self.val_label.configure(text=f"{float(value):.2f}")

    def update_conf_label(self, value):
        self.conf_label.configure(text=f"{float(value):.2f}")

    def check_run_button_state(self):
        model_path = self.model_path_entry.get()
        images_folder = self.images_folder_entry.get()
        output_directory = self.output_directory_entry.get()
        if model_path and images_folder and output_directory:
            self.run_button.configure(state="normal")
        else:
            self.run_button.configure(state="disabled")

    def run_code(self):
        model_path = self.model_path_entry.get()
        images_folder = self.images_folder_entry.get()
        output_directory = self.output_directory_entry.get()
        use_sahi = self.sahi_var.get()
        slice_size = int(self.slice_size_var.get())
        val_split = self.val_var.get()
        conf_threshold = self.conf_var.get()
        output_format = self.output_format_var.get()
        device = "cpu" if self.device_var.get() == "CPU" else "0"

        try:
            self.clear_output()
            self.worker_thread = ObjectDetectionWorker(conf_threshold, model_path, images_folder, output_directory, val_split, output_format, self.classes, device, use_sahi, slice_size)
            self.worker_thread.log_queue = queue.Queue()
            self.worker_thread.progress_queue = queue.Queue()
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
        if self.worker_thread.is_alive():
            while not self.worker_thread.progress_queue.empty():
                progress_value = self.worker_thread.progress_queue.get()
                self.update_progress(progress_value)

            while not self.worker_thread.log_queue.empty():
                log_message = self.worker_thread.log_queue.get()
                self.update_log(log_message)

            self.root.after(100, self.monitor_worker_thread)
        else:
            self.on_worker_finished()

    def clear_output(self):
        for widget in self.train_chart_frame.winfo_children():
            widget.destroy()
        for widget in self.val_chart_frame.winfo_children():
            widget.destroy()

    def cancel_code(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.terminate()
            self.worker_thread = None
            self.clear_output()
            self.progress_bar.set(0)

            self.log_text.configure(state="normal")
            self.log_text.insert(customtkinter.END, "Process Canceled.\n")
            self.log_text.configure(state="disabled")

            self.update_ui_state("normal")

    def on_worker_finished(self):
        train_detections, val_detections = self.worker_thread.get_detections()
        self.worker_thread = None
        self.progress_bar.set(100)

        self.log_text.configure(state="normal")
        self.log_text.insert(customtkinter.END, "Process Finished.\n")
        self.log_text.configure(state="disabled")

        self.update_ui_state("normal")

        try:
            self.update_output_charts(train_detections, val_detections)
        except Exception as e:
            self.show_error_message(str(e))

    def update_output_charts(self, train_detections, val_detections):
        if not train_detections and not val_detections:
            return

        # Train charts
        if train_detections:
            train_fig_count, train_fig_conf, train_fig_labels = update_output_charts(train_detections, [])
            self.clear_output_charts(self.train_chart_frame)
            self.display_chart(train_fig_count, self.train_chart_frame, row=0, col=0)
            self.display_chart(train_fig_labels, self.train_chart_frame, row=0, col=1)
            self.display_chart(train_fig_conf, self.train_chart_frame, row=1, col=0, colspan=2)

        # Val charts
        if val_detections:
            val_fig_count, val_fig_conf, val_fig_labels = update_output_charts([], val_detections)
            self.clear_output_charts(self.val_chart_frame)
            self.display_chart(val_fig_count, self.val_chart_frame, row=0, col=0)
            self.display_chart(val_fig_labels, self.val_chart_frame, row=0, col=1)
            self.display_chart(val_fig_conf, self.val_chart_frame, row=1, col=0, colspan=2)

    def clear_output_charts(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def display_chart(self, fig, frame, row, col, colspan=1):
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=row, column=col, columnspan=colspan, sticky='nsew')

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
        self.cancel_button.configure(state="normal" if state == "disabled" else "disabled")
        self.model_path_button.configure(state=state)
        self.model_path_entry.configure(state=state)
        self.images_folder_button.configure(state=state)
        self.images_folder_entry.configure(state=state)
        self.output_directory_button.configure(state=state)
        self.output_directory_entry.configure(state=state)
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

    def on_closing(self):
        # Clean up Matplotlib resources
        plt.close('all')
        self.root.destroy()