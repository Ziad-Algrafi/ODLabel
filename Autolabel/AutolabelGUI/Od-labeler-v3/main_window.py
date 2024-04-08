from tkinter import filedialog, messagebox, Label

import os
import queue
from PIL import Image, ImageTk
from customtkinter import CTkImage
import customtkinter
from thread_worker import ObjectDetectionWorker

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("OD-Labeler")
        self.root.geometry("1280x900")

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
        self.photo_gallery_frame.pack(side=customtkinter.RIGHT, padx=10, pady=10)

        self.tab_control = customtkinter.CTkTabview(self.photo_gallery_frame)
        self.tab_control.pack()

        self.input_tab = customtkinter.CTkFrame(self.tab_control)
        self.output_tab = customtkinter.CTkFrame(self.tab_control)

        self.tab_control.add("Input")
        self.tab_control.add("Output")

        self.input_photo_grid_frame = customtkinter.CTkFrame(self.input_tab, width=800, height=600)
        self.input_photo_grid_frame.pack(fill=customtkinter.BOTH, expand=True)

        self.output_tab_control = customtkinter.CTkTabview(self.output_tab, width=800, height=600)
        self.output_tab_control.pack(fill=customtkinter.BOTH, expand=True)

        self.train_tab = customtkinter.CTkFrame(self.output_tab_control)
        self.val_tab = customtkinter.CTkFrame(self.output_tab_control)

        self.output_tab_control.add("Train")
        self.output_tab_control.add("Val")


        self.train_photo_grid_frame = customtkinter.CTkFrame(self.train_tab, width=800, height=600)
        self.train_photo_grid_frame.pack(fill=customtkinter.BOTH, expand=True)

        self.val_photo_grid_frame = customtkinter.CTkFrame(self.val_tab, width=800, height=600)
        self.val_photo_grid_frame.pack(fill=customtkinter.BOTH, expand=True)

        self.worker_thread = None
        self.classes = []

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
                self.load_photos(folder_path, "input")
                self.check_run_button_state()
            except Exception as e:
                self.show_error_message(str(e))

    def load_photos(self, images_folder, tab_type):
        frame = None
        if tab_type == "input":
            frame = self.input_photo_grid_frame
        elif tab_type == "train":
            frame = self.train_photo_grid_frame
        elif tab_type == "val":
            frame = self.val_photo_grid_frame

        if frame is None:
            return

        self.clear_photo_grid(frame)

        if tab_type == "input":
            image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]
        elif tab_type == "train":
            train_dir = os.path.join(images_folder, "detected")
            if not os.path.exists(train_dir):
                return
            image_files = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]
        elif tab_type == "val":
            val_dir = os.path.join(images_folder, "detected")
            if not os.path.exists(val_dir):
                return
            image_files = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]

        row = 0
        col = 0
        for image_file in image_files:
            if tab_type == "input":
                img_path = os.path.join(images_folder, image_file)
            elif tab_type == "train":
                img_path = os.path.join(train_dir, image_file)
            elif tab_type == "val":
                img_path = os.path.join(val_dir, image_file)

            try:
                image = Image.open(img_path)
                image = image.resize((150, 150), resample=Image.BICUBIC)
                photo = ImageTk.PhotoImage(image)
                label = Label(frame, image=photo)
                label.image = photo
                label.grid(row=row, column=col, padx=5, pady=5)
                col += 1
                if col >= 3:
                    row += 1
                    col = 0
            except Exception as e:
                self.show_error_message(f"Error loading image '{image_file}': {str(e)}")

    def clear_photo_grid(self, photo_grid_frame):
        for widget in photo_grid_frame.winfo_children():
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
            self.worker_thread.output_image_queue = queue.Queue()
            self.worker_thread.progress_queue = queue.Queue()
            self.progress_bar.set(0)
            self.worker_thread.start()

            self.monitor_worker_thread() 


            self.log_text.configure(state="normal")
            self.log_text.insert(customtkinter.END, "Process Started.\n")

            self.run_button.configure(state="disabled")
            self.cancel_button.configure(state="normal")

            self.model_path_button.configure(state="disabled")
            self.model_path_entry.configure(state="disabled")

            self.images_folder_button.configure(state="disabled")
            self.images_folder_entry.configure(state="disabled")

            self.output_directory_button.configure(state="disabled")
            self.output_directory_entry.configure(state="disabled")
            self.sahi_checkbox.configure(state="disabled")

            self.classes_entry.configure(state="disabled")
            self.classes_label.configure(state="disabled")

            self.slice_size_label.configure(state="disabled")
            self.slice_size_entry.configure(state="disabled")

            self.device_label.configure(state="disabled")
            self.device_radiobutton_cpu.configure(state="disabled")
            self.device_radiobutton_gpu.configure(state="disabled")

            self.output_format_option_menu.configure(state="disabled")

            self.conf_slider.configure(state="disabled")
            self.val_slider.configure(state="disabled")

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

    def on_output_image(self, image_path, image_file):
        if image_file.startswith("train"):
            photo_grid_frame = self.train_photo_grid_frame
        elif image_file.startswith("val"):
            photo_grid_frame = self.val_photo_grid_frame

        image = Image.open(image_path)
        image.thumbnail((150, 180))
        photo = CTkImage(image)
        label = customtkinter.CTkLabel(photo_grid_frame, image=photo, text="")
        label.image = photo

        row = len(photo_grid_frame.winfo_children()) // 3
        col = len(photo_grid_frame.winfo_children()) % 3
        label.grid(row=row, column=col, padx=5, pady=5)   

    def clear_output(self):
        self.clear_photo_grid(self.train_photo_grid_frame)
        self.clear_photo_grid(self.val_photo_grid_frame)


    def cancel_code(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.terminate()
            self.worker_thread = None
            self.clear_output()
            self.progress_bar.set(0)

            self.run_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
            self.log_text.configure(state="normal")
            self.log_text.insert(customtkinter.END, "Process Canceled.\n")
            self.model_path_button.configure(state="normal")
            self.model_path_entry.configure(state="normal")
            self.images_folder_button.configure(state="normal")
            self.images_folder_entry.configure(state="normal")
            self.output_directory_button.configure(state="normal")
            self.output_directory_entry.configure(state="normal")
            self.sahi_checkbox.configure(state="normal")
            self.classes_entry.configure(state="normal")
            self.classes_label.configure(state="normal")
            self.slice_size_label.configure(state="normal")
            self.slice_size_entry.configure(state="normal")
            self.device_label.configure(state="normal")
            self.device_radiobutton_cpu.configure(state="normal")
            self.device_radiobutton_gpu.configure(state="normal")
            self.output_format_option_menu.configure(state="normal")
            self.conf_slider.configure(state="normal")
            self.val_slider.configure(state="normal")

    def on_worker_finished(self):
        self.worker_thread = None
        self.progress_bar.set(100)

        self.run_button.configure(state="enabled")
        self.cancel_button.configure(state="disabled")
        self.log_text.configure(state="normal")
        self.log_text.insert(customtkinter.END, "Process Finished.\n")
        self.model_path_button.configure(state="normal")
        self.model_path_entry.configure(state="normal")
        self.images_folder_button.configure(state="normal")
        self.images_folder_entry.configure(state="normal")
        self.output_directory_button.configure(state="normal")
        self.output_directory_entry.configure(state="normal")
        self.sahi_checkbox.configure(state="normal")
        self.classes_entry.configure(state="normal")
        self.classes_label.configure(state="normal")
        self.slice_size_label.configure(state="normal")
        self.slice_size_entry.configure(state="normal")
        self.device_label.configure(state="normal")
        self.device_radiobutton_cpu.configure(state="normal")
        self.device_radiobutton_gpu.configure(state="normal")
        self.output_format_option_menu.configure(state="normal")
        self.conf_slider.configure(state="normal")
        self.val_slider.configure(state="normal")

        output_directory = self.output_directory_entry.get()
        train_dir = os.path.join(output_directory, "train")
        val_dir = os.path.join(output_directory, "val")

        try:
            self.load_photos(train_dir, "train")
            self.load_photos(val_dir, "val")
        except Exception as e:
            self.show_error_message(str(e))

    def update_log(self, log_message):
        self.log_text.configure(state="normal")
        self.log_text.insert(customtkinter.END, log_message + "\n")
        self.log_text.see(customtkinter.END)

    def show_error_message(self, message):
        messagebox.showerror("Error", message)

    def update_progress(self, progress_value):
        self.progress_bar.set(progress_value / 100)       
     