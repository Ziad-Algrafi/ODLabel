import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
import os
import cv2

class YOLOGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Open Dictionary Auto label")

        self.model_path = tk.StringVar()
        self.images_folder = tk.StringVar()
        self.output_directory = tk.StringVar()
        self.classes = tk.StringVar(value="person, bus, car")

        self.device_var = tk.StringVar(value="cpu")

        ttk.Button(root, text="Select YOLO Model", command=self.select_model).grid(row=0, column=2, padx=10, pady=10)
        ttk.Button(root, text="Select Images Folder", command=self.select_images_folder).grid(row=1, column=2, padx=10, pady=10)
        ttk.Button(root, text="Select Output Directory", command=self.select_output_directory).grid(row=2, column=2, padx=10, pady=10)

        ttk.Label(root, text="Model Path:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.E)
        ttk.Label(root, textvariable=self.model_path).grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)

        ttk.Label(root, text="Images Folder:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.E)
        ttk.Label(root, textvariable=self.images_folder).grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)

        ttk.Label(root, text="Output Directory:").grid(row=2, column=0, padx=10, pady=10, sticky=tk.E)
        ttk.Label(root, textvariable=self.output_directory).grid(row=2, column=1, padx=10, pady=10, sticky=tk.W)

        ttk.Label(root, text="Classes:").grid(row=3, column=0, padx=10, pady=10, sticky=tk.E)
        ttk.Entry(root, textvariable=self.classes).grid(row=3, column=1, padx=10, pady=10, sticky=tk.W)

        ttk.Label(root, text="Log Run:").grid(row=4, column=0, padx=10, pady=10, sticky=tk.E)
        self.log_text = tk.Text(root, height=10, width=50, wrap=tk.WORD)
        self.log_text.grid(row=4, column=1, columnspan=2, padx=10, pady=10, sticky=tk.W)

        ttk.Label(root, text="Select Device:").grid(row=5, column=0, padx=10, pady=10)
        combobox = ttk.Combobox(root, textvariable=self.device_var, values=["cpu", "0"])
        combobox.grid(row=5, column=1, padx=10, pady=10)
        self.device_var.set("cpu")  

        self.run_button = ttk.Button(root, text="Run Code", command=self.run_code, state=tk.DISABLED)
        self.run_button.grid(row=5, column=2, columnspan=2, padx=10, pady=10)

        root.geometry("600x500")

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def select_model(self):
        path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("YOLO Model", "*.pt")])
        if path:
            self.model_path.set(path)
            self.enable_run_button()

    def select_images_folder(self):
        folder = filedialog.askdirectory(title="Select Images Folder")
        if folder:
            self.images_folder.set(folder)
            self.enable_run_button()

    def select_output_directory(self):
        folder = filedialog.askdirectory(title="Select Output Directory")
        if folder:
            self.output_directory.set(folder)
            self.enable_run_button()

    def enable_run_button(self):
        if all([self.model_path.get(), self.images_folder.get(), self.output_directory.get()]):
            self.run_button['state'] = tk.NORMAL
        else:
            self.run_button['state'] = tk.DISABLED

    def run_code(self):
        select_device(self.device_var.get())
        self.log_text.delete(1.0, tk.END)

        model = YOLO(self.model_path.get())
        model.set_classes(self.classes.get().split(', '))

        image_files = [f for f in os.listdir(self.images_folder.get()) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]

        if not image_files:
            self.log_text.insert(tk.END, "No images in the folder. Exiting.\n")

        for image_file in image_files:
            img_path = os.path.join(self.images_folder.get(), image_file)
            img = cv2.imread(img_path)

            results = model(img, conf=0.3, imgsz=640)
            self.log_text.insert(tk.END, f"Results for {image_file}:\n")

            result_txt_path = os.path.join(self.output_directory.get(), f"{os.path.splitext(image_file)[0]}.txt")

            with open(result_txt_path, 'a') as txt_file:
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        xy = box.xywhn.cpu().numpy()
                        c = box.cls.cpu().numpy()
                        class_id = int(c)
                        self.log_text.insert(tk.END, f" ({class_id}): {xy}\n")
                        self.log_text.insert(tk.END, str(class_id) + "\n")

                        for box in xy:
                            txt_file.write(f"{class_id} {box[0]} {box[1]} {box[2]} {box[3]}\n")

                        self.log_text.update_idletasks()
                        self.log_text.see(tk.END)

        self.log_text.insert(tk.END, f"Results saved in {self.output_directory.get()}\n")

    def on_close(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOGUI(root)
    root.mainloop()