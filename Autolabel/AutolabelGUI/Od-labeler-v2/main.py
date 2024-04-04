import customtkinter

from main_window import MainWindow

if __name__ == "__main__":
    customtkinter.set_appearance_mode("System")
    customtkinter.set_default_color_theme("blue")

    app = customtkinter.CTk()
    window = MainWindow(app)
    app.mainloop()

# Project_folder/
# │
# ├── main.py
# ├── main_window.py
# ├── thread_worker.py
# ├── detection_utils.py
# └── write_utils.py