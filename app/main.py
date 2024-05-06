import customtkinter
from .main_window import MainWindow


def launch_GUI ():
    customtkinter.set_appearance_mode("System")
    customtkinter.set_default_color_theme("dark-blue")
    app = customtkinter.CTk()
    MainWindow(app)
    app.mainloop()


if __name__ == "__main__":
    launch_GUI ()