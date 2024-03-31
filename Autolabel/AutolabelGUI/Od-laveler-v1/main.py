import sys
from PyQt6.QtWidgets import QApplication
from main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

# Project_folder/
# │
# ├── main.py
# ├── main_window.py
# ├── thread_worker.py
# ├── detection_utils.py
# └── write_utils.py