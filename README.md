# Autolabel

![GUI](https://github.com/ZiadAlgrafi/Autolabel-yolov8/assets/117011801/e7ba1f79-1808-4927-88d9-7e65c225d3a8)


Autolabel is a project designed to simplify the auto-labeling of classes in images using the YOLOv8 model. The provided GUI allows users to choose between a custom YOLOv8 model or the default `yolov8n.pt` model.

## Autolabel.py

`Autolabel.py` is the main script of this project, featuring a graphical user interface (GUI) for easy interaction. The script leverages the Ultralytics library for object detection.

### Usage

1. Download the repository:

 https://github.com/ZiadAlgrafi/Autolabel-yolov8.git
 
 
### Run from Command Line

To run the Autolabel script from the command line:

2. . Navigate to the project directory and install your preferred requirements MANUALLY from the command prompt CMD. Note choose your desired file and minimum or recommened requirements:

   
    cd path\to\Autolabel\Requirements
    

3. Activate the Python virtual environment if already done:

   
    . venv/bin/activate      # On Linux
    .\myenv\Scripts\activate # On Windows
  


4. Navigate to the Autolabel script and Run the Autolabel script using cmd:

    cd /home/user/Documents/Autolabel/AutolabelGUI
    python Autolabel.py 
    


### GUI Features

Select YOLO MODEL>> if you have custom model and you prefer your images to be labelled as how you have configured your model enter the path to your custom model. If you dont have custom model you can choose the default yolov8s.pt model from ultralytics library i have included it in requirements folder

Select Images Folder>> select the folder that contains your desired images to be labelled. Note choose the root path for the folder, Also based on the names of your images your labels will correspond to that.

Select Output Directory>> select where the labels results will be outputed. 

Select Device>> based on the your device is it GPU or CPU. If GPU choose 0. If CPU choose cpu

Run Code>> after adding paths and selecting device you can run the program using Run Code button. Wait few seconds based on your device and check the Log Run.

Log Run: will updates the logs of the detction of the model in XYWH values after it finish it will output Results saved in txt files (your directory)




