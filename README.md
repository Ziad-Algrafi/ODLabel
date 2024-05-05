![output](https://github.com/Ziad-Algrafi/ODLabel/assets/117011801/9fb27b28-eab1-4edb-9c1d-0535c4e0e99a)

# ODLabel

ODLabel (Open Dictionary Labeler) is a powerful tool for zero-shot object detection, labeling and visualization. It provides an intuitive graphical user interface for labeling objects in images using the YOLO-World model and supports various output formats such as YOLO, COCO, CSV, and XML.

## Features

- Select a YOLO-World model for object detection
- Choose an images folder for labeling
- Specify output directory for annotated data
- Define object categories for detection
- Use Slicing Adaptive Inference (SAHI) for improved detection for small objects
- Select device type (CPU or GPU) for inference
- Customize train/validation split ratio
- Adjust confidence level and non-maximum suppression (IoU) threshold
- Visualize input image statistics and output detection results

You can install ODLabel from [PyPI](https://pypi.org/project/odlabel/) using pip:

```bash
pip install odlabel

```

## Usage

To launch the ODLabel application, run the following command:

```bash
odlabel
```

1. Select a YOLO-World model file (.pt) for object detection.
2. Choose the folder containing the images you want to label.
3. Specify the output directory where the labeled data will be saved.
4. Enter the object categories you want to detect, separated by commas.
5. Configure additional options such as SAHI, device type, output format, train/validation split, confidence level, and NMS threshold.
6. Click the "Start" button to begin the labeling process.
7. Monitor the progress and view the detection results in the application.

## Model

| Model Type      | mAP  | mAP50 | mAP75 | Model                                                                                         |
| --------------- | ---- | ----- | ----- | --------------------------------------------------------------------------------------------- |
| yolov8s-world   | 37.4 | 52.0  | 40.6  | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt)   |
| yolov8s-worldv2 | 37.7 | 52.2  | 41.0  | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt) |
| yolov8m-world   | 42.0 | 57.0  | 45.6  | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-world.pt)   |
| yolov8m-worldv2 | 43.0 | 58.4  | 46.8  | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-worldv2.pt) |
| yolov8l-world   | 45.7 | 61.3  | 49.8  | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-world.pt)   |
| yolov8l-worldv2 | 45.8 | 61.3  | 49.8  | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-worldv2.pt) |
| yolov8x-world   | 47.0 | 63.0  | 51.2  | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-world.pt)   |
| yolov8x-worldv2 | 47.1 | 62.8  | 51.4  | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-worldv2.pt) |

## Update

You can upgrade ODLabel using pip:

```bash
pip install --upgrade odlabel

```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the [GitHub repository](https://github.com/Ziad-Algrafi/odlabel).

## Acknowledgements

ODLabel is built using the following open-source libraries:

- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Matplotlib](https://matplotlib.org)
- [OpenCV](https://opencv.org)
- [PyTorch](https://pytorch.org)

## Contact

For any questions or inquiries, please contact the maintainer at - ZiadAlgrafi@gmail.com.
