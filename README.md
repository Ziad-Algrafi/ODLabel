
![run test](https://github.com/Ziad-Algrafi/ODLabel/assets/117011801/91cb2a0f-66d3-4371-9b0b-a44b76047a53)

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

To launch the OD-Labeler application, run the following command:

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
