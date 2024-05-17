from setuptools import setup, find_packages

setup(
    name="odlabel-onnx",
    version="0.7.26.5",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "customtkinter",
        "matplotlib",
        "numpy",
        "opencv-python",
        "onnxruntime-gpu",
        "clip",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
    ],
    entry_points={
        "console_scripts": [
            "odlabel-onnx=app_onnx.main:launch_GUI",
        ],
    },
    author="Ziad-Algrafi",
    author_email="ZiadAlgrafi@gmail.com",
    description="A tool for object detection, labeling and visualization using ONNX",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ziad-Algrafi/odlabel",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    setup_requires=["setuptools>=38.6.0", "wheel"],
)