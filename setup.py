from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="67-finder",
    version="0.1.0",
    author="Bogdan11212",
    description="Real-time gesture and number 67 detector using YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bogdan11212/67-fineder",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0",
        "torch>=2.0",
        "torchvision>=0.15",
        "opencv-python>=4.8",
        "numpy>=1.24",
        "huggingface-hub>=0.19",
    ],
)
