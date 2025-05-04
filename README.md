
# Brain Tumor Classification using 3D Convolutional Neural Network (CNN)

This project utilizes a 3D Convolutional Neural Network (CNN) to classify brain tumor images into three categories: "Class 0", "Class 1", and "Class 2". The dataset was sourced from Pakistan PIMS and IDC and was collected over a span of 10 months. The data was initially in DICOM format and has been significantly preprocessed and cleaned using various scripts.
For Demo: https://huggingface.co/spaces/berrygudboi/FYP 

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The goal of this project is to train a CNN model to accurately classify brain tumor images into one of three categories. This can aid in medical diagnostics and research.

## Dataset
The dataset used in this project was collected from Pakistan PIMS and IDC. It was collected over a period of 10 months and initially existed in DICOM format. The dataset consists of the following classes:
- Class 0: No tumor
- Class 1: Hemo present
- Class 2: Ischemic present

## Preprocessing
The dataset underwent extensive preprocessing, including:
- Conversion from DICOM to a usable format.
- Cropping to the brain region.
- Windowing to normalize intensity values.
- Resizing or padding to a target shape of (256, 256, 100).
- Augmentation to improve model robustness.

## Model Architecture
The model used is a 3D CNN with the following architecture:
- Pretrained 3D ResNet R3D-18.
- Custom convolutional layers with ReLU activation.
- Max pooling layers.
- Flattened output followed by fully connected layers with ReLU activation.
- Final softmax output layer for classification.

## Training
The model was trained using the following parameters:
- Loss function: LabelSmoothingCrossEntropy
- Optimizer: AdamW
- Learning rate: 1e-5
- Batch size: 7
- Number of epochs: 15
- Device: CUDA if available

## Evaluation
The model's performance was evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score

## Usage
To train the model:
1. Ensure the dataset is organized in the specified directory structure.
2. Run the training script with the necessary parameters.

To make predictions:
1. Load the trained model.
2. Preprocess the new image as required by the model.
3. Use the model to make a prediction.

## Contributing
Contributions to this project are welcome. Please open an issue or submit a pull request with your changes.

## License
This project is licensed under the MIT License.

---

