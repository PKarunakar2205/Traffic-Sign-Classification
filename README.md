# Traffic-Sign-Classification
# Traffic Sign Classification using Machine Learning

## Introduction
This project aims to classify German traffic signs using machine learning and deep learning techniques. The dataset used is the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset available on Kaggle.

## Dataset
The dataset can be found here: [GTSRB Kaggle Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data)

### Dataset Overview
The dataset consists of images of various German traffic signs categorized into different classes. It includes:
- Over 50,000 images
- 43 distinct traffic sign categories
- Different lighting conditions and perspectives

## Features
- Data preprocessing and augmentation
- CNN-based deep learning model for classification
- Model evaluation using accuracy, precision, and recall
- Real-time traffic sign prediction

## Installation
To set up the environment and required dependencies, run:

```bash
pip install numpy pandas tensorflow keras matplotlib seaborn scikit-learn opencv-python
```

## Usage
### 1. Load and Preprocess Dataset
```python
import pandas as pd
import numpy as np
import cv2
import os

# Load dataset (example for custom loading)
data_path = "../input/gtsrb-german-traffic-sign"
```

### 2. Train a CNN Model
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### 3. Train and Evaluate
```python
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

## Evaluation Metrics
- **Accuracy**: Measures overall classification performance.
- **Precision & Recall**: Determines the model's ability to distinguish different signs.
- **Confusion Matrix**: Provides insight into misclassified traffic signs.

## Future Improvements
- Use transfer learning with pre-trained models like MobileNet or ResNet.
- Deploy the model as a web or mobile application for real-time traffic sign detection.
- Improve accuracy with additional data augmentation techniques.

## License
This project is open-source and available under the MIT License.

## Contributors
  P.Karunakar


## Acknowledgments
- Kaggle and GTSRB dataset providers.
- TensorFlow and Keras community for deep learning support.

