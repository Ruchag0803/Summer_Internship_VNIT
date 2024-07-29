# Fruit Quality Detection Project

## Overview

This project, completed during my internship at VNIT Nagpur, focuses on detecting the quality of fruits using machine learning models. The main objective is to classify fruits based on their quality, ensuring that only the best fruits reach the consumers.

## Dataset

The dataset consists of 12,000 images of fruits, which are split into an 80-20 train-test ratio. Each image is labeled with the quality of the fruit, which serves as the ground truth for training and testing the models.

## Models Used

Several machine learning models were used for this project, including:

- Convolutional Neural Network (CNN)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- You Only Look Once (YOLO)
- Naive Bayes (NV)
- Logistic Regression

### Training and Testing Accuracy

The models were trained and tested on the dataset, with the following accuracy results:

| Model               | Training Accuracy | Testing Accuracy |
|---------------------|-------------------|------------------|
| CNN                 | 90.97%            | 94.6%            |
| SVM                 | 100%              | 96.58%           |
| KNN                 | 93.19%            | 88.17%           |
| ANN                 | 99.94%            | 96.42%           |
| Naive Bayes         | 87.63%            | 85.42%           |
| Logistic Regression | 100%              | 96.58%           |
| Random Forest       | 100%              | 97.75%           |

## Quality Evaluation Techniques

### Convolutional Neural Network (CNN)

CNN is a deep learning algorithm that takes in an input image, assigns importance (learnable weights and biases) to various aspects/objects in the image, and is able to differentiate one from the other.

### Support Vector Machine (SVM)

SVM is a supervised machine learning algorithm which can be used for both classification or regression challenges. It performs classification by finding the hyperplane that best divides a dataset into classes.

### K-Nearest Neighbors (KNN)

KNN is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems. It is based on the concept of finding the closest data points in the training dataset.

### You Only Look Once (YOLO)

YOLO is a state-of-the-art, real-time object detection system. It applies a single neural network to the full image, dividing the image into regions and predicting bounding boxes and probabilities for each region.

### Naive Bayes (NV)

Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set.

### Logistic Regression

Logistic Regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist.

## Results

The following images showcase the results of the models:

1. **Result of YOLO**:
   ![image](https://github.com/user-attachments/assets/07aa9358-ca4f-48c1-8198-476706847164)

2. **Result of All Models**:
  ![image](https://github.com/user-attachments/assets/118ba5ba-c6cc-4e37-90b7-683eb76173e0)

3. **Grouping the Confidence Score into Categories**:
  ![image](https://github.com/user-attachments/assets/f8400b94-887e-434a-a72d-f0b84af03a74)

## Conclusion

This project demonstrates the effectiveness of various machine learning models in detecting the quality of fruits. The YOLO model, in particular, shows promising results in real-time object detection, making it a viable option for practical applications.

## Future Work

Future improvements could include:

- Increasing the dataset size for better model training.
- Experimenting with other deep learning models.
- Optimizing model parameters for better accuracy.
- Deploying the model in a real-world application for live fruit quality detection.

## Acknowledgments

I would like to thank my mentors at VNIT Nagpur for their guidance and support throughout this project.

