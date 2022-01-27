# COVID19_Face_Mask_Detection

## 1. Data
Training mask: 300 images
Training non mask: 300 images

Validation mask: 153 images
Validation non mask: 153 images

Testing mask: 50 images
Testing non mask: 50 images
* **Source:** Kaggle

## 2. Task
Use this data to predict whether a person is wearing a mask. Afterwards, the trained model is loaded to predict real-time object using OpenCV.

## 3. Technique
* Build a ConvNet from scratch.
* Address overfitting using Dropout
* Using transfer learning with MobileNetV2 for the better result.
* Deploy the model to predict real-time frame in OpenCV
