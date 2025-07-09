# Facial Emotion Classification (CNN)

## Objective
- Prepare and preprocess image data for emotion classification.
- Explore dataset characteristics
- Train a CNN model to classify facial expressions into predefined emotion categories.
- Use data augmentation to improve generalization.
-  Evaluate the model’s performance using multiple metrics and visual tools.
-  Visualize predictions, highlight misclassified samples, and interpret model behavior.
-  Save the best-performing trained model and all generated visualizations.
-  Document all steps clearly
  
## 1. Dataset Description
  The dataset used in this task is called "FER-2013 (Facial Expression Recognition 2013) "[1] that's contains 35,887 centered 48 × 48-pixel grayscale face images labeled with one of seven emotions clesses(Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples. This dataset is  widely used for developing and evaluating models in facial emotion recognition.
## 2. Image Preprocessing & Augmentation

  Data augmentation is a technique to artificially create new training data from existing training data. This is done by applying domain-specific techniques to examples from the training data that create new and different training examples and involves creating transformed versions of images in the training dataset that belong to the same class as the original image. Transforms include a range of operations from the field of image manipulation, such as shifts, flips, zooms, and much more. Image data augmentation is typically only applied to the training dataset, and not to the validation or test dataset.[2]
  

**2.1 Augmentation**

  ![image](https://github.com/user-attachments/assets/f18c5add-8bfe-4dd4-8a38-76bc0ab86e50)

Here we apply the augmentation on our train data (in train folder) by the following techiques:
- Rotation (±15°): handles varying head poses
- Zoom (±10%): handles varying in camera's distance
- Shear (±20%): handles slanted faces
- Horizontal Flip: makes orientation-invariant
- Brightness (70–130%): covers different lighting in the images
- fill_mode='reflect': fills in new pixels (may caused by rotation or zoom) by reflecting the image border.


**2.2 Image Preprocessing**

![image](https://github.com/user-attachments/assets/9c90b76a-e4e0-4697-b874-bf92e46eb19d)

And for the image preprocessing we apply the following:
- Resize all images to 48×48 pixels, keep the architecture simple.
- Convert to grayscale, the same channel format (the dataset in grayscale mode).
- Normalize pixel values to [0,1] because large input scales can make gradient updates noisy or slow.

These preprocessing guarantee that the model sees consistent, well-scaled inputs, learns faster and generalizes better.


  
## Data Visualization
- Class Distribution
  ![image](https://github.com/user-attachments/assets/a14b4269-4739-453f-bff2-a56d1f90ad5e)

  
Even after spliting the data in (70/15/15) the dataset remains highly imbalanced, the “happy” class has over 5,500 training images, while “disgust” has only around 300 and “surprise” about 2,500. This heavy skew means the model will see far more happy faces during training and consequently predict it more often, we handle later using class weights (more correction and training whould lead to better results)

![image](https://github.com/user-attachments/assets/1d4fee87-0261-4856-8a7f-8ecde47c48d1)

Here we can see clear view of the train 

- Sample Images (Original vs. Augmented)

## CNN Model Architecture

## Training Strategy
- Optimizer / Loss
- Callbacks (EarlyStopping, ModelCheckpoint, etc.)

## Evaluation Metrics
- Accuracy / F1-Score
- Confusion Matrix
- Classification Report

## Prediction Samples

## Project Structure

## Project Structure

```text
/content                     # Notebook files section
├── Task3/                   # GitHub repo folder in Colab
│   ├── README.md            # This file
│   └── Training_task3.ipynb # Colab notebook
├── task3/                   # Dataset splits' folder in Colab
│   ├── all_data/            # Unzipped original dataset
│   ├── train/               # 70% training split
│   ├── val/                 # 15% validation split
│   └── test/                # 15% test split
├── emotion_model.h5         # Best‐saved Keras model (at /content/emotion_model.h5)
/drive/                      # Mounted Google Drive
│   └── MyDrive/
│       └── archive.zip      # Original FER2013 ZIP from Kaggle
└── sample_data/             # Colab’s default sample folder
```
## How to Run

## References
