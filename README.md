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
  
## Dataset Description
  The dataset used in this task is called "FER-2013 (Facial Expression Recognition 2013) " [1] that's contains 35,887 centered 48 × 48-pixel grayscale face images labeled with one of seven emotions clesses(Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples. This dataset is  widely used for developing and evaluating models in facial emotion recognition.
## Image Preprocessing & Augmentation

## Data Visualization
- Class Distribution
  ![image](https://github.com/user-attachments/assets/a14b4269-4739-453f-bff2-a56d1f90ad5e)

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

## How to Run

## References
