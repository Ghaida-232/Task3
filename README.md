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


  
## 3. Data Visualization
- Class Distribution
  ![image](https://github.com/user-attachments/assets/a14b4269-4739-453f-bff2-a56d1f90ad5e)

  
Even after spliting the data in (70/15/15) the dataset remains highly imbalanced, the “happy” class has over 5,500 training images, while “disgust” has only around 300 and “surprise” about 2,500. This heavy skew means the model will see far more happy faces during training and consequently predict it more often, we handle later using class weights (more correction and training whould lead to better results)

- Class Distribution of train dataset (pie chart)

![image](https://github.com/user-attachments/assets/1d4fee87-0261-4856-8a7f-8ecde47c48d1)

Here we can see a clear view of the train dataset class distribution, the top three classes (Happy 25% + Neutral 17.3% + Sad 17%) together are nearly 60% of the data ,“Fear” is 14.3% .
while “Disgust” is only 1.5% and “Surprise” just 11.2%.

- Class Distribution of validation and test datasets (pie chart)

![image](https://github.com/user-attachments/assets/4a29c4c7-d153-49c7-bbe3-3d8a89dda394)

![image](https://github.com/user-attachments/assets/1c582b77-293f-41c4-965c-5c1868fea379)


And for the validation and test datasets, their class distribution are identical since they have the same split percentage (15%), the top three classes (Happy 25% + Neutral 17.3% + Sad 16.9%) together are nearly 60% of the data, “Fear” is 14.3%, “Angry” is 13.8%, while “Disgust” is only 1.5% and “Surprise” just 11.2%. Which is much similar to the train dataset class distribution.

All these visualizations reveals imbalances in the data samples which gives us a better view of the possible solutions such as applying class weights to produce more balanced training process that whould improve the model performence.

![image](https://github.com/user-attachments/assets/817e66fb-2fdc-4d61-97c0-f5c98668ada5)

This how we applied the class weights to enhance the classes' balance 

- Sample Images (Original vs. Augmented)

![image](https://github.com/user-attachments/assets/9ab1aa42-552e-4058-a9e3-fbf07fc25286)

Here is an examples of original and augmented images side by side for three emotion classes fear, happy, and surprise. Each row shows the original 48×48 grayscale face followed by four augmented variants, each applying small rotations (±15°), zoom shifts (±10%), shears (±20%), horizontal flips, and brightness adjustments (70–130%). These transformations simulate changes in pose, scale, slant, orientation, and lighting while preserving the core expression, helping the model learn to recognize emotions under varied real-world conditions.

## 4. CNN Model Architecture


![image](https://github.com/user-attachments/assets/bca1a5b7-3d5b-4cd8-a902-d5bec1717d3f)

Our model is a straightforward convolutional neural network (CNN):
- Three “conv” blocks: each block applies a set of filters (first 32 then 64 then 128) to the 48×48 grayscale image, normalizes and activates the results then downsamples via 2×2 pooling.
- Flatten layer: turns the final feature maps into a single long vector.
- Fully connected layer: 128 neurons with dropout (50%) to learn combinations of those features while reducing overfitting.
- Output layer: 7 neurons with softmax activation that give the probability of each emotion class.

This structure lets the network learn low-level patterns (edges, textures) in early layers, build up to high-level features (facial expressions) and then decide which of the seven emotions is most likely.


## 5. Training Strategy

![image](https://github.com/user-attachments/assets/4573149f-79d7-40cb-9bc6-5bc6fa3cb147)


Optimizer / Loss:
- **Adam optimizer**: a strong default that helps the network learn efficiently and reliably by adapting each weight’s step size based on past gradients.
- **Loss function**: categorical cross-entropy, the standard choice for multi-class classification, measures how well the predicted probability distribution matches the true labels.

  
- Callbacks (EarlyStopping, ModelCheckpoint, etc.):
- **EarlyStopping**: stop training when validation loss stops improving, preventing wasted epochs and overfitting.
- **ModelCheckpoint**: saves the model file (`emotion_model.h5`) each time validation loss reaches a new low, ensuring we keep the best version.
- **ReduceLROnPlateau**: lowers the learning rate when validation loss doesn’t improve, so the model can continue learning.

- 
## 6. Evaluation Metrics

- Classification Report and accuracy / F1-Score
  
![image](https://github.com/user-attachments/assets/fecb6882-0672-4a16-befd-67fb49517eca)


To evaluate our facial-emotion classifier, we looked at **Accuracy**, **Precision**, **Recall** and **F1-score**.  
- **Test Accuracy:** 45.4%, meaning the model correctly labels about 4.5 out of 10 faces (can be improve).  
- **Macro F1-Score:** 0.388, reflecting a balanced view of performance across all seven emotion classes, class wights helps get better F1-Score but still there are imblanace among the classes that affect the model accuracy and its all over performence.

From the detailed classification report:  
- **Best performance** on **Happy** (F1 = 0.769) and **Surprise** (F1 = 0.681), thanks to plenty amount of training examples and distinctive expressions.  
- **Moderate performance** on **Sad** (F1 = 0.337), **Angry** (F1 = 0.249), and **Neutral** (F1 = 0.285).  
- **Worst performance** on **Fear** (F1 = 0.148) and **Disgust** (F1 = 0.250), likely due to confusion in visual expressions and far fewer samples.

**Macro-average metrics** (equal weight per class) highlight minority-class struggles:  
- Precision = 0.396  
- Recall    = 0.418  
- F1-Score  = 0.388  

These insights show the model handles majority emotions well but needs targeted improvements—such as stronger augmentation or class-weighting—to boost recognition of under-represented expressions like fear and disgust.


- Confusion Matrix

![image](https://github.com/user-attachments/assets/0c227855-8400-4732-8c57-f42298dea0c1)


Confusion Matrix Insights:

- **Happy** faces are easiest: the model got about **82%** of them right.  
- **Surprise** is next, with **70%** correct, though some surprised faces still look like fear or sadness.  
- **Fear** is the most challenging: only **10%** are labeled correctly, most were mistaken for sad or neutral.  
- **Disgust** was right **38%** of the time, often confused with anger or sadness.  
- **Angry** faces were recognized **21%** of the time, with many mixed up as neutral or sad.  
- **Neutral** and **Sad** fall in between and often get confused with each other.

Clear and common emotions (happy, surprise) are handled well but rare ones (fear, disgust, angry) still need extra examples or special training to improve.  


## 7. Prediction Samples

![image](https://github.com/user-attachments/assets/64b9b119-fdeb-41ef-8413-f1efbc32d339)

Looking at the 10 examples above:

- **Correct (green):**  
  - Clear “happy” and “sad” faces are almost always may due to right—distinct mouth shapes.
  - surprise was correct when the face's expression was tend to be positve so it didn't confuses with fear

- **Mistakes (red):**  
  - **Fear → Happy:** lips get read as happiness.  
  - **Surprise → Fear:** both show wide eyes, so the model confuses those.  
  - **Fear → disgust** or **Angry → disgust:** furrowed brows sometimes look like a poutrevulsion.
  - **Happy → Sad**: the eys shape may get model confuse also the face shown from the side


## 8. Project Structure

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

## 9. How to Run

1.Install dependencies (requirements include TensorFlow, scikit-learn, matplotlib, seaborn, etc.)
pip install -r requirements.txt

2.Prepare the data
Download the FER-2013 dataset (archive.zip) from Kaggle. Save that file into drive (if using Colab) or into a local data/ folder., then unzip it:
unzip data/archive.zip -d data/fer2013_raw

3.Split into train/val/test then moves and balances images into emotion subfolders.(in colab)
Locally: python prepare_data.py --input_dir data/fer2013_raw --output_dir data --train_frac 0.70 --val_frac 0.15 --test_frac 0.15

4.Train the model, open and run all cells in Training_task3.ipynb.(in colab)
Locally: python train_emotion_model.py --data_dir data --batch_size 64 --epochs 30

5.Evaluate & visualize
python evaluate_model.py --model_path emotion_model.h5 --data_dir data

## 10. References

**The dataset used in this task:** https://www.kaggle.com/datasets/msambare/fer2013 [1].

**Data augmentation concept:** https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/ [2].

