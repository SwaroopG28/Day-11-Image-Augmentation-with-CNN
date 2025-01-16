# Image Augmentation with Convolutional Neural Networks (CNNs)

## Project Overview
This project demonstrates how **image augmentation** can improve the performance of a **Convolutional Neural Network (CNN)** for image classification. We use the **CIFAR-10 dataset**, which contains 60,000 32x32 color images classified into 10 categories, such as airplanes, cats, and cars.

By applying **image augmentation**, we artificially increase the size and diversity of the training dataset by introducing random transformations like rotations, flips, zooms, and shifts. This helps the model generalize better and reduces overfitting.

---

## Dataset
- **Name:** CIFAR-10
- **Description:** 60,000 32x32 color images in 10 classes with 6,000 images per class.
- **Classes:**
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck
- **Source:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## Key Features
1. **Image Augmentation:**
   - Techniques applied:
     - Random rotations.
     - Horizontal flips.
     - Random zooms and shifts.
     - Shearing transformations.
   - Helps create variations in the training data, improving generalization.

2. **CNN Architecture:**
   - Three convolutional layers with max-pooling and dropout.
   - Fully connected layer for classification.
   - Softmax activation for multi-class output.

3. **Evaluation:**
   - Training and validation accuracy/loss tracked over 10 epochs.
   - Test accuracy measured on unseen data.

---

## Project Steps

### Step 1: Load and Preprocess the Dataset
- CIFAR-10 dataset was loaded using TensorFlow.
- Normalized pixel values to the range [0, 1] to improve training performance.
- Displayed a few sample images to understand the dataset.

### Step 2: Image Augmentation
- Used `ImageDataGenerator` to apply augmentation techniques such as:
  - Rotation (up to 20 degrees).
  - Width and height shifts (up to 20%).
  - Horizontal flips.
  - Zoom (up to 15%).
- Visualized augmented images to confirm transformations.

### Step 3: Build the CNN Model
- Model architecture:
  1. Convolutional layers extract features from the images.
  2. Max-pooling layers reduce dimensionality while retaining important information.
  3. Dropout layers prevent overfitting by randomly deactivating neurons.
  4. Fully connected layers classify images into 10 categories.

### Step 4: Train the Model with Augmentation
- Model trained using augmented data for 10 epochs with a batch size of 64.
- Used `adam` optimizer and `sparse_categorical_crossentropy` loss.

### Step 5: Evaluate and Test the Model
- Evaluated model performance on the test set.
- Visualized training and validation accuracy/loss over epochs.
- Tested the model with a single image from the test set, displaying its true and predicted labels.

---

## Applications
1. **Generalization:** Helps train models that perform well on unseen data.
2. **Image Recognition:** Useful in tasks like object detection, face recognition, and medical imaging.
3. **Data Augmentation:** Essential for small datasets where collecting more data is not feasible.
