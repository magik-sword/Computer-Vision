![](UTA-DataScience-Logo.png)

# Mammogram Tumor Classification

This repository contains files that attempt to create a convolutional neural network (CNN) model that can effectively determine 
whether a breast tumor on a mammogram in the ["Breast Cancer"](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection/data) Kaggle dataset is malignant or benign.

## Overview
Breast cancer is one of the most commonly diagnosed cancers across the world and the most commonly diagnosed among women. Mammograms are a standard way of detecting early signs of breast cancer, but factors such as breast tissue density, machine artifacts, and human error often lead to malignant tumors going unseen. This project aims to minimize the amount of malignant tumors that go undiagnosed by training three different transfer models on the dataset and comparing their performances against eachother. The final model was a pre-trained DenseNet121 model that achieved a AUC score of 0.58 and a recall of 0.54.

## Summary of Work Done

### Data

* **Type:** Images (PNG/JPEG)
  * Input: 3,383 breast tumor images divided into three directories
  * Output: Benign (0) or Malignant (1)
* **Size:**
  * test: 336 images, 0: 208, 1: 128 
  * train: 2372 images, 0: 1,569, 1: 803 
  * valid: 675 images, 0: 448 images, 1: 227
* **Instances:**
  * Training: Entire train directory, 70% of images
  * Testing: Entire test directory (not used until final evaluation), 10% of images
  * Validation: Entire valid directory (used to evaluate models), 20% of images

#### Preprocessing / Clean Up

* **Pixel Normalization:** The pixel values were normalized to be within the range of [0,1].
* **Image Resizing:** Images were resized from 640x640 to 224x224.
* **Grayscale:** The images were converted from RGB to grayscale.
* **Data Augmentation:** Random rotations, zooming in and out, and horizontal flips were applied to the training set.
  
#### Data Visualization

![image](https://github.com/user-attachments/assets/7b4d3ad5-b5b9-472c-9c0d-c0432eec0c7b)

Sample batch of pre-augmentation mammograms

![image](https://github.com/user-attachments/assets/f918602e-8f1d-44d5-8a15-4fa852ef8075)

Sample batch of augmented mammograms (rotated, horizontally flipped, zoomed in/out)

#### Problem Formulation
  * **Input:** 224x224 grayscale images of mammogram images 
  * **Output:** Barplot comparing the predicted vs actual labels and ROC curve
  * **Models:**
    * MobileNetV2 (Baseline) 
    * MobilNetV2 (Augmented)
    * DenseNet121 (Augmented)
    * EfficientNetB0 (Augmented)
  * **Hyperparameters:**
     * epochs = 10
     * batch_size = 10
     * class_weight = {0: 0.7558954748247291, 1: 1.4769613947696139}
     * EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
     * learning_rate = 0.001
  * **Optimizer:** Adam(learning_rate=0.001)
  * **Loss Function:** "binary_crossentropy"
  * **Regularizer:** regularizers.l2(0.001)
    
### Training

After the initial baseline model is trained and created, all three of the sets are normalized, resized, and converted to grayscale. The data augmentations are only applied to the training set of images. Several safeguards are added to the augmented models to prevent overfitting on this relatively small and imbalanced dataset: EarlyStopping, dropout layers, L2 Regularization, and class weights. Unfortunately, overfitting still occured to some extent in all of the models. The models were compared using training vs validation recall/loss graphs and ROC-AUC curves. 

### Performance Comparison

![image](https://github.com/user-attachments/assets/ec8b755b-695a-4132-a80c-ffb8a7938aca)

Highly unstable and performs poorly. Expected given this is just a baseline.

![image](https://github.com/user-attachments/assets/006b767b-da15-43b3-a9e6-42613b3d569d)

Although the loss is improving, the massive decrease in validation recall and instability of the training recall suggests the model is struggling to generalize.

![image](https://github.com/user-attachments/assets/f54cc059-3b29-4025-bb09-ac0268537195)

This model overfit to an incredible extent. Although I'm still unsure why EfficientNetB0 specifically shot up all the way to 1.0 on validation recall except for epoch 3. 

![image](https://github.com/user-attachments/assets/c0923393-4972-4a00-b7cb-d7bb08b448d2)

Although it stopped early due to overfitting, the DenseNet121 model seemed to generalize the best. The validation recall was higher than the rest even if it was moderately unstable, and had a decreasing validation loss for the most part. 

![image](https://github.com/user-attachments/assets/cc704ecc-1b35-42a0-9e42-456a2e8d5e55)

As shown above, even the best performing models still struggled to generalize, only being slightly more accurate than a random guessing baseline. I highly suspect this is due to overfitting.

#### Performance Evaluation

![image](https://github.com/user-attachments/assets/04f3e11b-7ebf-4946-bebf-72260f3604d7)

![image](https://github.com/user-attachments/assets/ca0f3955-31ba-4efd-8924-d2a265a756da)

The evaluations above are based on the performance of DenseNet121 on the final test set. The model actually overpredicts the amount of malignant cases, leading to it missing a moderate amount of benign ones. This likely explains the slightly lower AUC score on the test set compared to the validation set. If this overfitting problem can be overcome, the DenseNet121 model likely could perfom excellently on the data. 

### Conclusions

Although transfer models such as DenseNet121 can perform well, extra care must be taken when working with an imbalanced and small dataset. A more effective strategy for addressing this problem likely lies in segmentation or cropping the images. 

### Future Work

The biggest challenge for this project was the class imabalnce in the dataset. Most of my time training was spent trying different ways to combat overfitting due to this imbalance. Most of the suggestions below are related to this problem I kept running into.

* **Heavier Models:** Due to being limited to my laptop for this project, I was limited to keeping the augmentations and transfer models relatively light so as to make training time manageable and avoidcrashing my computer. More complex models such as VGG16/19 or ResNet152 likely could give better results.
* **Segmentation:** A significant portion of each image is blank space that provides no valuable information to the models. Isolating the breast tissue through edge detection likely will produce greater results than this project. 
* **Creating New Images:** Another idea to explore is adding new images to the minority class. This could be done by copying existing malignant tumor pictures and applying small augmentations to them so they aren't exactly identical to existing images. 
* **Fine-Tuning:** Different combinations of optimizers, loss functions, and hyperparameters can and should be explored in the future to see if these can affect performance on the models tested here. 
  
### How to Reproduce Results

* Download and unzip the zip file to access the image directories
* Run the Baseline_Model notebook
* The next three notebooks can be run in any order, they serve as a way to control file size and experiment with different augmentations/preprocessing. The function that creates these models and the preprocessing pipeline can easily be edited to try out different models or strategies.
    *   MobileNetV2_Augmented.ipynb
    *   EfficientNetB0_Augmented.ipynb
    *   DenseNet121_Augmented.ipynb
 * Run the Model_Comparisons notebook to gather all the graphs and curves in one place so you can make a decision on final model deployment
 * After selecting a final model, load it in the Final_Evaluation notebook and see the results on the test set

### Overview of Files in Repository

* **Baseline_Model.ipynb:** Trains a MobileNetV2 model on the unprocessed images for a baseline and saves its history.
* **MobileNetV2_Augmented.ipynb:** Trains and saves a MobileNetV2 model/history on the augmented and preprocessed training images.
* **EfficientNetB0_Augmented.ipynb:** Trains and saves an EfficientNetB0 model/history on the augmented and preprocessed training images.
* **DenseNet121_Augmented.ipynb:** Trains and saves a DenseNet121_Augmented model/history on the augmented and preprocessed images.
* **Model_Comparisons.ipynb:** Loads the previous models and histories, compares the metrics by displaying training vs validation graphs and ROC-AUC curves.
* **Final_Evaluation.ipynb:** Loads the DenseNet121 model and runs predictions on the unseen test set. Contains a barplot, ROC-AUC curve, and final recall score.
* **breast-cancer-detection.zip:** The original zip file from Kaggle containing all of the images used for this project. 
  
### Required Libraries

* tensorflow
* keras
* matplotlib
* numpy
* scikit-learn
* pickle

#### Citations
Oza, Parita et al. “Image Augmentation Techniques for Mammogram Analysis.” Journal of imaging vol. 8,5 141. 20 May. 2022, [doi:10.3390/jimaging8050141](https://pmc.ncbi.nlm.nih.gov/articles/PMC9147240/#sec2-jimaging-08-00141)
