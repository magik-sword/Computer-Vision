![](UTA-DataScience-Logo.png)

# Mammogram Tumor Classification

This repository contains files that attempt to create a convolutional neural network (CNN) model that can effectively determine 
whether a breast tumor on a mammogram in the ["Breast Cancer"](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection/data) Kaggle dataset is malignant or benign.

## Overview
Breast cancer is one of the most commonly diagnosed cancers across the world and the most commonly diagnosed among women. Mammograms are a standard way of detecting early signs of breast cancer, but factors 
such as breast tissue density, machine artifacts, and human error often lead to malignant tumors going unseen. This project trained three different transfer models on the dataset and compared their performances
against eachother. 

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
  * Testing: Test directory (not used until final evaluation), 10% of images
  * Validation: Entire valid directory (used to evaluate models), 20% of images

#### Preprocessing / Clean Up

* **Pixel Normalization:** The pixel values were normalized to be within the range of [0,1].
* **Image Resizing:** Images were resized from 640x640 to 224x224.
* **Grayscale:** The images were converted from RGB to grayscale.
* **Data Augmentation:** Random rotations, zooming in and out, and horizontal flips were applied to the training set.
* 
#### Data Visualization

As noted in the notebooks, there is a significant class imbalance in the dataset:
* 77.8% Normal
* 13.9% Suspect
* 8.3% Pathological

The main focus of the data visualizations is to identify features that effectively separate the minority classes from the majority class.

![image](https://github.com/user-attachments/assets/2e9cf5a2-52bd-410c-8e9a-2d50a3b0c0e3)
![image](https://github.com/user-attachments/assets/36408af2-8bf7-4996-b237-ee18eb67721b)

Histogram_tendency represents the skewness of the fetal heart rate distribution during the CTG, while histogram_number_of_zeros shows the number of zero count bins in the histogram. The table shows that a left skew (-1.0) is more common in pathological cases: 27% of pathological cases have a left skew compared to less than 6% of normal or suspect cases. The graph shows that the majority of pathological class has no empty bins in their CTG histograms. 

![image](https://github.com/user-attachments/assets/a7d1a6c3-dbcd-4551-a3f1-9bce11d5a401)
![image](https://github.com/user-attachments/assets/77e22437-3dd9-44a8-9156-3045c615f54f)

Histogram_variance captures the variance of the fetal heart rate distribution during the CTG, while histogram_number_of_peaks indicates the number of distinct peaks. The data shows that almost all instances in the suspect class cluster near zero in histogram_variance. A large portion of the suspect class falls within the 0–5 range in histogram_number_of_peaks.

#### Problem Formulation
  * **Input:** 224x224 grayscale images of mammogram images 
  * **Output:** Barplot comparing the predicted vs actual labels
  * **Models:**
    * MobileNetV2 (Baseline) 
    * MobilNetV2 (Augmented)
    * DenseNet121 (Augmented)
    * EfficientNetB0 (Augmented)
  * **Hyperparameters:**
     * epochs = 10
     * batch_size = 10
     * class_weight = {0: 0.7558954748247291, 1: 1.4769613947696139}
     * optimizer = optimizers.Adam(learning_rate=0.001)

### Training

After the initial baseline model, the target column was label encoded for the XGBoost models. Multiple XGBoost models were trained using default hyperparameters, SMOTE, or Random Oversampling and then compared using macro-averaged F1-score and confusion matrices. Grid Search was then used on the best performing model. The main issue during training was that the suspect class had noticeably lower F1-scores and the confusion matrices showed that it was mislabeling instances more often than the other two classes. This is likely due to the fact that the suspect class often overlapped with the other classes in the data and there were very few features that effectively separated it from the other two. 

### Performance Comparison

![image](https://github.com/user-attachments/assets/83d1ff5a-2b23-4a6a-89e4-66347a0698b5)

One of the main metrics I used to evaluate the models was macro-averaged F1-score. It gives equal weight to each class, preventing the normal class from skewing the score and giving the false impression of better model performance. As seen in the table, the sampling techniques had little effect on model performance. The best performing model didn't use any sampling techniques and only had hyperparameter tuning applied.

![image](https://github.com/user-attachments/assets/5c4148fa-0d05-4245-be8c-e79ab1623428)

The other main metric I used was a confusion matrix to see a visual representation of how each model classified each case. The confusion matrix above is the one for the optimized XGBoost model and shows that the model is correctly classifying each instance in the test set most of the time, except for a slightly worse performance in the suspect class. 

### Conclusions

An XGBoost model with optimized hyperparameters was the most effective at determining fetal health status. 

### Future Work

* **Heavier Models:** Due to being limited to my laptop for this project, I was limited to keeping the augmentations and transfer models relatively light so as to make training time manageable and avoid
crashing my computer. More complex models such as VGG16/19 or ResNet152 likely could give better results.
* **Segmentation:** Following along the idea of removing the black regions that have no info, segmentation could be used to ensure only the breast tissue is being looked at by the models.
* **Creating New Images:** The class imbalance is the most likely reason for overfitting, one idea to be explored is artificially increasing the minority class by copying and adding new images to it
* **Hyperparameter Tuning:** Various other parameters for the models could be tried such as learning rate or choice of optimizer. 

### How to Reproduce Results
After downloading the CSV file, run the EDA and Preprocessing notebook to produce a preprocessed CSV file to use for training. Then, running the Training and Evaluations notebook will train all of the models and display the evaluations for each. 

### Overview of Files in Repository

* EDA and Preprocessing.ipynb: Contains data visualizations and summary info about the dataset. Produces a cleaned CSV file for training when run.
* Training and Evaluations.ipynb: Trains and evaluates several models using the cleaned dataset from the first notebook. 
* fetal_health.csv: The original dataset as provided by Kaggle.
  
### Required Libraries

* pandas
* numpy
* matplotlib
* ipython
* tabulate
* scikit-learn
* xgboost
* imbalanced-learn

#### Performance Evaluation

Evaluation functions generate classification reports and confusion matrices for each model in the Training and Evaluations notebook. The table comparing the F1-scores of each model is also at the end of this notebook.

#### Citations
Oza, Parita et al. “Image Augmentation Techniques for Mammogram Analysis.” Journal of imaging vol. 8,5 141. 20 May. 2022, [doi:10.3390/jimaging8050141](https://pmc.ncbi.nlm.nih.gov/articles/PMC9147240/#sec2-jimaging-08-00141)
