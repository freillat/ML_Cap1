# Capstone Project 1

The project will aim at building a model to classify satellite images.

THe dataset is publicly available from Kaggle:
https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
For convenience it has been uploaded on this github under dataset.

This is a project built in Python and using Neural Networks in Tensorflow.

About the files:

1. dataset folder:
This is split into 4 subfolders corresponding to the 4 image categories (cloudy, desert, green area and water) with 5631 images in total

2. Notebook: noteboook.ipynb goes through data preparation and data cleaning, EDA (Exploratory Data Analysis) as well as model fine-tuning (trying to identify the best combination of learning rate and momentum using SGD) and selection by comparing SGD to Adam. It assumes that it is being run with the dataset in data_dir but can be run anywhere assuming you point to the dataset by modifying wherever you have saved it

3. Train.py: standalone training of the selected model from step 2 above

