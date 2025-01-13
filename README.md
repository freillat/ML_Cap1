# Capstone Project 1

The project will aim at building a model to classify satellite images.

THe dataset is publicly available from Kaggle:
https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
For convenience it has been uploaded on this github under dataset.

This is a project built in Python and using Neural Networks in Tensorflow.

About the files:

1. dataset folder:
This is split into 4 subfolders corresponding to the 4 image categories (cloudy, desert, green area and water) with 5631 images in total

2. Notebook: noteboook.ipynb goes through data preparation and data cleaning, EDA (Exploratory Data Analysis) as well as model identification, going through the setup with or without dropout, looking at the best combinations of learning rate and momentum using SGD, and then with adam optimizer. The working assumption is that the dataset is avalailble in data_dir but can be run anywhere with the proper modification

3. train.py: standalone training script of the selected model from the work done in the notebook (step 2 above). The model file is then saved - both as a .keras file (standard TensorFlow format) and as a .tflie file for easier deployment

4. model_2025_satellite.keras and model_2025_satellite.tflite model files from the execution of the training script

Notes: environment is provided using pipenv thourgh Pipfile and Pipfile.lock
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run, such as: pipenv run python train.py