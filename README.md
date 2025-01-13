# Capstone Project 1

# Satellite Image Classification Using Neural Networks

The objective of this project is to develop a machine learning model that classifies satellite images into four distinct categories using deep learning, specifically convolutional neural networks (CNNs) in TensorFlow. This model will be trained on a publicly available dataset sourced from Kaggle, containing satellite images with mixed classes from sensors and Google Map snapshots. The goal is to enable automated classification of satellite imagery, with applications in land use mapping, environmental monitoring, and urban planning.

## Dataset Overview
The dataset consists of satellite images that are categorized into four classes, which represent different geographical features or land types. The images are diverse, with variations in resolution, perspective, and sensor type, providing both a challenge and an opportunity to build a robust model. The dataset is publicly available from Kaggle:
https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
For convenience it has been uploaded on this github under dataset.

## Data Preprocessing and Augmentation
To ensure that the model can efficiently learn from the dataset, several preprocessing and augmentation techniques are applied:
- **Resizing:** Images are resized to a consistent shape (e.g., 256x256 pixels) for uniformity across the dataset.
- **Normalization:** Pixel values are scaled to a range of 0 to 1 to improve training efficiency and speed.

## Model Architecture
Given that this is an image classification problem, convolutional neural networks (CNNs) are the most suitable architecture for extracting spatial features from the images. The model is structured as follows:
- **Convolutional Layers:** These layers will capture essential features like edges, textures, and patterns in the images.
- **Pooling Layers:** Max pooling layers will be used to reduce the spatial dimensions, helping the model focus on the most important features while minimizing computation.
- **Fully Connected Layers:** These layers will make the final predictions based on the features extracted by the convolutional layers.
- **Softmax Output Layer:** The final output layer will have four nodes corresponding to the four classes, with the softmax function providing the probability distribution for each class.

## Model Training
The model is trained using a supervised learning approach with **categorical cross-entropy** as the loss function, which is ideal for multi-class classification tasks. The model's parameters are updated using the **SGD optimizer** and the **Adam optimizer**, a widely-used algorithm that adapts the learning rate to ensure faster convergence. The training process involves adjusting the model’s weights to minimize the loss function, thereby improving the accuracy of predictions.

## Model Evaluation
To evaluate the performance of the trained model, the following metric will be used:
- **Accuracy:** This metric measures the percentage of correct classifications across all images in the validation dataset.

## Conclusion
This project provides an opportunity to leverage deep learning techniques, specifically CNNs, to tackle the challenging problem of satellite image classification. By the end of the project, we aim to have a model that can accurately classify satellite images into predefined categories, offering significant potential for various applications, including environmental monitoring, urban planning, disaster response, and land use classification.



## About the files:

1. dataset folder:
This is split into 4 subfolders corresponding to the 4 image categories (cloudy, desert, green area and water) with 5631 images in total

2. Notebook: noteboook.ipynb goes through data preparation and data cleaning, EDA (Exploratory Data Analysis) as well as model identification, going through the setup with or without dropout, looking at the best combinations of learning rate and momentum using SGD, and then with adam optimizer. The working assumption is that the dataset is avalailble in data_dir but can be run anywhere with the proper modification

3. train.py: standalone training script of the selected model from the work done in the notebook (step 2 above). The model file is then saved - both as a .keras file (standard TensorFlow format) and as a .tflie file for easier deployment

4. model_2025_satellite.keras and model_2025_satellite.tflite model files from the execution of the training script

5. lambda_function.py and test.py are scripts that provide a lambda function for cloud deployment and a python test script to be run locally (input within the script is the url of an image that gets passed to the model running locally using the lambda function in a docker image)

Notes: environment is provided using pipenv thourgh Pipfile and Pipfile.lock
To install pipenv: pip install pipenv
Then install the dependencies with: pipenv intall
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run, such as: pipenv run python train.py

To use the docker setup you will need to:

1. build the docker container running the following command in the project directory which contains the Dockerfile:
docker build -t satellite_model:latest .

2. run the docker image with the following command:
docker run -p 8080:8080 satellite_model:latest

3. then you can use test.py to run a test with the following command: pipenv run python test.py