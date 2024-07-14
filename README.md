# -Spam_Mail_Prediction_using_Machine_Learning
 Spam Mail Prediction using Machine Learning
This project focuses on building a machine learning model to predict spam emails. The model utilizes Natural Language Processing (NLP) techniques and Logistic Regression to classify emails as spam or ham (non-spam).

**Table of Contents**

Introduction

Dataset

Installation

Project Structure

Usage

Results

Contributing

License

**Introduction**

Spam emails are a significant issue in the digital world, causing inconvenience and potential security threats. This project aims to develop a predictive system to classify emails as spam or ham using a Logistic Regression model. The project involves data preprocessing, feature extraction using TF-IDF Vectorizer, model training, evaluation, and building a predictive system.

**Dataset**

The dataset used in this project is a CSV file containing email messages labeled as spam or ham. The dataset is loaded into a pandas DataFrame for preprocessing.

**Installation**
To run this project, ensure you have the following dependencies installed:

Python 3.x

numpy

pandas

scikit-learn

You can install the required packages using pip:


bash

Copy code

pip install numpy pandas scikit-learn

Project Structure

Spam Mail Prediction using Machine Learning.ipynb: Jupyter notebook containing the complete project code.

mail_data.csv: CSV file containing the email dataset.

**Usage**
Clone the repository or download the project files.
Ensure you have installed the required dependencies.
Open the Jupyter notebook Project 17. Spam Mail Prediction using Machine Learning.ipynb.
Run the notebook cells sequentially to execute the code.

**Steps**
Importing the Dependencies:

python

Copy code

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

Data Collection & Pre-Processing:

Load the data from the CSV file.

Replace null values with an empty string.

Label encoding: Spam (0) and Ham (1).

Splitting the Data:

Split the dataset into training and testing sets (80% training, 20% testing).

Feature Extraction:

Use TfidfVectorizer to convert text data into feature vectors.
Training the Model:

Train a Logistic Regression model using the training data.
Evaluating the Model:

Evaluate the model's accuracy on training and testing data.
Building a Predictive System:

Use the trained model to predict whether an input email is spam or ham.
Results
The Logistic Regression model achieved the following accuracies:

**Training Data Accuracy:** accuracy_on_training_data

**Testing Data Accuracy:** accuracy_on_test_data
Contributing
Contributions are welcome! If you have any suggestions or improvements, please submit a pull request or open an issue.

License
This project is licensed under the MIT License.
