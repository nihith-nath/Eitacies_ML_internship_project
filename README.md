# Eitacies_ML_internship_project
i am uploading and documenting all files in my summer 2024 internship at EITACIES inc


# PCI-DSS Detection System

## Overview

I developed a PCI-DSS detection system using a textual dataset, starting with data loading and exploration to identify and understand the dataset's structure. After standardizing and cleaning the text data through lowercasing, punctuation removal, tokenization, and stop words elimination, I created a new feature `contains_number` to capture numeric characters in the messages. Using TF-IDF, I transformed the text data into numerical features and combined them with the new feature for model training. I trained both Logistic Regression and Decision Tree models, evaluated their performance, and generated detailed metrics.

To enhance usability, I implemented a function to predict PCI-DSS compliance for custom input messages, preprocessing the input and extracting features before making predictions. The results, including classification reports and predictions, were saved to a JSON file. Additionally, I created a function to load the JSON data into a MongoDB collection, facilitating efficient storage and retrieval. This comprehensive approach included model deployment, tuning to reduce bias, feature engineering, and database integration to ensure a robust PCI-DSS compliance detection system.

## MongoDB Deployment

![MongoDB Deployment](PCI-DSS-TASK-1/mongo%20db%20deployment.png)
