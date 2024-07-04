# Eitacies_ML_internship_project
i am uploading and documenting all files in my summer 2024 internship at EITACIES inc

#Task - 1

# PCI-DSS Detection System

## Overview

I developed a PCI-DSS detection system using a textual dataset with the following steps:

- **Data Loading and Exploration**:
  - Loaded the dataset and examined its structure.
  - Identified the labels indicating PCI-DSS compliance.

- **Data Preprocessing**:
  - Standardized and cleaned the text data through:
    - Lowercasing
    - Punctuation removal
    - Tokenization
    - Stop words elimination
  - Created a new feature `contains_number` to capture numeric characters in the messages.

- **Feature Engineering**:
  - Used TF-IDF to transform text data into numerical features.
  - Combined TF-IDF features with the `contains_number` variable.

- **Model Training and Evaluation**:
  - Trained both Logistic Regression and Decision Tree models.
  - Evaluated model performance and generated detailed metrics.

- **Custom Message Prediction**:
  - Implemented a function to predict PCI-DSS compliance for custom input messages.
  - Preprocessed input, extracted features, and made predictions.
  - Saved results, including classification reports and predictions, to a JSON file.

- **Database Integration**:
  - Created a function to load JSON data into a MongoDB collection.
  - Facilitated efficient storage and retrieval of results.

This comprehensive approach included model deployment, tuning to reduce bias, feature engineering, and database integration to ensure a robust PCI-DSS compliance detection system.

## MongoDB Deployment

![MongoDB Deployment](PCI-DSS-TASK-1/mongo%20db%20deployment.png)

