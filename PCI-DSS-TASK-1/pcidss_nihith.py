
#Test 1 

print("hello" )
#---------------------------------------------------------#
## Task-1 :

# Work on the PCI-DSS ( Payment Card Industry Data Security Standard)  detection by using the text dataset.


'''
my inference :
It's a set of rules and guidelines that organizations that handle credit card information must follow to protect cardholder data from unauthorized access and breaches. 

'''

#---------------------------------------------------------#
### importing all necessary libraries 

import pandas as pd
import string
import nltk
import json

nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from scipy.sparse import hstack
from pymongo import MongoClient


#---------------loading the dataset and understanding the data --------------------#


#file_path = '/Users/HP/Desktop/credit_pcidss_data.xlsx'
file_path = '/Users/HP/Desktop/credit_pcidss_data_v2.xlsx'

credit_pci_df = pd.read_excel(file_path)

print(credit_pci_df.head())

print(f'the dimension of the data set is : {(credit_pci_df.shape)}')

print('The header (column names) of the dataset as a list is:')
print(credit_pci_df.columns.tolist())

## in the columns label is the predictor that is a boolean value that says whether the message the only variable in the dataset contains credit card data or not !!.

## looks like data is loaded now we move on to further steps


#-------------------Exploring the data ---------------------------#


print(credit_pci_df.describe())

# checking for missing values
print(credit_pci_df.isnull().sum())

# see the label vloumn distribution 

label_counts = credit_pci_df['label'].value_counts()

print(label_counts)

#--------------------data preprocessing-------------------------#

'''

# I am performing text preprocessing to standardize and clean the PCI-DSS dataset's textual data, ensuring it is ready for subsequent analysis and model training. 
This includes tasks like lowercasing, punctuation removal,tokenization, and stop words elimination to improve data quality and model performance.

'''
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercasing the text
    text = text.lower()
    # Removing punctuation using translation method
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenizing the text into words
    tokens = word_tokenize(text)
    # Removing stop words using lambda function
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def contains_number(text):
    return int(any(char.isdigit() for char in text))



credit_pci_df['processed_message'] = credit_pci_df['message'].apply(preprocess_text)

#-------------------Creating a new variable----------------------#


credit_pci_df['contains_number'] = credit_pci_df['processed_message'].apply(contains_number)

# Displaying the processed DataFrame

print(credit_pci_df[['label', 'message', 'processed_message','contains_number']].head(10))


#-------------------TF- IDF----------------------#

'''

I'm using TF-IDF to transform text data because it helps weigh the importance of words by considering their frequency in each document 
relative to their occurrence across the entire dataset, which enhances the accuracy of my text classification model for PCI-DSS detection.


'''

# Continuing from the preprocessed DataFrame

text_data = credit_pci_df['processed_message']
labels = credit_pci_df['label']

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3))

# Fit TF-IDF Vectorizer to the preprocessed text
X_tfidf = tfidf_vectorizer.fit_transform(text_data)



print(X_tfidf)


# Combine TF-IDF features with contains_number
contains_number_feature = credit_pci_df['contains_number'].values.reshape(-1, 1)
X_combined = hstack([X_tfidf, contains_number_feature])

print(X_combined)

#---------------performing Logistic regression----------------------#

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, labels, test_size=0.2, random_state=42)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# Evaluating model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.2f}")

# Confusion matrix and classification report
print("\nConfusion Matrix of Logistic regression:")
print(confusion_matrix(y_test, y_pred))

class_report_lr = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#---------------performing Decision Trees algorithm----------------------#


# Initializing the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Training the model
dt_classifier.fit(X_train, y_train)

# Predicting on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix of decision tree:")
print(conf_matrix)

class_report_dt = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report_dt)


#---------------Creating a function for custom message prediction  ----------------------#

Custom_input = input("please enter message to predict pci-dss compliance :")


def predict_pcidss_compliance(input_text):
    processed_text = preprocess_text(input_text)
    contains_number_feature = contains_number(processed_text)
    tfidf_features = tfidf_vectorizer.transform([processed_text])
    combined_features = hstack([tfidf_features, [[contains_number_feature]]])
    prediction_pcidss = dt_classifier.predict(combined_features)
    if prediction_pcidss == 1:
        prediction_msg =  f'your message "{input_text}" is  not following PCI-DSS complaince '

    else: 
        prediction_msg = f'your message "{input_text}" is following PCI-DSS complaince'

    result = {
        "classification Report for Logistic Regression" : class_report_lr,
        "classification Report for Decision Tree" : class_report_dt,
        "Text": input_text,
        "Prediction": prediction_msg
    }
    print(prediction_msg)
    return result

    

rslt =predict_pcidss_compliance(Custom_input)



#---------------Saving the result into a Json File ----------------------#
json_file_name = 'pcidss_compliance_results.json'

with open(json_file_name, 'w') as json_file:
    json.dump(rslt, json_file, indent=4)
    print(f"the Output is Successfully saved  as {json_file_name} file  ")



#---------------Function to load JSON data into MongoDB----------------------#

def load_json_to_mongodb(json_file_path, db_name, collection_name, mongo_uri="mongodb://localhost:27017"):
    # Create a MongoDB client
    client = MongoClient(mongo_uri)

    # Access the database
    db = client[db_name]

    # Access the collection
    collection = db[collection_name]

    # Open the JSON file and load its contents
    with open(json_file_path, 'r') as file:
        data = json.load(file)

        # If the JSON file contains an array of documents
        if isinstance(data, list):
            collection.insert_many(data)
        else:
            collection.insert_one(data)

    print(f"Data from {json_file_path} has been successfully imported into the {db_name}.{collection_name} collection.")


json_file_path = r'/Users/HP/Downloads/pcidss_compliance_results.json'  
db_name = 'Eitacies_nn_db'
collection_name = 'PCIDSS_collection'

load_json_to_mongodb(json_file_path, db_name, collection_name)

