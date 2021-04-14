## Data Science Nanodegree
### Disaster Response Pipeline

### Table of Contents
* Required Libraries
* Introduction
* Files
* ETL Pipeline
* ML Pipeline
* Flask Web App

### 1. Required Libraries
Except the default pandas, numpy and scikitlearn libraries, I have used nltk and sqlalchemy libraries.

### 2. Introduction
This project's aim is to create a pipeline that will predict the precense of a disaster based on the messages send. For that, messages are pre-processed, trained and tested towards certain labels.
### 3. Files
Data is provided by Figure 8.

### 4. ETL Pipeline
Python script data/process_data.py performs following tasks on the data.

* Loads and formats the databases
* Cleans the data
* Stores the databases in a SQLite database table

### 5. ML Pipeline
File models/train_classifier.py contains machine learning pipeline that:

* Loads data from the SQLite database table
* Splits the data as training and testing sets
* Builds a machine learning pipeline
* Trains and tunes the created model with GridSearchCV
* Reports out the metrics of the final model
* Exports the final model as a pickle file

### 6. Flask Web App
Run the following command in the app's directory to run your web app python run.py
Run env | grep command to obtain space-id and workspace-id.

Go to http://0.0.0.0:3001/ (change the 0.0.0.0 part according to the information obtained with env|grep)
