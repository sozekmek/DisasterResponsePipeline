## Data Science Nanodegree
### Disaster Response Pipeline
During a natural disaster, need for medical aid or rescue become very intense. Predicting where is need for support proves to be very important for rapid response and proper distribution of workforce. With respect to the message database Figure 8 provided, I have prepared a Machine Learning pipeline to predict whether there is a need / request or not. 

### Table of Contents
* Required Libraries
* Introduction
* Files
* ETL Pipeline
* ML Pipeline
* Flask Web App
* How To Run
* Dashboard

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
Web App created with Flask is used for demonstrating the label distribution of the data through a dashboard.

### 7. How To Run
To run ETL pipeline that cleans data and stores in database python: data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Run the following command in the app's directory to run your web app python run.py
Run env | grep WORK command to obtain space-id and workspace-id.

Go to https://SPACEID-3001.SPACEDOMAIN with respect to the parameters obtained from previous command.

### 8. Dashboard

![image](https://user-images.githubusercontent.com/61328773/114700337-13204000-9d2a-11eb-9e83-8a4846f4dc91.png)

