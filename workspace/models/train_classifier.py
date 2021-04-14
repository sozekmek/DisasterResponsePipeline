import sys
import pandas as pd
import re
import nltk
import string
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
nltk.download(['punkt', 'wordnet', 'stopwords'])
from sklearn import multioutput
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
from tqdm import tqdm, tqdm_notebook

def load_data(database_filepath):
    """
    Function: Loading the database from the SQL table
    Args:
      database_filepath(str): Path of the messages database as str
    Return:
      X,Y(dataframes): Two databases for inputs and output of the machine learning pipeline
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("end_table",engine)
    X = df["message"]
    Y = df.drop(["message","genre","id","original"],axis=1)
    return X,Y


def tokenize(text):
    """
    Function: Splitting the messages and returning the root of them without the stop words
    Args:
      text(str): List of messages as str
    Return:
      tokens(list of str): A list of messages after process
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())   
    
    # initilize  NLTK packages
    stop_words = nltk.corpus.stopwords.words("english")
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    remove_punc_table = str.maketrans('', '', string.punctuation)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
     Function: Building Random Forest Classifier model
     Return:
       cv(list of str): The model itself ready to be trained
     """

    # Create a pipeline
    rfc_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('rfc',  multioutput.MultiOutputClassifier(RandomForestClassifier()))
    ])
    # Create Grid search parameters
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'rfc__estimator__min_samples_split': [2, 4]
    }

    cv = GridSearchCV(rfc_pipeline, param_grid=parameters, verbose=10)

    return cv


def evaluate_model(model, X_test, y_test):
    """
    Function: Evaluating and printing the F1 score, precision and recall for each           output category of the dataset.
    Args:
    model: The model
    X_test: Test messages
    y_test: Test labels
    """
    y_pred = model.predict(X_test)
    for idx, col in enumerate(y_test):
        print(col)
        print(classification_report(y_test[col], y_pred[:, idx]))
    accuracy = (y_pred == y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))

def save_model(model, model_filepath):
    """
    Function: Saving pickle file of the model
    Args:
    model: The model
    model_filepath (str): Path of the pickle file
    """

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
