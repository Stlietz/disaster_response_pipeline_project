# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import os
import pickle


import nltk
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """load cleaned data from SQLite database
    Input: Database_filepath
    Output: X: pandas DataFrame with Messages, Y: pandas DataFrame with categories
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response',engine)

    X = df["message"]
    Y = df.iloc[:,4:]
    category_names = Y.columns.values

    return X, Y, category_names

def remove_punctuation(text, pattern=r"[^a-zA-Z0-9]"):
    """
    By default (regex pattern), remove all chars except for a-z, A-Z, 0-9
    INPUT: raw text
    OUTPUT: returns text without punctuation
    """ 
    
    text = re.sub(pattern, " ", text) # default: Anything that isn't A through Z or 0 through 9 will be replaced by a space
    return text

def substitute_url(text, substitute="urlplaceholder", pattern=r'http\S+'):
    """
    substitutes url in a string with the string "urlplaceholder"

    INPUT: Raw Text
    OUTPUT: returns text with URL (if any) replaced by the substitute (default = "urlplaceholder")
    """
    return re.sub(pattern, substitute, text)

def tokenize(text):
    """
    function to tokenize the raw text
    makes use other user-defined functions "substitute_url" and "remove_punctuation"
    
    returns a list of cleaned (text) tokens; additionally removed uppercase, whitspaces and lemmatizes the text - excludes stopwords
    """
    text_1 = substitute_url(text) 
    
    text_2 = remove_punctuation(text_1)
    # tokenize text
    tokens = word_tokenize(text_2)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # get stop words
    stop_words = stopwords.words("english")
    
    # iterate through each token
    #  lemmatize, make all text lowercase, strip, remove whitespaces and exclude stop words
    clean_tokens = \
    [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if tok not in stop_words]

    return clean_tokens

def build_model(parameters = {'clf__estimator__alpha': (1, 0.05, 1e-1, 0.15)}):
    """
    build a ML pipeline for to classify messages - Multilabel Classification (!) - using cross validation

    Classifier:
    - Multinomial Naive Bayes combined with MultiOutputClassifier (Multilabel Classification)
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Input: model and test data (X and Y)
    make predictions on unseen test data, subsequently PRINT SKLearn's classification report 
    """

    Y_preds = model.predict(X_test)
    print(classification_report(Y_test.values, Y_preds, target_names=category_names))

def save_model(model, model_filepath):
    """
    Input: Model, Filepath
    Output: Pickle file, stores resp. serializes the (model) object.
    """
    pickle.dump(model, open(model_filepath, "wb"))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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