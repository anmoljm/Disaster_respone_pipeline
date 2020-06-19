# import libraries
import sys
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
import pickle

import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sqlalchemy import create_engine



def load_data(database_filepath):
    """
    Saves cleaned data to an SQL database
    Args:
    df pandas_dataframe: Cleaned data returned from clean_data() function
    database_file_name str: File path of SQL Database into which the cleaned\
    data is to be saved
    Returns:
    None
    """    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response',engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis = 1)
    category_names = list(df.columns[4:])
    return X,Y,category_names

def tokenize(text):
    """
    Tokenizes text data
    Args:
    text str: Messages as text data
    Returns:
    words list: Processed text after normalizing, tokenizing and lemmatizing
    """

    #normalizing the text and detect urls
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize
    words = word_tokenize(text)
    
    #remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    #lemmatization
    lemmed = [WordNetLemmatizer().lemmatize(w).strip() for w in words]
    
    #stemming
    #stemmed = [PorterStemmer().stem(w).strip() for w in lemmed]
    
    return lemmed


def build_model():
    """
    Build model with GridSearchCV
    
    Returns:
    Trained model after performing grid search
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100]
        #'clf__estimator__min_samples_split': [2, 4]
        } 

    cv = GridSearchCV(pipeline, param_grid=parameters) 

    return cv

def evaluate_model(model, x_test, y_test, category_names):
    """
    Shows model's performance on test data
    Args:
    model: trained model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels
    """
    y_pred = model.predict(x_test)

    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(y_test.iloc[:, i].values, y_pred[:,i])))

def save_model(model, model_filepath):
    """
    Saves the model to a Python pickle file    
    Args:
    model: Trained model
    model_filepath: Filepath to save the model
    """    
    pickle.dump(model, open(model_filepath, "wb"))
    return None


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