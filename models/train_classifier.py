import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize, pos_tag
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from sklearn import svm
import pickle
nltk.download(['punkt','wordnet','stopwords'])
from sklearn.base import BaseEstimator, TransformerMixin

class count_unique_words(BaseEstimator, TransformerMixin):
    """
    counts the number of unique words in each row of the input
    this class makes a transformer out of a python function
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        output = pd.Series(X).apply(lambda x: len(set(tokenize(x)))).values
        return output.reshape(-1,1)


def load_data(database_filepath):
    """
    loads data from the database whose filepath is given as input and returns two dataframes and a list of column names
    (str) -> df, df, list of (str)
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('select * from disaster_df', con = engine)
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    returns a list of lemmatized lowercased tokens from the input text. function also removes punctuations from text
    (str) -> list of (str)
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = [lemmatizer.lemmatize(re.sub(r'[^\w ]','',word.lower())) for
            word in word_tokenize(re.sub(r'[^\w ]','',text)) if word not in stop_words]
    return tokens

def build_model():
    """
    returns an instance of the pipeline containing text processing steps and model building steps
    """
    pipeline = Pipeline([('features', FeatureUnion([
        ('text_pipe', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, binary=True,
                                ngram_range=(1,2))),
            ('tfidf', TfidfTransformer(smooth_idf=False))])),
        ('wordcount', count_unique_words())
    ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(bootstrap=True,
                                                            min_samples_split=2)))
    ])
    return pipeline
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluates model on the input predictors and target variables for each of the category names provided
    (model, df, df, list of (str)) -> None
    """
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))

def save_model(model, model_filepath):
    """
    saves model in model_filepath as a pickle file
    """
    pickle.dump(model, open(model_filepath,'wb'))


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
