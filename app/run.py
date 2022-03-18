import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

class count_unique_words(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = pd.Series(X).apply(lambda x: len(set(tokenize(x)))).values
        return output.reshape(-1,1)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
def getTopNwords(x, ngram_range=(1,1)):
    vec = CountVectorizer(ngram_range = ngram_range, stop_words="english").fit(x)
    bow = vec.transform(x)
    sumWords = bow.sum(axis = 0)
    wordFreq = [(word, sumWords[0, index]) for word, index in vec.vocabulary_.items()]
    gramDF = pd.DataFrame(wordFreq, columns = ['word','Frequency'])
    gramDF = gramDF.sort_values(by='Frequency', ascending = False)
    return gramDF

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    single_word_df = getTopNwords(df.message.dropna(),ngram_range=(1,1)).head(10)
    single_word = single_word_df.word
    single_freq = single_word_df.Frequency
    target = pd.melt(df.iloc[:,4:], var_name = 'Target', value_name="indicator")
    target = target.groupby('Target')['indicator'].sum().reset_index()
    target.columns = ['Target', 'Count']
    target = target.sort_values(by = 'Count', ascending = False).head(10)
    
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=single_word,
                    y=single_freq
                )
            ],

            'layout': {
                'title': 'Unigrams Plot of Message Data',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Keywords"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=target.Target,
                    y=target.Count
                )
            ],

            'layout': {
                'title': 'Distribution of most frequent Target variables',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Target Variables"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()