import sys
import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import joblib 
import lightgbm as lgb
from sklearn.decomposition import TruncatedSVD
import pickle

def load_data(database_filepath):
    '''
     Load data from database file path
     Output feature set, target and target categories
    
     database_filepath: database file path
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath) 
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])   

    return X, Y, Y.columns


def tokenize(text):
    '''
    Tokenize, lemmatize, normalize, strip, remove stop words from the text
    
    :param text: input text
    '''
    # Initialization
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))
    
    # Get clean tokens after lemmatization, normalization, stripping and stop words removal
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if tok not in stopWords:
            clean_tokens.append(clean_tok)

    return clean_tokens
    


def build_model():
    '''
    Build a pipeline with TFIDF, truncated SVD, and an LGBMclassifier
    Input: Input text

    '''
    #Build pipeline
    pipeline2 = Pipeline([  
        ('vect', CountVectorizer()),
        ('best', TruncatedSVD()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(lgb.LGBMClassifier()))
    ])

    #Parameter tunning for grid search
    parameters2 = {'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100],
              'clf__estimator__learning_rate': [1,2] }

    # Initialize GridSearch 
    cv2 = GridSearchCV(pipeline2, param_grid=parameters2)

    return cv2


def evaluate_model(model, X_test, y_test, category_names):
    '''
    Function to test the model, report the F1 score, precision and recall for each output category (classification report).
    Input: Model, test set for X and y
    Output: Prints classification report

    '''
    #predict with model
    y_pred = model.predict(X_test)

    # Turn prediction into DataFrame
    y_pred = pd.DataFrame(y_pred,columns=category_names)

    # For each category column, print performance
    for col in category_names:
        print(f'Column Name:{col}\n')
        print(classification_report(y_test[col],y_pred[col]))
            


def save_model(model, model_filepath):
    '''
    Save model to a pickle file
    
    model: model object
    model_filepath: model output file path

    '''

    joblib.dump(model, model_filepath) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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