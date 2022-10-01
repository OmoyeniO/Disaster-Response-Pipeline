import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load data
    Input: messages and categories data 
    Output merged file
    
    messages_filepath: file path for messages data
    categories_filepath: file path for categories data
    '''
    
    #load datasets
    messages = pd.read_csv('./data/disaster_messages.csv')
    categories = pd.read_csv('./data/disaster_categories.csv')
    
    # merge datasets
    df = pd.merge(messages, categories, on ='id', how = 'left') 


    return df


def clean_data(df):
    '''
    1. Split categories into separate category columns.
    2. Convert category values to just numbers 0 or 1.
    3. Replace categories column in df with new category columns.
    4. Remove duplicates.
    
    Input: input data (df)
    Output: Cleaned data
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = row.str.split('-').apply(lambda x:x[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').apply(lambda x:x[1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df


def save_data(df, database_filename):
    '''
    Save data into database
    
    df: Cleaned data
    database_filename: the database filename
    
    '''
    # Create database engine
    engine = create_engine('sqlite:///Disaster_response.db')
    
    # Save df to database
    df.to_sql('Messages', engine, index=False)
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()