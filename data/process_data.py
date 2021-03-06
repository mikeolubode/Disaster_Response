import sys
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    """
    merges messages and categories into a single dataframe
    inputs
    ------
    filepath of messages (str)
    filepath of categories (str)
    output
    ------
    dataframe containing messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on='id', how='inner')

def clean_data(df1):
    """
    split categories into columns of binary values indicating presence or absence of category
    (df) -> df
    
    """
    df = df1.copy()
    categories = df.categories.str.split(';', expand = True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[0:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int).astype(bool).astype(int)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df,categories], axis =1)
    df = df.drop_duplicates()
    return df
    
def save_data(df, database_filename):
    """
    saves dataframe df into database, replaces table if already exists
    return: None
    """
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_df', engine, if_exists = 'replace', index=False)


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
