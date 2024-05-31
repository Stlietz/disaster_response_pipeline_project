# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    """
    Load and combine datasets

    Input:
    - Messages dataset (filepath)
    - Categories dataset (file path)

    Output: pandas DataFrame, both datasets merged
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id", how="inner")

    return df


def clean_data(df):
    """
    Input: pandas DataFrame
    Output: cleaned DataFrame with one-hot encoded categories
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # one hot encoded df / dummy variables should only take values 0 or 1
    # i.e. will replace 2's with 1's
    mask = categories.related==2

    categories.loc[mask, "related"] = 1

    # drop the original categories column from `df`
    df.drop("categories", axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    #drop all categories with 0 counts - otherwise ML Pipeline will run into error
    to_drop_list = []
    to_drop_list.append(df.iloc[:,4:].columns[(df.iloc[:,4:].sum(axis=0) == 0)][0])

    for col in list(to_drop_list):
        df.drop(col, axis=1, inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save data to SQLite database

    Input: 
    - df
    - path to SQL database

    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response', engine, index=False, if_exists="replace")  



def main():
    """
    Main Function of the ETL-Script
    Given the provided user input, loads, cleans and stores the data
    """
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