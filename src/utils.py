# utils.py
# utility functions for the Smart MCQ Solver project
# this file contains text cleaning and preprocessing functions

import re
import numpy as np


def clean_text(text):
    """
    clean raw text for NLP processing.
    steps:
    - convert to lowercase
    - remove special characters (keep only letters, numbers, spaces)
    - remove extra whitespace
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def tokenize(text):
    """
    simple whitespace tokenizer.
    first cleans the text then splits by spaces.
    returns a list of word tokens.
    
    example:
        tokenize("Hello World!") -> ['hello', 'world']
    """
    cleaned = clean_text(text)
    tokens  = cleaned.split()
    return tokens


def combine_all_options(row):
    """
    combines the question prompt with all 5 answer options
    into one single text string.
    
    this gives models full context about the question
    and all possible answers together.
    
    format: 'prompt A B C D E'
    """
    parts = [
        str(row['prompt']),
        str(row['A']),
        str(row['B']),
        str(row['C']),
        str(row['D']),
        str(row['E'])
    ]
    combined = ' '.join(parts)
    return combined


def preprocess_dataframe(df):
    """
    applies all preprocessing to a dataframe.
    adds a 'full_text' column with combined text.
    
    args:
        df: pandas dataframe with columns prompt, A, B, C, D, E
    returns:
        df with new 'full_text' column added
    """
    df = df.copy()
    df['full_text'] = df.apply(combine_all_options, axis=1)
    return df


if __name__ == '__main__':
    # quick test to make sure everything works
    sample_text = "What is the capital of France?? It's Paris!!"
    print('original:', sample_text)
    print('cleaned:', clean_text(sample_text))
    print('tokens:', tokenize(sample_text))