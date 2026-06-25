# utils.py
# utility functions for the Smart MCQ Solver project
# updated: added mAP@3 metric and TF-IDF feature extraction

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── TEXT PREPROCESSING ────────────────────────────────────────

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
    """
    cleaned = clean_text(text)
    tokens  = cleaned.split()
    return tokens


def combine_all_options(row):
    """
    combines the question prompt with all 5 answer options
    into one single text string.
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
    return ' '.join(parts)


def preprocess_dataframe(df):
    """
    applies all preprocessing to a dataframe.
    adds a 'full_text' column with combined text.
    """
    df = df.copy()
    df['full_text'] = df.apply(combine_all_options, axis=1)
    return df


# ── MAP@3 METRIC ─────────────────────────────────────────────

def average_precision_at_k(actual, predicted, k=3):
    """
    compute average precision at k for a single question.
    
    args:
        actual    : correct answer string e.g. 'B'
        predicted : list of predictions e.g. ['B', 'A', 'C']
        k         : cutoff rank (default 3)
    returns:
        float: AP@K score between 0 and 1
        
    scoring:
        correct at rank 1 -> 1.000
        correct at rank 2 -> 0.500
        correct at rank 3 -> 0.333
        not in top 3      -> 0.000
    """
    predicted = predicted[:k]
    score     = 0.0
    num_hits  = 0

    for i in range(len(predicted)):
        if predicted[i] == actual:
            num_hits += 1
            score    += num_hits / (i + 1.0)

    return score


def map_at_k(actuals, predictions, k=3):
    """
    mean average precision at k across all questions.
    
    args:
        actuals     : list of true answer strings
        predictions : list of lists, each with top-k predictions
    returns:
        float: mAP@K score
    """
    total = 0.0
    for i in range(len(actuals)):
        total += average_precision_at_k(actuals[i], predictions[i], k)
    return total / len(actuals)


def get_top3_predictions(probs, classes):
    """
    convert probability array to top 3 class labels.
    sorts by probability descending and returns top 3 labels.
    
    args:
        probs   : numpy array of probabilities
        classes : list of class names e.g. ['A','B','C','D','E']
    returns:
        list of top 3 class labels
    """
    sorted_indices = np.argsort(probs)[::-1]
    top3 = [classes[idx] for idx in sorted_indices[:3]]
    return top3


# ── TF-IDF FEATURES ──────────────────────────────────────────

def build_tfidf_features(train_texts, test_texts, max_features=5000):
    """
    build TF-IDF feature matrices for train and test data.
    
    TF-IDF = Term Frequency x Inverse Document Frequency
    - TF: how often a word appears in a document
    - IDF: penalizes words that appear in many documents
    - result: high score for words that are important in one doc but rare overall
    
    args:
        train_texts  : list of training text strings
        test_texts   : list of test text strings
        max_features : number of top words to keep (default 5000)
    returns:
        X_train : sparse matrix of shape (n_train, max_features)
        X_test  : sparse matrix of shape (n_test, max_features)
        vectorizer: fitted TfidfVectorizer object
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train    = vectorizer.fit_transform(train_texts)
    X_test     = vectorizer.transform(test_texts)

    print(f'TF-IDF feature matrix shape: {X_train.shape}')
    return X_train, X_test, vectorizer


def compute_cosine_similarity_scores(row, tfidf_vectorizer):
    """
    compute cosine similarity between the prompt and each option.
    
    cosine similarity measures the angle between two vectors.
    score of 1.0 = identical direction (very similar text)
    score of 0.0 = completely different text
    
    args:
        row             : single dataframe row with prompt and options
        tfidf_vectorizer: fitted TfidfVectorizer
    returns:
        dict of {option: similarity_score}
    """
    options     = ['A', 'B', 'C', 'D', 'E']
    prompt_vec  = tfidf_vectorizer.transform([clean_text(row['prompt'])])
    scores      = {}

    for opt in options:
        opt_vec     = tfidf_vectorizer.transform([clean_text(row[opt])])
        sim_score   = cosine_similarity(prompt_vec, opt_vec)[0][0]
        scores[opt] = sim_score

    return scores


if __name__ == '__main__':
    # test mAP@3
    actuals     = ['B', 'A', 'C']
    predictions = [['B', 'A', 'C'], ['C', 'A', 'B'], ['C', 'A', 'B']]
    score       = map_at_k(actuals, predictions)
    print(f'mAP@3 test: {round(score, 4)} (expected 0.8333)')
    assert abs(score - 0.8333) < 0.001
    print('utils.py working correctly!')