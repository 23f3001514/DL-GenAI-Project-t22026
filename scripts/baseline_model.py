# baseline_model.py
# logistic regression baseline model for Smart MCQ Solver
# this is the first proper model - uses TF-IDF features with logistic regression
# surprisingly works very well because the correct answer has lexical overlap with prompt

import numpy as np
import pandas as pd
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# importing our utility functions
import sys
sys.path.append('../src')
from utils import (
    preprocess_dataframe,
    build_tfidf_features,
    map_at_k,
    get_top3_predictions
)

# file paths
TRAIN_PATH  = '/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv'
TEST_PATH   = '/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv'
SAMPLE_PATH = '/kaggle/input/competitions/smart-mcq-solver-challenge/sample_submission.csv'

WANDB_PROJECT = '23f3001514-t22026'
WANDB_ENTITY  = '23f3001514-instituition'
WANDB_KEY     = 'YOUR_WANDB_API_KEY_HERE'


def load_and_preprocess():
    print('loading data...')
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    print('preprocessing...')
    train = preprocess_dataframe(train)
    test  = preprocess_dataframe(test)

    print(f'train shape: {train.shape}')
    print(f'test shape: {test.shape}')
    return train, test


def train_logistic_regression(X_train, y_train, C=1.0):
    print('training logistic regression...')

    lr_model = LogisticRegression(
        max_iter    = 1000,
        C           = C,
        solver      = 'lbfgs',
        multi_class = 'multinomial'
    )
    lr_model.fit(X_train, y_train)
    print('training done!')
    return lr_model


def evaluate_model(model, X_train, y_train):
    print('evaluating with 5-fold cross validation...')

    f1_scores  = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
    acc_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    avg_f1  = f1_scores.mean()
    avg_acc = acc_scores.mean()

    # mAP@3 on training data
    train_probs = model.predict_proba(X_train)
    train_top3  = []
    for i in range(len(train_probs)):
        top3 = get_top3_predictions(train_probs[i], list(model.classes_))
        train_top3.append(top3)

    map3 = map_at_k(y_train.tolist(), train_top3)

    print(f'CV Macro F1  : {round(avg_f1, 4)}')
    print(f'CV Accuracy  : {round(avg_acc, 4)}')
    print(f'Train mAP@3  : {round(map3, 4)}')

    return avg_f1, avg_acc, map3


def generate_submission(model, X_test, sample_path, output_path):
    print('generating test predictions...')

    test_probs = model.predict_proba(X_test)
    top3_preds = []

    for i in range(len(test_probs)):
        top3   = get_top3_predictions(test_probs[i], list(model.classes_))
        pred_str = ' '.join(top3)
        top3_preds.append(pred_str)

    submission = pd.read_csv(sample_path)
    submission['Prediction'] = top3_preds
    submission.to_csv(output_path, index=False)

    print(f'submission saved to {output_path}')
    print(submission.head())
    return test_probs


def main():
    # wandb login
    wandb.login(key=WANDB_KEY, relogin=True)

    # load data
    train, test = load_and_preprocess()
    y_train     = train['answer']

    # build TF-IDF features
    X_train, X_test, vectorizer = build_tfidf_features(
        train['full_text'].tolist(),
        test['full_text'].tolist(),
        max_features=5000
    )

    # start wandb run
    run = wandb.init(
        project = WANDB_PROJECT,
        entity  = WANDB_ENTITY,
        name    = 'logistic-regression-tfidf',
        config  = {
            'model'        : 'LogisticRegression',
            'features'     : 'TF-IDF',
            'max_features' : 5000,
            'C'            : 1.0,
            'max_iter'     : 1000
        }
    )

    # train model
    lr_model = train_logistic_regression(X_train, y_train, C=1.0)

    # evaluate
    f1, acc, map3 = evaluate_model(lr_model, X_train, y_train)

    # log to wandb
    wandb.log({
        'macro_f1' : f1,
        'accuracy' : acc,
        'map_at_3' : map3
    })
    run.finish()

    # generate submission
    generate_submission(
        lr_model, X_test,
        SAMPLE_PATH,
        '/kaggle/working/submission_lr.csv'
    )

    print('baseline model complete!')


if __name__ == '__main__':
    main()