# inference.py
# final inference script that combines all 3 models and generates submission
# this is the script that produces the final kaggle submission

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('../src')
from utils import (
    clean_text,
    tokenize,
    preprocess_dataframe,
    build_tfidf_features,
    map_at_k,
    get_top3_predictions
)
from models import TextCNN, TextCRNN, train_model, get_test_probabilities

# file paths
TRAIN_PATH  = '/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv'
TEST_PATH   = '/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv'
SAMPLE_PATH = '/kaggle/input/competitions/smart-mcq-solver-challenge/sample_submission.csv'
OUTPUT_PATH = '/kaggle/working/submission.csv'

# model config
MAX_LENGTH    = 256
VOCAB_SIZE_MAX = 15000
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE    = 32

# ensemble weights - tuned based on individual model performance
# LR gets highest weight because it performs best on this task
WEIGHT_LR   = 0.70
WEIGHT_CNN  = 0.20
WEIGHT_CRNN = 0.10


def build_vocab_and_sequences(train, test):
    """build vocabulary and convert texts to integer sequences."""

    all_texts = train['full_text'].tolist() + test['full_text'].tolist()

    # counting word frequencies
    word_counter = Counter()
    for text in all_texts:
        words = tokenize(text)
        for word in words:
            word_counter[word] += 1

    # building vocab - only words that appear at least 2 times
    vocab = ['<PAD>', '<UNK>']
    for word, count in word_counter.most_common(VOCAB_SIZE_MAX):
        if count >= 2:
            vocab.append(word)

    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    vocab_size  = len(vocab)
    print(f'vocabulary size: {vocab_size}')

    def text_to_sequence(text):
        words   = tokenize(text)
        indices = [word_to_idx.get(w, 1) for w in words[:MAX_LENGTH]]
        while len(indices) < MAX_LENGTH:
            indices.append(0)
        return indices

    # converting to sequences
    train_seqs = [text_to_sequence(t) for t in train['full_text']]
    test_seqs  = [text_to_sequence(t) for t in test['full_text']]

    return train_seqs, test_seqs, vocab_size


def create_data_loaders(train_seqs, test_seqs, y_encoded):
    """create pytorch data loaders."""
    from torch.utils.data import random_split

    X_train = torch.tensor(train_seqs, dtype=torch.long)
    X_test  = torch.tensor(test_seqs,  dtype=torch.long)
    y_train = torch.tensor(y_encoded,  dtype=torch.long)

    full_dataset = TensorDataset(X_train, y_train)
    train_size   = int(0.8 * len(full_dataset))
    val_size     = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def run_inference():
    print('=' * 50)
    print('SMART MCQ SOLVER - INFERENCE')
    print('=' * 50)
    print(f'device: {DEVICE}')

    # load and preprocess data
    print('\nloading data...')
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    train = preprocess_dataframe(train)
    test  = preprocess_dataframe(test)
    y_train_labels = train['answer']

    # ── MODEL 1: Logistic Regression ──────────────────────────
    print('\ntraining logistic regression...')
    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(
        train['full_text'].tolist(),
        test['full_text'].tolist(),
        max_features=5000
    )
    lr_model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                   multi_class='multinomial')
    lr_model.fit(X_train_tfidf, y_train_labels)
    lr_test_probs = lr_model.predict_proba(X_test_tfidf)
    print('LR done!')

    # ── PREPARE DL DATA ───────────────────────────────────────
    print('\npreparing deep learning data...')
    label_encoder = LabelEncoder()
    y_encoded     = label_encoder.fit_transform(y_train_labels)
    class_names   = label_encoder.classes_

    train_seqs, test_seqs, vocab_size = build_vocab_and_sequences(train, test)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_seqs, test_seqs, y_encoded
    )

    # ── MODEL 2: TextCNN ──────────────────────────────────────
    print('\ntraining TextCNN...')
    cnn_model = TextCNN(
        vocab_size    = vocab_size,
        embedding_dim = 128,
        num_classes   = 5,
        num_filters   = 128,
        kernel_sizes  = [2, 3, 4]
    ).to(DEVICE)

    cnn_model = train_model(
        model        = cnn_model,
        model_name   = 'textcnn-scratch',
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = DEVICE,
        num_epochs   = 20,
        wandb_config = {
            'model'        : 'TextCNN',
            'vocab_size'   : vocab_size,
            'embedding_dim': 128,
            'num_filters'  : 128,
            'kernel_sizes' : [2, 3, 4]
        }
    )
    cnn_test_probs = get_test_probabilities(cnn_model, test_loader, DEVICE)

    # ── MODEL 3: TextCRNN ─────────────────────────────────────
    print('\ntraining TextCRNN...')
    crnn_model = TextCRNN(
        vocab_size    = vocab_size,
        embedding_dim = 128,
        num_classes   = 5,
        num_filters   = 128,
        hidden_size   = 128
    ).to(DEVICE)

    crnn_model = train_model(
        model        = crnn_model,
        model_name   = 'textcrnn-cnn-lstm',
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = DEVICE,
        num_epochs   = 20,
        wandb_config = {
            'model'         : 'TextCRNN',
            'vocab_size'    : vocab_size,
            'embedding_dim' : 128,
            'num_filters'   : 128,
            'hidden_size'   : 128,
            'bidirectional' : True
        }
    )
    crnn_test_probs = get_test_probabilities(crnn_model, test_loader, DEVICE)

    # ── ENSEMBLE ──────────────────────────────────────────────
    print('\ncreating ensemble...')
    print(f'weights: LR={WEIGHT_LR} CNN={WEIGHT_CNN} CRNN={WEIGHT_CRNN}')

    ensemble_probs = (
        WEIGHT_LR   * lr_test_probs   +
        WEIGHT_CNN  * cnn_test_probs  +
        WEIGHT_CRNN * crnn_test_probs
    )

    # getting top 3 predictions
    options      = ['A', 'B', 'C', 'D', 'E']
    final_preds  = []
    for i in range(len(ensemble_probs)):
        top3     = get_top3_predictions(ensemble_probs[i], options)
        pred_str = ' '.join(top3)
        final_preds.append(pred_str)

    # saving submission
    submission = pd.read_csv(SAMPLE_PATH)
    submission['Prediction'] = final_preds

    # validation checks
    assert len(submission) == len(test), 'row count mismatch!'
    assert submission['Prediction'].isnull().sum() == 0, 'missing predictions!'
    assert all(len(p.split()) == 3 for p in submission['Prediction']), 'need 3 preds each!'

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f'\nsubmission saved to {OUTPUT_PATH}')
    print(submission.head(10).to_string())
    print('\ndone!')


if __name__ == '__main__':
    run_inference()