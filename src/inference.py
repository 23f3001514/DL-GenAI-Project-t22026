# ============================================================
# inference.py - Inference Script for MCQ Solver
# Author: Rohan Kumar | Roll: 23f3001514
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset

# ============================================================
# TEXT PREPROCESSING
# ============================================================
def combine_text(row):
    return row['prompt'] + ' ' + row['A'] + ' ' + row['B'] + ' ' + row['C'] + ' ' + row['D'] + ' ' + row['E']

def build_vocab(texts, max_vocab=10000):
    word_counts = Counter(word for text in texts for word in text.lower().split())
    vocab = ['<PAD>', '<UNK>'] + [w for w, c in word_counts.most_common(max_vocab)]
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx

def text_to_sequence(text, word2idx, max_len=256):
    tokens = text.lower().split()
    seq = [word2idx.get(w, 1) for w in tokens[:max_len]]
    seq += [0] * (max_len - len(seq))
    return seq

# ============================================================
# DATASET CLASS FOR TRANSFORMERS
# ============================================================
class MCQDataset(Dataset):
    def __init__(self, texts, tokenizer, labels=None, max_len=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ============================================================
# INFERENCE FUNCTIONS
# ============================================================
def get_top3_lr(model, tfidf, test, le):
    """Logistic Regression inference"""
    test_features = tfidf.transform(test['full_text'])
    probs = model.predict_proba(test_features)
    top3_preds = []
    for prob in probs:
        top3_idx = np.argsort(prob)[::-1][:3]
        top3_labels = [le.classes_[i] for i in top3_idx]
        top3_preds.append(' '.join(top3_labels))
    return top3_preds

def get_top3_neural(model, X_test_seq, le, device, batch_size=32):
    """CNN/CRNN inference"""
    model.eval()
    all_probs = []
    test_loader = DataLoader(TensorDataset(X_test_seq), batch_size=batch_size)
    with torch.no_grad():
        for batch in test_loader:
            batch_x = batch[0].to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    top3_preds = []
    for prob in all_probs:
        top3_idx = np.argsort(prob)[::-1][:3]
        top3_labels = [le.classes_[i] for i in top3_idx]
        top3_preds.append(' '.join(top3_labels))
    return top3_preds

def get_top3_transformer(model, test_loader, le, device):
    """Transformer (DistilBERT/RoBERTa) inference"""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    top3_preds = []
    for prob in all_probs:
        top3_idx = np.argsort(prob)[::-1][:3]
        top3_labels = [le.classes_[i] for i in top3_idx]
        top3_preds.append(' '.join(top3_labels))
    return top3_preds

def save_submission(predictions, sample_sub_path, output_path='submission.csv'):
    """Save predictions to submission file"""
    sub = pd.read_csv(sample_sub_path)
    sub['Prediction'] = predictions
    assert len(sub) > 0, "Submission is empty!"
    assert 'Prediction' in sub.columns, "Missing Prediction column!"
    sub.to_csv(output_path, index=False)
    print(f"✅ Submission saved to {output_path}")
    print(f"Shape: {sub.shape}")
    print(sub.head())
    return sub

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    test = pd.read_csv('/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv')
    test['full_text'] = test.apply(combine_text, axis=1)
    print(f"Test shape: {test.shape}")
    print("✅ Inference script ready!")
