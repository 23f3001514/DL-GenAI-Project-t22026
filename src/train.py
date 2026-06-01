# ============================================================
# train.py - Training Script for MCQ Solver
# Author: Rohan Kumar | Roll: 23f3001514
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from collections import Counter
import wandb

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "train_path": "/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv",
    "test_path": "/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv",
    "max_len": 256,
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.001,
    "embed_dim": 128,
    "num_filters": 128,
    "hidden_size": 128,
    "wandb_entity": "23f3001514-instituition",
    "wandb_project": "23f3001514-t22026"
}

# ============================================================
# DATA LOADING
# ============================================================
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def combine_text(row):
    return row['prompt'] + ' ' + row['A'] + ' ' + row['B'] + ' ' + row['C'] + ' ' + row['D'] + ' ' + row['E']

def preprocess(train, test):
    train['full_text'] = train.apply(combine_text, axis=1)
    test['full_text'] = test.apply(combine_text, axis=1)
    return train, test

# ============================================================
# TOKENIZER
# ============================================================
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
# MODELS
# ============================================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=128, kernel_sizes=[2,3,4]):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            c = torch.max(c, dim=2)[0]
            pooled.append(c)
        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        return self.fc(x)

class TextCRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=128, hidden_size=128):
        super(TextCRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(num_filters, hidden_size, batch_first=True,
                           num_layers=2, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        x, (hidden, _) = self.lstm(x)
        x = torch.cat([hidden[-2], hidden[-1]], dim=1)
        x = self.dropout(x)
        return self.fc(x)

# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_logistic_regression(X_train, y_train, config):
    run = wandb.init(
        entity=config['wandb_entity'],
        project=config['wandb_project'],
        name="logistic-regression-tfidf",
        config={"model": "LogisticRegression"}
    )
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_train, y_train)
    score = cross_val_score(lr, X_train, y_train, cv=5, scoring='f1_macro').mean()
    print(f"LR Macro F1: {score:.4f}")
    wandb.log({"macro_f1": score})
    wandb.finish()
    return lr

def train_neural_model(model, train_loader, val_loader, config, model_name, device):
    run = wandb.init(
        entity=config['wandb_entity'],
        project=config['wandb_project'],
        name=model_name,
        config=config
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    best_val_acc = 0

    for epoch in range(config['epochs']):
        model.train()
        train_correct = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_correct += (outputs.argmax(1) == batch_y).sum().item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_correct += (outputs.argmax(1) == batch_y).sum().item()

        train_acc = train_correct / (len(train_loader.dataset))
        val_acc = val_correct / (len(val_loader.dataset))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}.pt')

        wandb.log({"epoch": epoch+1, "train_acc": train_acc, "val_acc": val_acc})
        print(f"Epoch {epoch+1} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

    wandb.finish()
    return model

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess
    train, test = load_data(CONFIG['train_path'], CONFIG['test_path'])
    train, test = preprocess(train, test)

    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(train['answer'])

    # TF-IDF for LR
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(train['full_text'])

    # Train LR
    lr_model = train_logistic_regression(X_train_tfidf, y_encoded, CONFIG)
    print("✅ Logistic Regression trained!")

    # Build vocab for neural models
    all_texts = train['full_text'].tolist() + test['full_text'].tolist()
    vocab, word2idx = build_vocab(all_texts)

    # Prepare tensors
    X_seq = torch.tensor([text_to_sequence(t, word2idx) for t in train['full_text']], dtype=torch.long)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    dataset = TensorDataset(X_seq, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Train CNN
    cnn_model = TextCNN(len(vocab), CONFIG['embed_dim'], 5).to(device)
    train_neural_model(cnn_model, train_loader, val_loader, CONFIG, "textcnn-scratch", device)
    print("✅ CNN trained!")

    # Train CRNN
    crnn_model = TextCRNN(len(vocab), CONFIG['embed_dim'], 5).to(device)
    train_neural_model(crnn_model, train_loader, val_loader, CONFIG, "textcrnn-cnn-lstm", device)
    print("✅ CRNN trained!")

    print("\n🎉 All models trained successfully!")
