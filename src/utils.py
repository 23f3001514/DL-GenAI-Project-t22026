# ============================================================
# utils.py - Utility Functions for MCQ Solver
# Author: Rohan Kumar | Roll: 23f3001514
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ============================================================
# DATA UTILITIES
# ============================================================
def load_data(train_path, test_path, sample_sub_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sub = pd.read_csv(sample_sub_path)
    print(f"Train: {train.shape} | Test: {test.shape}")
    return train, test, sub

def combine_text(row):
    return row['prompt'] + ' ' + row['A'] + ' ' + row['B'] + ' ' + row['C'] + ' ' + row['D'] + ' ' + row['E']

def check_missing(df, name="DataFrame"):
    missing = df.isnull().sum()
    print(f"\n=== {name} Missing Values ===")
    print(missing)
    return missing

# ============================================================
# EDA UTILITIES
# ============================================================
def plot_answer_distribution(train, save_path='eda_plots.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    train['answer'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Answer Distribution')
    axes[0].set_xlabel('Answer')
    axes[0].set_ylabel('Count')
    train['prompt_length'] = train['prompt'].apply(len)
    axes[1].hist(train['prompt_length'], bins=30, color='orange')
    axes[1].set_title('Question Length Distribution')
    axes[1].set_xlabel('Character Length')
    axes[1].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")

def analyze_length_pattern(train):
    for col in ['A', 'B', 'C', 'D', 'E']:
        train[f'{col}_len'] = train[col].apply(len)
    train['correct_len'] = train.apply(lambda r: len(r[r['answer']]), axis=1)
    train['max_len'] = train[['A_len','B_len','C_len','D_len','E_len']].max(axis=1)
    train['correct_is_longest'] = train['correct_len'] == train['max_len']
    pct = train['correct_is_longest'].mean() * 100
    print(f"Correct answer is longest option: {pct:.1f}% of the time")
    return pct

# ============================================================
# EVALUATION UTILITIES
# ============================================================
def evaluate_model(y_true, y_pred, label_encoder):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=label_encoder.classes_))
    return acc, f1

def plot_training_history(train_accs, val_accs, model_name):
    plt.figure(figsize=(10, 4))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model_name}_history.png')
    plt.show()

def compare_models(results):
    """
    results = [
        {"model": "LR", "val_acc": 0.99, "kaggle_score": 0.74},
        ...
    ]
    """
    df = pd.DataFrame(results)
    print("\n=== MODEL COMPARISON ===")
    print(df.to_string(index=False))
    
    plt.figure(figsize=(10, 4))
    plt.bar(df['model'], df['kaggle_score'], color='steelblue')
    plt.title('Model Kaggle Score Comparison')
    plt.xlabel('Model')
    plt.ylabel('MAP@3 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    return df

# ============================================================
# SUBMISSION UTILITIES  
# ============================================================
def validate_submission(sub, expected_rows=500):
    assert len(sub) == expected_rows, f"Expected {expected_rows} rows, got {len(sub)}"
    assert 'Prediction' in sub.columns, "Missing Prediction column!"
    for pred in sub['Prediction']:
        parts = pred.split()
        assert len(parts) == 3, f"Expected 3 predictions, got {len(parts)}"
        for p in parts:
            assert p in ['A','B','C','D','E'], f"Invalid prediction: {p}"
    print(f"✅ Submission valid! {len(sub)} rows")
    return True
