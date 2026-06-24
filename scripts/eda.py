# eda.py
# exploratory data analysis for the Smart MCQ Solver project

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# file paths -
TRAIN_PATH = '/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv'
TEST_PATH  = '/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv'

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    return train, test

def basic_info(train, test):
    print('=' * 50)
    print('DATASET BASIC INFO')
    print('=' * 50)
    print('train shape:', train.shape)
    print('test shape:', test.shape)
    print()
    print('columns:', train.columns.tolist())
    print()
    print('missing values in train:')
    print(train.isnull().sum())
    print()
    print('missing values in test:')
    print(test.isnull().sum())

def answer_distribution(train):
    print()
    print('answer distribution:')
    counts = train['answer'].value_counts().sort_index()
    print(counts)
    print()
    print('answer distribution percentage:')
    print(round(counts / len(train) * 100, 2))

def question_length_analysis(train):
    print()
    print('question length analysis:')
    train['prompt_len'] = train['prompt'].apply(lambda x: len(str(x)))
    print('min length  :', train['prompt_len'].min())
    print('max length  :', train['prompt_len'].max())
    print('mean length :', round(train['prompt_len'].mean(), 2))
    print('median length:', train['prompt_len'].median())

def option_length_analysis(train):
    print()
    print('option length analysis (avg characters per option):')
    for opt in ['A', 'B', 'C', 'D', 'E']:
        avg_len = train[opt].apply(lambda x: len(str(x))).mean()
        print(f'  option {opt}: {round(avg_len, 1)} chars')

def plot_eda(train):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Smart MCQ Dataset - EDA', fontsize=13)

    # plot 1 - answer distribution
    counts = train['answer'].value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, color='steelblue', edgecolor='black')
    axes[0].set_title('Answer Distribution')
    axes[0].set_xlabel('Answer Option')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 3, str(v), ha='center')

    # plot 2 - question length distribution
    prompt_lens = train['prompt'].apply(lambda x: len(str(x)))
    axes[1].hist(prompt_lens, bins=30, color='orange', edgecolor='black')
    axes[1].set_title('Question Length Distribution')
    axes[1].set_xlabel('Number of Characters')
    axes[1].set_ylabel('Count')
    axes[1].axvline(prompt_lens.mean(), color='red', linestyle='--',
                    label=f'mean={round(prompt_lens.mean())}')
    axes[1].legend()

    # plot 3 - average option length
    opt_lengths = {}
    for opt in ['A', 'B', 'C', 'D', 'E']:
        opt_lengths[opt] = train[opt].apply(lambda x: len(str(x))).mean()
    axes[2].bar(opt_lengths.keys(), opt_lengths.values(), color='green', edgecolor='black')
    axes[2].set_title('Average Option Length')
    axes[2].set_xlabel('Option')
    axes[2].set_ylabel('Avg Characters')

    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=100, bbox_inches='tight')
    plt.show()
    print('EDA plots saved to eda_plots.png')

def sample_questions(train, n=3):
    print()
    print(f'sample {n} questions from training data:')
    print('=' * 60)
    for i in range(n):
        row = train.iloc[i]
        print(f'Question {i+1}:')
        print(f'  Prompt : {str(row["prompt"])[:150]}...')
        print(f'  A      : {str(row["A"])[:80]}...')
        print(f'  B      : {str(row["B"])[:80]}...')
        print(f'  Answer : {row["answer"]}')
        print()

if __name__ == '__main__':
    print('loading data...')
    train, test = load_data()
    basic_info(train, test)
    answer_distribution(train)
    question_length_analysis(train)
    option_length_analysis(train)
    sample_questions(train)
    plot_eda(train)
    print('EDA complete!')