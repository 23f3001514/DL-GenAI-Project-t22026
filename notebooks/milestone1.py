# Milestone 1 

import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_PATH = '/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv'
train = pd.read_csv(TRAIN_PATH)

# Q1 - Answer frequency distribution
freq = train['answer'].value_counts()
print('Answer frequencies:', freq.to_dict())
print('Most frequent:', freq.max(), '| Least frequent:', freq.min())
print('Q1 Answer:', freq.max() + freq.min())

# Q2 - Unique words in prompt column
def clean_q2(text):
    text = str(text).lower()
    for ch in string.punctuation:
        text = text.replace(ch, '')
    return text

all_words = set()
for p in train['prompt']:
    for w in clean_q2(p).split():
        all_words.add(w)
print('Q2 Answer (unique words):', len(all_words))

# Q3 - Words in Row ID 1 after stop word removal
row1 = train[train['id'] == 1].iloc[0]
row1_words = clean_q2(row1['prompt']).split()
filtered = [w for w in row1_words if w not in ENGLISH_STOP_WORDS]
print('Q3 Answer (words after stop words):', len(filtered))

# Q4 - TF-IDF vocabulary size
combined = [str(r['prompt'])+' '+str(r['A'])+' '+str(r['B'])+' '+str(r['C'])+' '+str(r['D'])+' '+str(r['E'])
            for _, r in train.iterrows()]
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(combined)
print('Q4 Answer (TF-IDF vocab size):', len(tfidf.vocabulary_))

# Q5 - Cosine similarity prompt vs option A for Row ID 1
pv = tfidf.transform([str(row1['prompt'])])
av = tfidf.transform([str(row1['A'])])
sim = cosine_similarity(pv, av)[0][0]
print('Q5 Answer (cosine sim):', round(sim, 4))

# Q6 - % where highest cosine sim = correct answer
options = ['A','B','C','D','E']
correct = 0
for _, row in train.iterrows():
    pv = tfidf.transform([str(row['prompt'])])
    scores = {opt: cosine_similarity(pv, tfidf.transform([str(row[opt])]))[0][0] for opt in options}
    if max(scores, key=scores.get) == row['answer']:
        correct += 1
print('Q6 Answer (% correct):', round(correct/len(train)*100, 2))

# mAP@3 helper
def ap3(actual, predicted):
    score, hits = 0.0, 0
    for i, p in enumerate(predicted[:3]):
        if p == actual:
            hits += 1
            score += hits/(i+1.0)
    return score

def map3(actuals, preds):
    return sum(ap3(a,p) for a,p in zip(actuals,preds)) / len(actuals)

# Q7 - ground truth C, pred C A B
print('Q7 Answer:', ap3('C', ['C','A','B']))

# Q8 - ground truth B, pred D B E
print('Q8 Answer:', ap3('B', ['D','B','E']))

# Q9 - Majority class baseline
top3 = freq.index[:3].tolist()
preds = [top3]*len(train)
print('Q9 Answer (majority baseline mAP@3):', round(map3(train['answer'].tolist(), preds), 4))

# Q10 - TF-IDF pipeline mAP@3
tfidf_preds = []
for _, row in train.iterrows():
    pv = tfidf.transform([str(row['prompt'])])
    scores = {opt: cosine_similarity(pv, tfidf.transform([str(row[opt])]))[0][0] for opt in options}
    tfidf_preds.append(sorted(scores, key=scores.get, reverse=True)[:3])
print('Q10 Answer (TF-IDF mAP@3):', round(map3(train['answer'].tolist(), tfidf_preds), 4))