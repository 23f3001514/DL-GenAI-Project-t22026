# Deep Learning & GenAI Project
## MCQ Solver using Deep Learning & NLP

---

## Student Details
- **Name:** ROHAN KUMAR
- **Roll Number:** 23f3001514
- **Term:** T2 2026
- **Course:** Deep Learning & Generative AI (BS Data Science)
- **Kaggle:** rohankumar1818
- **W&B Project:** https://wandb.ai/23f3001514-instituition/23f3001514-t22026

---

## Problem Statement
Build ML/AI systems capable of solving complex multiple choice questions.
Each question has 5 options (A-E). Task is to predict top 3 most likely
correct answers ranked by confidence. Evaluated using MAP@3 metric.

---

## Models Built
| Model | Type | Val Accuracy | Kaggle Score |
|---|---|---|---|
| Logistic Regression + TF-IDF | Classical ML | ~99% | 0.74064 |
| TextCNN | From Scratch | 100% | 0.73815 |
| TextCRNN (CNN+LSTM) | From Scratch | 99.5% | 0.75270 |
| DistilBERT | Pretrained Transformer | 100% | 0.74688 |

---

## Best Kaggle Score: 0.75270 (CRNN Model)

---

## Repository Structure

- notebooks/ → Jupyter/Kaggle notebooks
- src/ → Training and inference scripts
- scripts/ → Utility scripts
- data/ → Data samples or references
- models/ → Saved model checkpoints
- reports/ → Project report
- requirements.txt
- README.md

---

## Tools Used
- **Kaggle** — Competition + inference
- **GitHub** — Code + version control
- **Weights & Biases** — Experiment tracking
- **Hugging Face** — Pretrained models (DistilBERT)
- **PyTorch** — Deep learning framework

---

## Milestones
| Milestone | Status | Description |
|---|---|---|
| M0 | Done | Registration and Setup |
| M1 | Done | EDA + Baseline |
| M2 | Done | Classical ML |
| M3 | Done | CNN from scratch |
| M4 | Done | CRNN CNN+LSTM |
| M5 | Done | DistilBERT Fine-tuning |

---

## Setup Instructions
Run pip install -r requirements.txt to install all dependencies.
