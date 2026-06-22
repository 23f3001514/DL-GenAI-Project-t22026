# Deep Learning & GenAI Project
## Smart MCQ Solver Challenge

**Student:** Rohan Kumar  
**Roll Number:** 23f3001514  
**Term:** T2 2026  
**Course:** Deep Learning & Generative AI (BS Data Science)  
**Kaggle:** rohankumar1818  
**W&B Project:** https://wandb.ai/23f3001514-instituition/23f3001514-t22026  

---

## Problem Statement

Build ML/AI systems capable of solving complex multiple choice questions.  
Each question has 5 options (A to E). The task is to predict the **top 3 most likely correct answers** in ranked order.  
Evaluation metric: **Mean Average Precision at 3 (mAP@3)**

Higher score when correct answer appears earlier in predictions:
- Correct answer at rank 1 → score = 1.000
- Correct answer at rank 2 → score = 0.500  
- Correct answer at rank 3 → score = 0.333

---

## Dataset

| Split | Rows | Description |
|---|---|---|
| Train | 2000 | Questions with ground truth answers |
| Test  | 500  | Questions to predict top 3 answers |

Each row has: `id`, `prompt`, `A`, `B`, `C`, `D`, `E`, `answer`

---

## Repository Structure

```
DL-GenAI-Project-t22026/
├── notebooks/          → Jupyter/Kaggle notebooks
├── src/                → Helper modules and utilities
├── scripts/            → Training and inference scripts
├── data/               → Data references (not actual data)
├── models/             → Saved model checkpoints
├── reports/            → Project report
├── requirements.txt    → All dependencies
└── README.md
```

---



## Tools Used

- **Kaggle** — Competition platform and GPU for training
- **GitHub** — Version control and code hosting
- **Weights & Biases** — Experiment tracking and visualization
- **HuggingFace** — Pretrained transformer models
- **PyTorch** — Deep learning framework
- **scikit-learn** — Classical ML models

---

## Progress

Work in progress — updating as project develops.