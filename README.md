# MLOps Assignment 2 - Goodreads Genre Classification

Fine-tuned **DistilBERT** model for book genre classification using Hugging Face Transformers on Kaggle GPU.

## Results

| Metric       | Score    |
|--------------|----------|
| Accuracy     | 0.2625   |
| F1 Score     | 0.2249   |
| Eval Loss    | 3.2795   |

## Links

- **Kaggle Notebook**: [https://www.kaggle.com/code/g25ait2106/ml-ops-assignment-2](https://www.kaggle.com/code/g25ait2106/ml-ops-assignment-2)
- **Hugging Face Model**: [https://huggingface.co/BhavikSolanki/ML-OPS-Assignment2](https://huggingface.co/BhavikSolanki/ML-OPS-Assignment2)
- **W&B Dashboard**: Not available (W&B tracking had authentication issues in Kaggle environment)

## Project Overview

This project demonstrates a complete MLOps pipeline:
- Loading pre-trained model from Hugging Face
- Fine-tuning on Kaggle GPU
- Model evaluation
- Deployment to Hugging Face Hub

## Setup Instructions

```bash
pip install transformers datasets scikit-learn torch wandb
