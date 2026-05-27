# MLOps Assignment 2 - Goodreads Genre Classification

Fine-tuned **DistilBERT** model for multi-class book genre classification using Hugging Face Transformers, trained on Kaggle GPU with experiment tracking using Weights & Biases (W&B).

## Results

| Metric       | Score    |
|--------------|----------|
| Accuracy     | 0.2125   |
| F1 Score     | 0.1707   |
| Eval Loss    | 3.2107   |

## Links

- **Kaggle Notebook**: [https://www.kaggle.com/code/g25ait2106/ml-ops-assignment-2](https://www.kaggle.com/code/g25ait2106/ml-ops-assignment-2)
- **Hugging Face Model**: [https://huggingface.co/BhavikSolanki/ML-OPS-Assignment2](https://huggingface.co/BhavikSolanki/ML-OPS-Assignment2)
- **W&B Dashboard**: [View Run](https://wandb.ai/bhaviksolanki-indian-institute-of-technology-jodhpur/mlops-assignment2/runs/a77d3jdv) *(Click to see training metrics, loss curves, and system stats)*

## Project Overview

This project demonstrates a complete **MLOps pipeline**:
- Loading pre-trained model from Hugging Face Hub
- Data preprocessing and tokenization
- Fine-tuning using Hugging Face Trainer
- Experiment tracking with Weights & Biases
- Model evaluation and artifact logging
- Deployment to Hugging Face Model Hub

## Model Details

- **Base Model**: `distilbert-base-uncased`
- **Task**: Text Classification (Book Genres)
- **Training Platform**: Kaggle (GPU T4)
- **Epochs**: 3
- **Batch Size**: 16
- **Dataset**: Sampled UCSD Goodreads Reviews

## Setup Instructions

```bash
pip install transformers datasets wandb scikit-learn torch