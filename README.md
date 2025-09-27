# Spam Detection ML Project

A machine learning project for detecting spam in emails and SMS messages using various classification algorithms.

## Features

- Email spam detection
- SMS spam detection
- Multiple ML algorithms (Naive Bayes, SVM, Random Forest)
- Text preprocessing and feature extraction
- Model evaluation and comparison
- Web interface for testing

## Project Structure

```
spam-detection/
├── data/                 # Dataset files
├── src/                  # Source code
│   ├── preprocessing.py  # Text preprocessing utilities
│   ├── models.py        # ML model implementations
│   ├── train.py         # Training script
│   └── predict.py       # Prediction utilities
├── notebooks/           # Jupyter notebooks for exploration
├── models/              # Saved trained models
├── requirements.txt     # Python dependencies
└── app.py              # Flask web application
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train models:
```bash
python src/train.py
```

3. Run web app:
```bash
python app.py
```

## Dataset

The project uses the SMS Spam Collection dataset and Enron email dataset for training and evaluation.
