# Tweet Sentiment Analysis

A machine learning project for classifying tweet sentiment using TF-IDF features, bigrams, an opinion lexicon, and a linear Support Vector Machine classifier.

## Project Overview

This project explores supervised sentiment classification on short social media text. The notebook walks through the full machine learning workflow: data loading, preprocessing, feature extraction, cross-validation, error analysis, and final evaluation.

The final approach combines:

- TF-IDF feature weighting
- Unigram and bigram features
- Opinion lexicon features
- Text normalisation
- Weighted evaluation metrics
- Linear SVM classification

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── sentiment-dataset.tsv
├── tweet_sentiment_analysis.ipynb
└── report/
    └── project_report.md
```

## Technologies Used

- Python
- NumPy
- NLTK
- scikit-learn
- Matplotlib
- Jupyter Notebook

## Methodology

The project uses a traditional NLP pipeline:

1. Load tab-separated tweet sentiment data.
2. Extract the tweet text and sentiment label.
3. Apply preprocessing and text normalisation.
4. Convert tweets into feature vectors using TF-IDF.
5. Add opinion lexicon based features.
6. Train a linear SVM classifier.
7. Evaluate performance using 10-fold cross-validation.
8. Analyse false positives, false negatives, and confusion matrix outputs.
9. Evaluate the final model on a held-out test set.

## Key Findings

The strongest configuration used TF-IDF with an opinion lexicon and an n-gram range of `(1, 2)`. Unigrams and bigrams helped capture individual sentiment words and short contextual phrases, such as negated expressions.

The opinion lexicon provided a noticeable improvement because it added sentiment-focused features that are especially useful for short tweets. Stop word removal reduced performance in some cases, likely because sentiment can depend on words such as negations.

Best reported weighted scores:

| Metric | Score |
|---|---:|
| Precision | 0.8785 |
| Recall | 0.8781 |
| F1 Score | 0.8781 |
| Accuracy | 0.8781 |

## Required Data Files

The dataset file `sentiment-dataset.tsv` is included in the repository root because the notebook reads it directly from that location.

The notebook also expects the following opinion lexicon files in the repository root:

```text
positive-words.txt
negative-words.txt
```

These files are not bundled here because they were not included with the project upload. Add them to the repository root before running the notebook.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook tweet_sentiment_analysis.ipynb
```

Then run the notebook cells in order.

## Notes

The notebook downloads the NLTK WordNet resource when it is first run. An internet connection may be required the first time this resource is downloaded.
