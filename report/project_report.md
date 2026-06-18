# Project Report: Tweet Sentiment Analysis

## Overview

This project builds a binary sentiment classifier for tweets. The goal is to classify each tweet as either positive or negative using text preprocessing, feature extraction, and supervised machine learning.

The implementation uses a linear Support Vector Machine trained on feature dictionaries produced from the tweet text. The final version combines TF-IDF weighting, unigram and bigram features, and opinion lexicon features.

## Data Input and Preprocessing

The dataset is stored as a tab-separated file. Each row contains a tweet identifier, a sentiment label, and the tweet text. The tweet identifier is excluded because it does not provide useful predictive information for sentiment classification.

Preprocessing includes:

- Converting text to lowercase
- Tokenising tweet text
- Removing punctuation in selected configurations
- Testing lemmatisation using WordNet
- Building unigram and bigram features
- Adding opinion lexicon based sentiment features

## Feature Extraction

Several feature extraction strategies were explored:

| Feature Strategy | Purpose |
|---|---|
| Binary features | Records whether a token appears in a tweet |
| Term frequency | Counts how often each token appears |
| Normalised term frequency | Divides token counts by total tweet length |
| TF-IDF | Weights terms by local frequency and corpus-level rarity |
| Opinion lexicon features | Adds positive and negative sentiment word counts |

The final approach used TF-IDF with an opinion lexicon. This performed best because TF-IDF reduced the influence of common terms while the lexicon added sentiment-specific signal.

## Model and Evaluation

The classifier uses a linear Support Vector Machine. The training process is evaluated with 10-fold cross-validation. In each fold, one section of the training data is used for validation while the remaining folds are used for training.

Weighted precision, recall, and F1 score are used because the dataset contains more positive tweets than negative tweets. Weighted metrics account for this class imbalance while still reporting a balanced view of performance.

## Error Analysis

The error analysis investigates:

- True positives and true negatives
- False positives and false negatives
- Confusion matrix patterns
- Class imbalance between positive and negative tweets
- Frequently occurring tokens in misclassified tweets

The analysis showed that the classifier was stronger at identifying positive sentiment than negative sentiment. Some errors were caused by sarcasm, informal language, swear words, named entities, and words that require wider context.

## Feature Engineering Findings

The experiments showed several useful patterns:

1. An n-gram range of `(1, 2)` performed best. Bigrams helped capture short contextual phrases, including negated sentiment.
2. Larger n-gram ranges reduced performance, likely due to increased sparsity.
3. Stop word removal often reduced performance because some stop words help preserve sentiment context.
4. Punctuation removal generally improved performance by reducing noise.
5. Normalising term frequency reduced performance, likely because tweets are already short.
6. The opinion lexicon produced the strongest improvement because it added sentiment-focused features.
7. Lemmatisation helped most feature configurations, but had limited impact when TF-IDF and opinion lexicon features were combined.

## Best Reported Result

The best configuration used TF-IDF with an opinion lexicon and an n-gram range of `(1, 2)`, without lemmatisation.

| Metric | Score |
|---|---:|
| Average Precision | 0.8785 |
| Average Recall | 0.8781 |
| Average F1 Score | 0.8781 |
| Average Accuracy | 0.8781 |

## Future Improvements

Potential improvements include:

- Shuffling the dataset with a controlled random seed
- Testing alternative stop word lists
- Removing URLs and hashtags more explicitly
- Adding emoji-based sentiment features
- Exploring additional sentiment lexicons
- Testing alternative classifiers
- Extending the task from binary sentiment to a graded sentiment scale
