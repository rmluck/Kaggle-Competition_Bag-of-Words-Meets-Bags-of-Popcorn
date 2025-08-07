# IMDB Review Sentiment Classifier

*By Rohan Mistry - Last updated August 7, 2025*

---

## 📖 Overview

Natural language processing pipeline developed to classify IMDB movie reviews as positive or negative based on review sentiment analysis. Built for the Bag of Words Meets Bags of Popcorn Kaggle competition. Implemented text preprocessing, Google's Word2Vec embeddings, and deep learning classifiers such as Random Forest and XGBoost to achieve 86% cross-validated accuracy.

**Dataset**: Used the competition-provided IMDB movie review dataset containing 100,000 total multi-paragraph reviews (25,000 for labeled training, 50,000 for unlabeled training, and 25,000 for testing). Reviews are raw text with binary sentiment labels (IMDB rating < 5 is equivalent to sentiment score of 0, IMDB rating >= 7 is equivalent to sentiment score of 1). See the [competition page](https://www.kaggle.com/competitions/word2vec-nlp-tutorial) for full competition and dataset details.

---

## 📁 Contents

```bash
├── data/                       # Raw training and test data
├── models/                     # Pre-saved models
├── notebooks/                  # Tutorial notebooks
├── outputs/                    # Submission files
├── src/
│   └── deep_learning_model.py  # Deep learning model script
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## 🌟 Features

* **Text Preprocessing**: HTML tag removal, non-letter character filtering, stopword removal, tokenization.
* **Sentiment Analysis**: Implemented TF-IDF weighting and lemmatization for text.
* **Modeling**: Three modeling methods: bags of words, average vector operations, and centroid clustering.
* **Classifiers**: Random Forest baseline. XGBoost classifier with cross-validation and hyperparameter tuning. Early stopping to reduce overfitting.
* **Evaluation**: Stratified K-Fold cross-validation with classification reports for each fold.
* **Results**: Achieved ~86% accuracy with tuned XGBoost classifier.

---

## 🚧 Future Improvements

* Experiment with LightGBM and CatBoost for further performance gains.
* Test other deep learning models (e.g., LSTM, BERT) on the same dataset.
* Apply dimensionality reduction (e.g., Truncated SVD) to TF-IDF features to further reduce overfitting.
* Ensemble multiple models for potential accuracy boost.
* Deploy as a simple web application for interactive predictions.
* Analyze and address error cases (e.g., sarcasm detection, domain-specific vocabulary).

---

## 🧰 Tech Stack

* Python, Jupyter notebook
* **Text Preprocessing and Data Analysis**: `pandas`, `numpy`
* **Sentiment Analysis**:  `nltk`, `gensim` Word2Vec
* **Deep Learning Modeling**: `scikit-learn`, `xgboost`

## 🙏 Contributions / Acknowledgements

This project was built independently for the Kaggle competition.

**Citations**

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). "Learning Word Vectors for Sentiment Analysis." The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011). [ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf](http://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf)
