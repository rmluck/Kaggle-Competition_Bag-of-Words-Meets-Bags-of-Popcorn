import pandas as pd
import numpy as np
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec

# PRIORITY IMPROVEMENTS:
# Try variants of data cleaning and preprocessing (punctuation, contractions, lemmatization, stemming, number replacement, domain-specific stopword list, etc.)
# Hyperparameter tuning (vector_size 100-300, min_count 10-100, window 5-15, epochs 10-20, sg 0/1, samplee 1e-5 to 1e-3, etc.)
# Use stratified K-fold cross validation to evaluate the model performance. Use confusion matrices and classification reports to analyze the model performance. Consider precision/recall.
# 1. Train classifier using pretrained embeddings
# 2. Instead of Random Forest, use boosting methods which outperform on high-dimensional features like vector averages or bag-of-centroids. Try XGBoost or LightGBM.
# 3. Instead of simple mean of word vectors, use TF-IDF weighted mean. Then during vector averaging, multiply each word vector by its TF-IDF weight.
# 4. Use gensim.models.doc2vec and TaggedDocument to train a Doc2Vec model. Then use document vectors for classification.
# 5. Use transformers from Hugging Face and fine-tune a BERT-based model on the dataset.
# 6. Try ensemble methods.


def clean_review(raw_review: str, stopwords: set, remove_stopwords: bool=False) -> list[str]:
    """
    Cleans the raw review text by removing HTML tags and markup, punctuation, numbers, and stopwords.

    Parameters:
        raw_review (str): The raw review text to be cleaned.
        stopwords (set): A set of stopwords to be removed from the review.
        remove_stopwords (bool): Whether to remove stopwords from the review. Default is False.

    Returns:
        list[str]: A list of cleaned words from the review.
    """

    # Remove HTML tags and markup from review
    review_text = BeautifulSoup(raw_review).get_text()

    # Remove punctuation and numbers from review
    review_text = re.sub("[^a-zA-Z]", " ", review_text)

    # Convert review to lowercase and split into words
    words = review_text.lower().split()

    # Remove stopwords from the list of words
    if remove_stopwords:
        words = [word for word in words if word not in stopwords]

    # Return the cleaned list of words
    return words


def split_review_into_sentences(review: str) -> list:
    """
    Splits a review into sentences.

    Parameters:
        review (str): The review text to be split into sentences.

    Returns:
        list: A list of sentences from the review.
    """

    # Use NLTK's sent_tokenize to split the review into sentences
    sentences = sent_tokenize(review)

    # Return the list of sentences
    return sentences


def clean_reviews(raw_reviews: pd.Series, model_type: str, stopwords: set,remove_stopwords: bool=False) -> list[str] | list[list[str]]:
    """
    Cleans a series of raw reviews.

    Parameters:
        raw_reviews (pd.Series): A pandas Series containing raw review texts.
        model_type (str): The type of model to be used.
        stopwords (set): A set of stopwords to be removed from the reviews.
        remove_stopwords (bool): Whether to remove stopwords from the reviews. Default is False.

    Returns:
        list[list[str]]: A list of lists, where each inner list contains cleaned words from a review.
    """

    # Iterate through the raw reviews and clean each one after splitting it into sentences
    cleaned_reviews = []
    for review in raw_reviews:
        # Clean reviews based on the model type
        if model_type == "bag_of_words":
            # Clean the entire review as a single string
            cleaned_reviews.append(clean_review(review, stopwords, remove_stopwords=remove_stopwords))
        elif model_type == "word2vec":
            # Split the review into sentences and clean each sentence
            sentences = split_review_into_sentences(review)
            
            cleaned_review_words = []
            for sentence in sentences:
                # Check if the sentence is not empty before cleaning
                if sentence.strip():
                    sentences = clean_review(sentence, stopwords, remove_stopwords=remove_stopwords)
                    cleaned_review_words.extend(sentences)
            cleaned_reviews.append(cleaned_review_words)

    # Return the list of cleaned reviews
    return cleaned_reviews


def bag_of_words(reviews: list, vectorizer: CountVectorizer) -> tuple[list, list]:
    """
    Creates a bag of words representation of the cleaned reviews.

    Parameters:
        reviews (list): A list of cleaned review strings.
        vectorizer (CountVectorizer): A CountVectorizer instance to convert the cleaned reviews into a bag of words representation.

    Returns:
        list: A bag of words representation of the reviews, where each review is represented as a vector of word counts.
        list: The vocabulary of the bag of words representation.
    """

    # Fit the vectorizer to the cleaned reviews and transform them into a bag of words representation
    bag_of_words = vectorizer.fit_transform(reviews).toarray()

    # Get the vocabulary of the bag of words representation
    vocabulary = vectorizer.get_feature_names_out()

    # Return the bag of words representation and the vocabulary
    return bag_of_words, vocabulary


def train_random_forest_classifier(training_inputs: list, target_values: pd.Series, folds: int=5) -> RandomForestClassifier:
    """
    Creates a Random Forest classifier and fits it to the training data and target values. Uses stratified K-fold cross-validation to evaluate the model performance.

    Parameters:
        model_type (str): The type of model being trained.
        training_inputs (list): The training data inputs, which can be a bag of words representation or feature vectors.
        target_values (pd.Series): The labels corresponding to the training inputs.
        folds (int): The number of folds for cross-validation. Default is 5.

    Returns:
        RandomForestClassifier: The trained Random Forest classifier.
    """

    # Create a StratifiedKFold instance for cross-validation
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    # Initialize a list to hold the accuracy scores for each fold
    accuracy_scores = []

    # Iterate through the folds and train the Random Forest classifier
    for fold_index, (train_index, val_index) in enumerate(skf.split(training_inputs, target_values)):
        # Split the training inputs and target values into training and validation sets
        X_train, X_val = training_inputs[train_index], training_inputs[val_index]
        y_train, y_val = target_values[train_index], target_values[val_index]

        # Create a Random Forest classifier with 100 trees
        forest = RandomForestClassifier(n_estimators=100, random_state=42)

        # Fit the Random Forest classifier to the training data and target values
        forest = forest.fit(X_train, y_train)

        # Predict the target values for the validation set
        y_pred = forest.predict(X_val)

        # Calculate the accuracy score for the current fold
        accuracy = accuracy_score(y_val, y_pred)
        accuracy_scores.append(accuracy)

        # Print classification report for this fold
        print(f"Fold {fold_index + 1} Classification Report:")
        print(classification_report(y_val, y_pred))
    
    print(f"Average Accuracy across {folds} folds: {np.mean(accuracy_scores):.4f}")

    # Predict the target values for the entire training set
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(training_inputs, target_values)

    # Return the trained Random Forest classifier
    return final_model


def make_feature_vector(words: list, model: Word2Vec, num_features: int) -> np.ndarray:
    """
    Calculates the average feature vector for a list of words using a Word2Vec model.
    
    Parameters:
        words (list): The list of words to be averaged.
        model (Word2Vec): The trained Word2Vec model.
        num_features (int): The dimensionality of the word vectors.

    Returns:
        np.ndarray: The average feature vector for the words.
    """

    # Initialize an empty feature vector of zeros
    feature_vector = np.zeros((num_features,), dtype="float32")

    # Build set that contains the words in the model's vocabulary
    index_to_word_set = set(model.wv.index_to_key)

    # Iterate through the words and add their vectors to the feature vector
    num_words = 0
    for word in words:
        if word in index_to_word_set:
            num_words += 1
            feature_vector = np.add(feature_vector, model.wv[word])

    # Divide the result by the number of words to get the average
    if num_words > 0:
        feature_vector = np.divide(feature_vector, num_words)

    # Return the average feature vector
    return feature_vector


def get_average_feature_vectors(reviews: list, model: Word2Vec, num_features: int) -> np.ndarray:
    """
    Calculates the average feature vector for each review.
    
    Parameters:
        reviews (list): The list of reviews, where each review is a list of words.
        model (Word2Vec): The trained Word2Vec model.
        num_features (int): The dimensionality of the word vectors.

    Returns:
        np.ndarray: An array of average feature vectors for each review.
    """

    # Initialize an empty list to hold the average feature vectors
    feature_vectors = []

    # Iterate through the reviews and calculate the average feature vector for each one
    for review in reviews:
        # Check if the review is not empty before calculating the feature vector
        if review:
            feature_vector = make_feature_vector(review, model, num_features)
            feature_vectors.append(feature_vector)

    # Convert the list of feature vectors to a NumPy array
    return np.array(feature_vectors)


def find_cluster_centers(model: Word2Vec, k: int) -> dict:
    """
    Finds the cluster centers for the Word2Vec model using KMeans clustering.

    Parameters:
        model (Word2Vec): The trained Word2Vec model.
        k (int): The number of clusters to form.

    Returns:
        dict: A dictionary mapping words to their corresponding cluster centers.
    """

    # Initialize a KMeans instance with the specified number of clusters
    kmeans = KMeans(n_clusters=k)

    # Fit the KMeans model to the Word2Vec vectors and predict the cluster centers
    cluster_centers = kmeans.fit_predict(model.wv.vectors)

    # Return a dictionary mapping words to their corresponding cluster centers
    return dict(zip(model.wv.index_to_key, cluster_centers))


def bag_of_centroids(reviews: list, cluster_centers: dict) -> list[list[int]]:
    """
    Creates a bag of centroids representation for each review.

    Parameters:
        reviews (list): The list of cleaned reviews, where each review is a list of words.
        cluster_centers (dict): The dictionary mapping words to their corresponding cluster centers.

    Returns:
        list[list[int]]: A list of bags of centroids, where each bag corresponds to a review and contains counts of each centroid.
    """

    # Initialize an empty list to hold the bags of centroids
    bags_of_centroids = []

    # Iterate through the reviews and create a bag of centroids for each one
    for review in reviews:
        # Initialize a bag of centroids with zeros for the review
        num_centroids = max(cluster_centers.values()) + 1
        bag_of_centroid = np.zeros(num_centroids, dtype="float32")

        # Iterate through the words in the review and update its bag of centroids
        for word in review:
            if word in cluster_centers:
                bag_of_centroid[cluster_centers[word]] += 1

        # Append the bag of centroids to the list
        bags_of_centroids.append(bag_of_centroid)

    # Return the list of bags of centroids
    return bags_of_centroids


def save_results_to_csv(results: list, test_ids: list, output_file: str):
    """
    Saves the predicted sentiment results to a CSV file.

    Parameters:
        results (list): The list of predicted sentiment labels.
        test_ids (list): The list of IDs corresponding to the test reviews.
        output_file (str): The path to the output CSV file.
    """

    # Create a DataFrame with the test IDs and predicted sentiment results
    results_df = pd.DataFrame(data={"id": test_ids, "sentiment": results})

    # Save the DataFrame to a CSV file
    results_df.to_csv(output_file, index=False, quoting=3)


if __name__ == "__main__":
    # Import the training data and test data
    print("Importing labeled training data...")
    training_data = pd.read_csv("data/labeled_train_data.tsv", header=0, delimiter="\t", quoting=3)
    print("Finished importing labeled training data. Importing unlabeled training data...")
    unlabeled_training_data = pd.read_csv("data/unlabeled_train_data.tsv", header=0, delimiter="\t", quoting=3)
    print("Finished importing unlabeled training data. Importing test data...")
    test_data = pd.read_csv("data/test_data.tsv", header=0, delimiter="\t", quoting=3)
    print("Finished importing test data.")

    # Download the NLTK stopwords and punkt tokenizer
    print("Downloading NLTK stopwords and punkt tokenizer...")
    nltk.download("stopwords")
    nltk.download("punkt_tab")
    stopwords = set(stopwords.words("english"))

    # Determine the model type ["bag_of_words", "word2vec"]
    model_type = "word2vec"

    # Clean the training reviews and test reviews based on the model type
    if model_type == "bag_of_words":
        print("Cleaning training reviews for bag of words model...")
        cleaned_training_reviews = clean_reviews(training_data["review"], model_type, stopwords, True)
        print("Finished cleaning training reviews. Cleaning test reviews for bag of words model...")
        cleaned_test_reviews = clean_reviews(test_data["review"], model_type, stopwords, True)
        print("Finished cleaning test reviews.")
    elif model_type == "word2vec":
        print("Cleaning labeled training reviews for word2vec model...")
        cleaned_labeled_training_reviews = clean_reviews(training_data["review"], model_type, stopwords, False)
        print("Finished cleaning labeled training reviews. Cleaning unlabeled training reviews for word2vec model...")
        cleaned_unlabeled_training_reviews = clean_reviews(unlabeled_training_data["review"], model_type, stopwords, False)
        print("Finished cleaning unlabeled training reviews. Cleaning test reviews for word2vec model...")
        cleaned_test_reviews = clean_reviews(test_data["review"], model_type, stopwords, False)
        print("Finished cleaning test reviews.")

    # Train the model on the cleaned training reviews and predict sentiment on the cleaned test reviews
    if model_type == "bag_of_words":
        # Create a CountVectorizer to convert the cleaned reviews into a bag of words representation
        print("Training bag of words model...")
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

        # Create a bag of words representation of the cleaned training reviews
        print("Creating bag of words representation for training reviews...")
        bag_of_words_representation, vocabulary = bag_of_words(cleaned_training_reviews, vectorizer)
        print("Finished creating bag of words representation for training reviews.")

        # Train the Random Forest classifier using the bag of words representation and the training labels
        print("Training Random Forest classifier...")
        random_forest_classifier = train_random_forest_classifier(bag_of_words_representation, training_data["sentiment"])
        print("Finished training Random Forest classifier.")

        # Predict the sentiment of the cleaned test reviews
        print("Predicting sentiment for test reviews...")
        input_samples = vectorizer.transform(cleaned_test_reviews).toarray()
        results = random_forest_classifier.predict(input_samples)
        print("Finished predicting sentiment for test reviews.")
    elif model_type == "word2vec":
        # Set values for Word2Vec parameters
        num_features = 300      # Word vector dimensionality
        min_word_count = 40     # Minimum word count to consider a word in the vocabulary
        num_workers = 4         # Number of worker threads to train the model
        context = 10            # Context window size
        downsampling = 1e-3     # Downsample setting for frequent words

        # Create a Word2Vec model using the cleaned training reviews
        print("Training Word2Vec model...")
        word2vec_model = Word2Vec(sentences=cleaned_labeled_training_reviews + cleaned_unlabeled_training_reviews, vector_size=num_features, min_count=min_word_count, workers=num_workers, window=context, sample=downsampling)
        word2vec_model.init_sims(replace=True)
        print("Finished training Word2Vec model.")

        # Save the Word2Vec model to a file
        print("Saving Word2Vec model...")
        word2vec_model.save("models/word2vec_model")
        print("Finished saving Word2Vec model.")

        # Get the average feature vectors for the cleaned training reviews
        print("Cleaning labeled training reviews for Word2Vec model...")
        cleaned_training_reviews = clean_reviews(training_data["review"], model_type, stopwords, True)
        print("Finished cleaning labeled training reviews.")

        # Determine the vector operation to use to combine word vectors ["vector_average", "cluster"]
        method = "cluster"
        if method == "vector_average":
            training_feature_vectors = get_average_feature_vectors(cleaned_training_reviews, word2vec_model, num_features)

            # Get the average feature vectors for the cleaned test reviews
            cleaned_test_reviews = clean_reviews(test_data["review"], model_type, stopwords, True)
            test_feature_vectors = get_average_feature_vectors(cleaned_test_reviews, word2vec_model, num_features)

            # Train the Random Forest classifier using the average feature vectors and the training labels
            random_forest_classifier = train_random_forest_classifier(training_feature_vectors, training_data["sentiment"])

            # Predict the sentiment of the cleaned test reviews
            results = random_forest_classifier.predict(test_feature_vectors)
        elif method == "cluster":
            # Set the number of clusters for KMeans
            num_clusters = word2vec_model.wv.vectors.shape[0] // 5

            # Find the cluster centers for the Word2Vec model
            print("Finding cluster centers for Word2Vec model...")
            cluster_centers = find_cluster_centers(word2vec_model, num_clusters)
            print("Finished finding cluster centers for Word2Vec model.")

            # Create a bag of centroids representation for the cleaned training reviews and cleaned test reviews
            print("Creating bag of centroids representation for training reviews...")
            training_bag_of_centroids_representation = bag_of_centroids(cleaned_training_reviews, cluster_centers)
            print("Finished creating bag of centroids representation for training reviews. Creating bag of centroids representation for test reviews...")
            test_bag_of_centroids_representation = bag_of_centroids(cleaned_test_reviews, cluster_centers)
            print("Finished creating bag of centroids representation for test reviews.")

            # Train the Random Forest classifier using the bag of centroids representation and the training labels
            print("Training Random Forest classifier with bag of centroids representation...")
            random_forest_classifier = train_random_forest_classifier(np.array(training_bag_of_centroids_representation), training_data["sentiment"])
            print("Finished training Random Forest classifier with bag of centroids representation.")

            # Predict the sentiment of the cleaned test reviews
            print("Predicting sentiment for test reviews using bag of centroids representation...")
            results = random_forest_classifier.predict(np.array(test_bag_of_centroids_representation))
            print("Finished predicting sentiment for test reviews using bag of centroids representation.")

    # Save the predicted sentiment results to a CSV file
    print("Saving results to CSV file...")
    if model_type == "bag_of_words":
        save_results_to_csv(results, test_data["id"], "output/bag_of_words_submission.csv")
    elif model_type == "word2vec":
        if method == "vector_average":
            save_results_to_csv(results, test_data["id"], "output/word2vec_vector_average_submission.csv")
        elif method == "cluster":
            save_results_to_csv(results, test_data["id"], "output/word2vec_cluster_submission.csv")
    print("Finished saving results to CSV file.")