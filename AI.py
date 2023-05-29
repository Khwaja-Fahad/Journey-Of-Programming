import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.classify import NaiveBayesClassifier

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('punkt')

# Preprocessing function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

# Feature extraction function
def extract_features(tokens):
    return dict([(token, True) for token in tokens])

# Training data
train_data = [
    ("I love this product!", "positive"),
    ("This is the worst experience ever.", "negative"),
    ("The movie was okay.", "neutral"),
    # Add more labeled examples here
]

# Preprocess and extract features from training data
train_features = [(extract_features(preprocess_text(text)), label) for (text, label) in train_data]

# Train the Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_features)

# Test data
test_data = [
    "I hate it.",
    "It's fantastic!",
    "I'm not sure how I feel about this.",
    # Add more test examples here
]

# Preprocess and extract features from test data
test_features = [extract_features(preprocess_text(text)) for text in test_data]

# Predict sentiments for test data
predictions = [classifier.classify(features) for features in test_features]

# Print the predictions
for text, sentiment in zip(test_data, predictions):
    print(f"Text: {text}\nSentiment: {sentiment}\n")




'''Project Name: Sentiment Analyzer
Description: The Sentiment Analyzer is an AI-based project that analyzes the sentiment of text data, helping you understand the emotional tone behind a piece of text. It utilizes natural language processing (NLP) techniques and machine learning algorithms to classify text into positive, negative, or neutral sentiments.

The project involves the following steps:

Data Collection: Gather a dataset of labeled text data with sentiment annotations. This dataset can include social media posts, customer reviews, or any other text data with associated sentiment labels.

Data Preprocessing: Clean and preprocess the text data by removing unnecessary characters, converting to lowercase, and handling special cases like stemming or lemmatization. Split the dataset into training and testing subsets.

Feature Extraction: Utilize techniques such as bag-of-words, word embeddings (e.g., Word2Vec or GloVe), or TF-IDF to represent the text data as numerical feature vectors.

Model Training: Train a machine learning model, such as a Naive Bayes classifier, logistic regression, or a deep learning model (e.g., recurrent neural network or transformer), using the training dataset and the extracted features.

Model Evaluation: Evaluate the trained model using the testing dataset and performance metrics like accuracy, precision, recall, and F1 score. Fine-tune the model if necessary.

Sentiment Prediction: Build an interface that allows users to input text, and the trained model predicts the sentiment (positive, negative, or neutral) of that text.

By implementing this project, you'll gain hands-on experience in NLP, machine learning, and building AI-based applications. You can also expand the project by incorporating more advanced techniques, such as sentiment analysis for multilingual text or sentiment analysis for specific domains like movie reviews or product feedback. '''




