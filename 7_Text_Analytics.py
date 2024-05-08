# Importing necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample document
document = "Text analytics is the process of analyzing unstructured text data and extracting meaningful insights from it."

# Tokenization
tokens = word_tokenize(document)

# Part-of-speech tagging (POS tagging)
pos_tags = pos_tag(tokens)

# Stopwords removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# Stemming
porter = PorterStemmer()
stemmed_tokens = [porter.stem(token) for token in filtered_tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

# Print preprocessed tokens
print("Original Tokens:", tokens)
print("POS Tags:", pos_tags)
print("Filtered Tokens (Stopwords removed):", filtered_tokens)
print("Stemmed Tokens:", stemmed_tokens)
print("Lemmatized Tokens:", lemmatized_tokens)

# Create TF-IDF representation of the document
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([document])

# Print TF-IDF representation
print("\nTF-IDF Representation:")
print(tfidf_matrix.toarray())
print("Feature Names:", tfidf_vectorizer.get_feature_names())
