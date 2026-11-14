import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Ensure required NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text_pipeline(text):
    """
    Consolidated text preprocessing pipeline:
    1. Lowercases the input text.
    2. Tokenizes the text into individual words.
    3. Removes standard English stopwords.
    4. Removes punctuation and special non-alphanumeric characters.
    5. Applies Snowball stemming and WordNet lemmatization.
    6. Strips numerical digits.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text_lower = text.lower()
    
    # 2. Tokenize
    tokens = word_tokenize(text_lower)
    
    # 3. Stopwords removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Re-join to apply regex and morphological processors
    temp_text = " ".join(filtered_tokens)
    
    # 4. Remove punctuation & symbols
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', temp_text)
    
    # 5. Stemming and Lemmatization
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    
    processed_tokens = []
    for word in word_tokenize(clean_text):
        stemmed = stemmer.stem(word)
        lemmatized = lemmatizer.get_lemma_classifier(stemmed) if hasattr(lemmatizer, 'get_lemma_classifier') else lemmatizer.lemmatize(stemmed)
        processed_tokens.append(lemmatized)
        
    processed_text = " ".join(processed_tokens)
    
    # 6. Remove numbers
    final_text = re.sub(r'\d+', '', processed_text)
    
    # Normalize white spaces
    return " ".join(final_text.split())


def get_message_vector(text, word2vec_model):
    tokens = word_tokenize(text)
    vectors = [
        word2vec_model.wv[token]
        for token in tokens
        if token in word2vec_model.wv
    ]

    if not vectors:
        return np.zeros(word2vec_model.wv.vector_size)

    return np.mean(vectors, axis=0)