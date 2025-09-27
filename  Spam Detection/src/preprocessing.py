import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_stem(self, text):
        """Tokenize and stem text"""
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.tokenize_and_stem(text)
        return text

class FeatureExtractor:
    def __init__(self, method='tfidf', max_features=5000):
        self.method = method
        self.max_features = max_features
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words='english'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words='english'
            )
    
    def fit_transform(self, texts):
        """Fit vectorizer and transform texts"""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer"""
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names from vectorizer"""
        return self.vectorizer.get_feature_names_out()