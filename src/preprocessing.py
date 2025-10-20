import re
import unicodedata
import pandas as pd
import spacy
import contractions
from typing import Optional
from nltk.corpus import stopwords

# Load English stopwords from NLTK
import nltk
nltk.download('stopwords')
EN_STOPWORDS = set(stopwords.words('english'))

# Load spaCy English model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

class TextPreprocessor:
    """
    Text preprocessing pipeline optimized for the Fake vs Real Text Kaggle competition.
    
    Features:
    - Remove URLs, emails, mentions, hashtags, HTML tags
    - Expand contractions
    - Lowercase and accent removal
    - Stopwords removal
    - Lemmatization with spaCy
    - Aggressive cleaning: remove punctuation and numbers
    """
    
    # Regex patterns to clean common noise
    url_pattern = re.compile(r"http[s]?://\S+")
    email_pattern = re.compile(r"\S+@\S+")
    mention_pattern = re.compile(r"@\w+")
    hashtag_pattern = re.compile(r"#\w+")
    html_pattern = re.compile(r"<[^>]+>")
    
    # Aggressive parameter removes punctuation and numbers making text cleaner for TF-IDF
    # or vector-based models
    def __init__(self, lowercase: bool = True, remove_accents: bool = True,
                 remove_stopwords: bool = True, aggressive: bool = False):
        self.lowercase = lowercase
        self.remove_accents_flag = remove_accents
        self.remove_stopwords = remove_stopwords
        self.aggressive = aggressive

    # -> str means the function clean_text output is expected as str
    # It's a hint type from PEP 484
    def clean_text(self, text: str) -> str:
        """Full preprocessing pipeline for a single text."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs, emails, mentions, hashtags, HTML
        text = self.url_pattern.sub("", text)
        text = self.email_pattern.sub("", text)
        text = self.mention_pattern.sub("", text)
        text = self.hashtag_pattern.sub("", text)
        text = self.html_pattern.sub("", text)
        
        # Expand contractions (can't to cannot, I'm to I am)
        text = contractions.fix(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove accents
        if self.remove_accents_flag:
            text = unicodedata.normalize("NFD", text)
            text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        
        # Aggressive cleaning: remove punctuation, numbers and special characters
        if self.aggressive:
            text = re.sub(r"[^a-z\s]", " ", text)
        
        # Normalize whitespace
        text = " ".join(text.split())
        
        # Lemmatization + stopwords removal
        doc = nlp(text)
        tokens = [
            token.lemma_ for token in doc
            if (not self.remove_stopwords or token.lemma_ not in EN_STOPWORDS)
        ]
        
        return " ".join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Apply preprocessing to a full DataFrame column."""
        df = df.copy()
        df[text_column] = df[text_column].apply(self.clean_text)
        return df
