import re
import unicodedata
import pandas as pd
import spacy
import contractions
from typing import Optional
from nltk.corpus import stopwords
import config

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
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_mentions(text)
        text = self.remove_hashtags(text)
        text = self.remove_html_tags(text)
        
        # Expand contractions (can't to cannot, I'm to I am)
        text = self.expand_contractions(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove accents
        if self.remove_accents_flag:
            text = unicodedata.normalize("NFD", text)
            text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        
        # Remove non-ASCII characters
        text = self.remove_non_ascii(text)

        # Aggressive cleaning: remove punctuation, numbers and special characters
        if self.aggressive:
            text = re.sub(r"[^a-z\s]", " ", text)
        
        # Normalize whitespace
        text = self.remove_extra_spaces(text)
        
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

    def remove_urls(self, text) -> str:
        return self.url_pattern.sub("", text)
    
    def remove_emails(self, text: str) -> str:
        return re.sub(self.email_pattern, '', text)
    
    def remove_html_tags(self, text) -> str:
        return self.html_pattern.sub("", text)

    def remove_mentions(self, text) -> str:
        return self.mention_pattern.sub("", text)

    def remove_hashtags(self, text) -> str:
        return self.hashtag_pattern.sub("", text)

    def expand_contractions(self, text) -> str:
        return contractions.fix(text)

    def remove_extra_spaces(self, text) -> str:
        return " ".join(text.split())
    
    def remove_non_ascii(self, text: str) -> str:
        return "".join(c for c in text if ord(c) < 128)

def get_text_statistics(df: pd.DataFrame, text_column: str) -> dict:
    lengths = df[text_column].str.len()
    word_counts = df[text_column].str.split().str.len()
    
    return {
        'num_texts': len(df),
        'avg_length': lengths.mean(),
        'max_length': lengths.max(),
        'min_length': lengths.min(),
        'avg_words': word_counts.mean(),
        'max_words': word_counts.max(),
        'min_words': word_counts.min(),
    }