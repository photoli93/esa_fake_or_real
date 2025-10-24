import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join('..', 'src')))
import config

# Define Custom Dataset Class
class TextPairDataset(Dataset):
    """
    A custom PyTorch Dataset for the chunked data.
    """
    def __init__(self, encodings, labels):
        # encodings contains 'input_ids' and 'attention_mask'
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # item is a dictionary of tensors for the model
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # labels is the target class, converted to a tensor
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Tokenization Function 
def tokenize_data(df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int):
    """
    Tokenizes the exploded DataFrame, which has one text chunk per row, 
    and prepares it for DistilBERT training.

    Args:
        df: The exploded DataFrame containing 'text_chunk' and 'label' columns.
        tokenizer: The pre-trained DistilBERT tokenizer
        max_length: The maximum sequence length (for pre-trained DistilBert : max_length=512).

    Returns:
        A TextPairDataset object ready for use with a DataLoader
    """

    # Use the 'text_chunk' column containing the pre-chunked text strings
    texts = df['text_chunk'].tolist()

    # Tokenize the texts. Return 'input_ids' and 'attention_mask'
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt' # Return PyTorch tensors
    )

    # Use the 'label' column containing the binary 0/1 label
    try:
        labels = df['label'].tolist()
    except KeyError:
        raise KeyError(
            "The 'label' column is missing from the DataFrame. "
            "Please ensure you have created the binary label (0/1) in your "
            "preprocessing notebook (02_data_preprocessing.ipynb) before saving the exploded CSVs."
        )

    # 4. Return the custom dataset object
    return TextPairDataset(encodings, labels)

# Main
if __name__ == "__main__":
    print("--- Starting Dataset Creation and Tokenization (03_dataset_creation.py) ---")

    # Define file paths based on config
    TRAIN_CLEANED_FILE = config.PROCESSED_DATA_DIR / "train_exploded.csv"
    VAL_CLEANED_FILE = config.PROCESSED_DATA_DIR / "val_exploded.csv"
    
    TOKENIZED_TRAIN_PATH = config.PROCESSED_DATA_DIR / "tokenized_train.pt"
    TOKENIZED_VAL_PATH = config.PROCESSED_DATA_DIR / "tokenized_val.pt"

    # Instantiate Tokenizer
    print(f"Loading tokenizer: {config.TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    max_len = config.MAX_SEQ_LENGTH

    # Process Training Data
    print(f"Loading training data from: {TRAIN_CLEANED_FILE}")
    try:
        train_df = pd.read_csv(TRAIN_CLEANED_FILE)
    except FileNotFoundError:
        print(f"Error: Training data not found at {TRAIN_CLEANED_FILE}. Please run 02_data_preprocessing.ipynb first.")
        sys.exit(1)
        
    print(f"Tokenizing training data (Max Length: {max_len})...")
    train_dataset = tokenize_data(train_df, tokenizer, max_len)
    
    # Save the tokenized dataset
    torch.save(train_dataset, TOKENIZED_TRAIN_PATH)
    print(f"Successfully created and saved tokenized training dataset to: {TOKENIZED_TRAIN_PATH}")
    print(f"Dataset size: {len(train_dataset)}")
    
    # Process Validation Data
    print(f"\nLoading validation data from: {VAL_CLEANED_FILE}")
    try:
        val_df = pd.read_csv(VAL_CLEANED_FILE)
    except FileNotFoundError:
        print(f"Warning: Validation data not found at {VAL_CLEANED_FILE}. Skipping validation tokenization.")
        val_df = None

    if val_df is not None:
        print(f"Tokenizing validation data (Max Length: {max_len})...")
        val_dataset = tokenize_data(val_df, tokenizer, max_len)
        
        # Save the tokenized dataset
        torch.save(val_dataset, TOKENIZED_VAL_PATH)
        print(f"Successfully created and saved tokenized validation dataset to: {TOKENIZED_VAL_PATH}")
        print(f"Dataset size: {len(val_dataset)}")
    
    print("\nDatasets creation complete")
