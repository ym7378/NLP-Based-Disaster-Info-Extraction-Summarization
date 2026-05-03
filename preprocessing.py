import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load dataset
df = pd.read_csv("disaster_raw_data.csv")

print("Columns in dataset:", df.columns)

# Identify possible text columns
possible_text_cols = ['title', 'content', 'description', 'text', 'news']

existing_text_cols = [col for col in possible_text_cols if col in df.columns]

if not existing_text_cols:
    print("No expected text columns found. Using all columns except date/source.")
    existing_text_cols = [col for col in df.columns if col.lower() not in ['date', 'source']]

# Fill missing values
for col in existing_text_cols:
    df[col] = df[col].fillna("")

# Combine text columns
df['combined_text'] = df[existing_text_cols].agg(" ".join, axis=1)

# Convert to lowercase
df['combined_text'] = df['combined_text'].str.lower()

# Remove punctuation and numbers
df['combined_text'] = df['combined_text'].apply(
    lambda x: re.sub(r'[^a-zA-Z\s]', '', x)
)

# Remove extra spaces
df['combined_text'] = df['combined_text'].apply(
    lambda x: re.sub(r'\s+', ' ', x).strip()
)

# Stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

df['cleaned_text'] = df['combined_text'].apply(clean_text)

# Save cleaned dataset
df.to_csv("disaster_cleaned_data.csv", index=False)

print("\nPreprocessing completed successfully!")
print("Cleaned dataset saved as disaster_cleaned_data.csv")
print("Cleaned dataset saved as disaster_cleaned_data.csv")