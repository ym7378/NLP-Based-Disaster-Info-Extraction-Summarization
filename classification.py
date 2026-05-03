import pandas as pd
import os

# 1️ Check current directory files
print("Files in current directory:", os.listdir())

# 2️ Load your cleaned dataset
# Make sure this filename matches your actual cleaned CSV
df = pd.read_csv("disaster_cleaned_data.csv")

print("Dataset loaded successfully!")
print("First 5 rows:")
print(df.head())

# 3️ Define keywords for categories
category_keywords = {
    "Emergency Request": ["help", "trapped", "rescue", "urgent", "need assistance", "evacuate"],
    "Damage Report": ["collapsed", "destroyed", "damaged", "flooded", "burnt", "wreckage"],
    "General Information": ["update", "information", "report", "status", "alert", "forecast"],
    "Relief Update": ["relief", "aid", "supplies", "donation", "volunteers", "assistance"]
}

# Keywords for priority levels
priority_keywords = {
    "High": ["trapped", "urgent", "immediate", "rescue", "critical"],
    "Medium": ["injured", "damaged", "flooded", "burnt", "blocked"],
    "Low": ["report", "information", "update", "forecast"]
}

# 4️ Classification function
def classify_message(text):
    text_lower = str(text).lower()  # convert to string and lowercase
    category = "General Information"
    priority = "Low"

    # Determine category
    for cat, words in category_keywords.items():
        if any(word in text_lower for word in words):
            category = cat
            break

    # Determine priority
    for prio, words in priority_keywords.items():
        if any(word in text_lower for word in words):
            priority = prio
            break

    return category, priority

# 5️ Apply classification to the dataframe
df[['Category', 'Priority']] = df['cleaned_text'].apply(lambda x: pd.Series(classify_message(x)))

# 6️ Save the updated dataframe
output_file = "classified_disaster_data.csv"
df.to_csv(output_file, index=False)

print(f"Classification completed! Saved to {output_file}")

# 7️ Quick summary
print("Category counts:")
print(df['Category'].value_counts())
print("Priority counts:")
print(df['Priority'].value_counts())