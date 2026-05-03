import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load cleaned dataset
df = pd.read_csv("disaster_cleaned_data.csv")

# Store extracted entities
all_entities = []

for index, row in df.iterrows():
    text = str(row['cleaned_text'])
    doc = nlp(text)
    
    for ent in doc.ents:
        all_entities.append({
            "original_text": row['text'],
            "entity": ent.text,
            "entity_label": ent.label_,
            "source": row['source'],
            "date": row['date']
        })

# Convert to DataFrame
entities_df = pd.DataFrame(all_entities)

# Save to CSV
entities_df.to_csv("disaster_entities.csv", index=False)

print("NER extraction completed successfully!")
print("Entities saved as disaster_entities.csv")