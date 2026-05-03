import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
print("Loading classified dataset...")

df = pd.read_csv("classified_disaster_data.csv")

print("Dataset loaded. Total records:", len(df))

# Make sure column exists
if "cleaned_text" not in df.columns:
    raise ValueError("Column 'cleaned_text' not found in CSV file.")

# -----------------------------
# 2️⃣ Load Faster Summarization Model
# -----------------------------
print("Loading summarization model...")

model_name = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model loaded successfully on", device)

# -----------------------------
# 3️⃣ Generate Summaries (Batch Processing = Faster)
# -----------------------------
summaries = []
batch_size = 8   # Increase to 16 if system has good RAM

texts = df["cleaned_text"].astype(str).tolist()

for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i + batch_size]

    # Limit extremely long inputs
    batch_texts = [text[:1000] for text in batch_texts]

    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=60,
            min_length=20,
            do_sample=False
        )

    batch_summaries = tokenizer.batch_decode(
        summary_ids,
        skip_special_tokens=True
    )

    summaries.extend(batch_summaries)

# -----------------------------
# 4️⃣ Save Output
# -----------------------------
df["Summary"] = summaries

output_file = "disaster_summarized_data.csv"
df.to_csv(output_file, index=False)

print("\nSummarization completed successfully!")
print("Saved as:", output_file)
