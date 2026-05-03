# NLP-Based-Disaster-Info-Extraction-Summarization
NLP-based system for extracting disaster-related information and generating automated summaries from multi-source text data like social media and news.
# NLP-Based-Disaster-Info-Extraction-Summarization
NLP-based system for extracting disaster-related information and generating automated summaries from multi-source text data like social media and news. 
# 🌍 DisasterNLP: Information Extraction & Summarization System

## 🚀 Overview

This project implements an **NLP-based intelligent system** that extracts disaster-related information and generates summaries from multi-source text data.

The system processes data from **Reddit and News sources**, extracts key entities, classifies messages, assigns priority, and generates concise summaries for disaster response.

---

## 📂 Project Structure

```bash
Project/


│── data_collection.py           # Collect data (Reddit + RSS)
│── preprocessing.py             # Text cleaning
│── ner_extraction.py            # Named Entity Recognition
│── entity_analysis.py           # Analyzing entities
│── classification.py            # Category + Priority
│── summarization.py             # Text summarization
│── dashboard.py                 # Visualization (Streamlit)



│── requirements.txt             # Dependencies
```

---

## 🔄 Complete Workflow

### 🔹 1. Data Collection

* Sources:

  * Reddit (citizen reports)
  * News RSS feeds
* Keywords:

  ```
  flood, earthquake, cyclone, landslide, fire, collapse, rescue, trapped
  ```

---

### 🔹 2. Data Preprocessing

Performed using:

* NLTK
* SpaCy
* Regex

Steps:

* Lowercasing
* Removing URLs, hashtags, emojis
* Stopword removal
* Tokenization

➡ Output: `disaster_cleaned_data.csv`

---

### 🔹 3. Information Extraction (NER)

Using **SpaCy**

Extracts:

* Location
* Disaster type
* Casualties
* Damage

➡ Output: `disaster_entities.csv`

---

### 🔹 4. Classification

Categories:

* Emergency Request
* Damage Report
* General Information
* Relief Update

Also assigns:

* 🔥 High Priority
* ⚠ Medium Priority
* 🟢 Low Priority

➡ Output: `classified_disaster_data.csv`

---

### 🔹 5. Summarization

Using **Transformer Models (BART/T5)**

* Summarizes high-priority messages
* Reduces information overload

➡ Output: `disaster_summarized_data.csv`

---

### 🔹 6. Visualization & Dashboard

Built using:

* Streamlit
* Matplotlib

Displays:

* Entity distribution
* Top locations
* Top organizations
* Final summarized insights

---

## 🧠 Tech Stack

* Python
* NLTK
* SpaCy
* Scikit-learn
* Transformers (BART/T5)
* Pandas
* Streamlit

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install models

```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords
```

### 3. Run pipeline

```bash
python data_collection.py
python preprocessing.py
python ner_extraction.py
python classification.py
python summarization.py
```

### 4. Run dashboard

```bash
streamlit run dashboard.py
```

---

## 📊 Outputs

* Cleaned dataset
* Extracted entities
* Classified messages with priority
* Summarized disaster reports
* Visualization graphs

---

## 💡 Key Features

✔ Multi-source data collection
✔ End-to-end NLP pipeline
✔ Entity extraction (NER)
✔ Disaster classification
✔ Priority detection
✔ Transformer-based summarization
✔ Interactive dashboard

---

## ⚠️ Limitations

* Rule-based classification
* Limited real-time capability
* English-only dataset

---

## 🚀 Future Enhancements

* BERT-based classification
* Real-time streaming data
* Multilingual support
* GIS integration

---

## 👩‍💻 Authors

* Sabana Asmi G
* Yuvashree M

---
