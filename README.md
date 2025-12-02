# 🎧 AI Podcast Topic Classifier Episode Recommender
This application leverages Natural Language Processing (NLP) to analyze podcast episode descriptions and automatically determine their topic category.
Using similarity analysis, it also recommends the Top-3 most relevant podcast episodes for users to explore further.
---
## 🚀 Key Features
- **Topic Classification** using ML
- **Top-3 Podcast Recommendations**
- **Prediction Confidence Chart**
- Works in real-time with user input
---

## 🧠How it Works
- User inputs a podcast episode description.
- The text is transformed into vector form using TF-IDF.
- Topic is predicted using a trained Logistic Regression classifier.
- Similar episodes are fetched using cosine similarity scores.
- Results are visualized and recommendations are shown instantly.
---
## Tech Stack
- Streamlit UI
- TF-IDF Vectorizer (Scikit-learn)
- Logistic Regression
- Cosine Similarity for Recommendations
- Plotly Visualizations

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

