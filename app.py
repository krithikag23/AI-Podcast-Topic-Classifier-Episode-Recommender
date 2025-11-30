import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

st.set_page_config(page_title="AI Podcast Topic Classifier", layout="wide")
st.title("🎧 AI-Powered Podcast Topic Classifier & Recommender")

st.write("Type a podcast episode summary and let AI detect the topic + recommend similar episodes!")

# Load dataset
@st.cache_resource
def load_dataset():
    categories = [
        'sci.space', 'comp.graphics', 'rec.sport.baseball',
        'talk.politics.misc', 'sci.med', 'business'
    ]
    data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    return data

data = load_dataset()

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data.data)
y = data.target

# Train model
model = LogisticRegression(max_iter=300)
model.fit(X, y)

categories = data.target_names


# User input
episode_text = st.text_area("Paste a podcast transcript/summary:", 
                            placeholder="Example: Elon Musk launches rocket to Mars...")

btn = st.button("Predict Topic")

if btn and episode_text.strip():
    vector = vectorizer.transform([episode_text])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0]

    st.success(f"Detected topic: **{categories[pred]}**")

    # Probabilities bar chart
    df_prob = pd.DataFrame({"Topic": categories, "Probability": prob})
    fig = px.bar(df_prob, x="Topic", y="Probability", title="Prediction Confidence")
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # Recommendation engine
    sim_scores = cosine_similarity(vector, X).flatten()
    top_indices = sim_scores.argsort()[::-1][:3]

    st.subheader("🎯 Recommended Similar Podcast Episodes")
    for idx in top_indices:
        st.write(f"- Episode {idx} (Topic: **{categories[y[idx]]}**)")  # simplified display

else:
    st.info("Enter a podcast description and click Predict!")