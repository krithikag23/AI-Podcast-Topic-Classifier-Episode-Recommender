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
