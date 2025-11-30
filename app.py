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