import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import re
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and punctuation, and apply stemming
    tokens = [ps.stem(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)

tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
Model = pickle.load(open('model.pkl', 'rb'))

st.title("Spam Email/SMS Classifier")
input_txt = st.text_area("Enter the message")

if st.button("Predict"):
    processed_txt = preprocess_text(input_txt)
    vectorized_txt = tfidf_vectorizer.transform([processed_txt]).toarray()
    num_char = np.array([[len(input_txt)]])  # shape (1, 1)
    final_features = np.hstack((vectorized_txt, num_char))
    prediction = Model.predict(final_features)[0]
    st.write("Spam" if prediction == 1 else "Not Spam")