import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer


filename = 'best_model.pkl'
pickle_in = open(filename, 'rb')
classifier = pickle.load(pickle_in)

st.title('Sarcasm Prediction')
text = st.text_input("Please mention your text here")
submit = st.button('Predict')

vect = TfidfVectorizer(text)

prediction = classifier.predict(vect)

if prediction == 1:
    st.write("Your text is sarcastic!")

else:
    st.write("Your text is not sarcastic!")
