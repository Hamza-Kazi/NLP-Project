import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    pickle_in = open('best_model.pkl', 'rb')
    classifier = pickle.load(pickle_in)

    st.title('Sarcasm Prediction')
    text = st.text_input("Please mention your text here")
    submit = st.button('Predict')

    text = [text]
    #st.write(text)
    
    #vectorizer = TfidfVectorizer()  
    #vect = vectorizer.fit_transform(text)

    #prediction = classifier.predict(vect)

    prediction = classifier.predict(text)


    if prediction == 1:
        st.write("Your text is sarcastic!")

    else:
        st.write("Your text is not sarcastic!")

main()
