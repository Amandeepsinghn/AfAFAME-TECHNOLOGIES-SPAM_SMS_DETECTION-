import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the model and vectorizer
filename = 'MultinomialNB.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)

vectorizer_filename = 'CountVectorizer.pkl'
with open(vectorizer_filename, 'rb') as file:
    cv = pickle.load(file)

# Initialize Porter Stemmer
ps = PorterStemmer()

# Streamlit UI
st.title('SMS Spam Detection')

# User input
user_input = st.text_area("Enter your message:")

if st.button('Predict'):
    if user_input:
        # Preprocess the input
        text = re.sub(r"[^a-zA-Z]", " ", user_input)
        text = text.lower()
        text = text.split()
        text = [ps.stem(word) for word in text if word not in stopwords.words("english")]
        text = " ".join(text)

        # Transform input
        input_vector = cv.transform([text]).toarray()

        # Predict
        prediction = model.predict(input_vector)

        # Display result
        if prediction[0] == 1:
            st.write("The message is classified as **Spam**")
        else:
            st.write("The message is classified as **Not Spam**")
    else:
        st.write("Please enter a message to classify.")
