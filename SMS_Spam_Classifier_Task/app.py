import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the vectorizer and naive model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Transform_text function for text preprocessing
import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):

    text = text.lower() # Converting to lowercase
    text = nltk.word_tokenize(text) # Tokenize

    # Removing special characters and retaining alphanumeric words
    text = [word for word in text if word.isalnum()]
    
    # Removing stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    # Applying stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# Streamlit code
st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter message:")

if st.button('Predict'):

    # Preprocess input message
    transformed_sms = transform_text(input_sms)
    
    # Vectorize preprocessed message
    vector_input = tfidf.transform([transformed_sms])
    
    # Predict result
    result = model.predict(vector_input)[0]
    
    # Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")