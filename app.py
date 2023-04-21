import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import time



ps = PorterStemmer()


def transform_text(text):
    text = text.lower()  # lowercase text
    text = nltk.word_tokenize(text)  # to split text into list of words

    y = []
    for i in text:
        if i.isalnum():  # to remove special characters (to keep alpha numeric characters)
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Text Spam Detection')

input_sms = st.text_area('Enter the text')

if st.button('Predict'):

    with st.spinner('Wait for it...'):
        time.sleep(2)
    # st.success('Done!')

    # 1. preprocess
    transform_sms = transform_text(input_sms)

    # 2. vectorize
    vector_input = tfidf.transform([transform_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. display
    if result == 1:
        st.header(':red[Spam]')
    else:
        st.header(':green[Not Spam]')