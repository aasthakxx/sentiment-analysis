import streamlit as st
import pandas as pd
import nltk
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

etoi={'joy': 0, 'fear': 1, 'anger': 2, 'sadness': 3}
itoe={etoi[i]:i for i in etoi}


try:
    stop_words = list(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = list(stopwords.words('english'))


try:
    lem=WordNetLemmatizer()
    lem.lemmatize("hello")
except:
    nltk.download("wordnet")
    lem=WordNetLemmatizer()

def cleaning(x):
    new_text=[]
    for line in x:
        filtered=[]
        small=line.lower()
        list_of_words=small.split()
        for word in list_of_words:
            if word not in stop_words:
                filtered.append(word)
        filtered_string=""
        for i in filtered:
            filtered_string+=i+" "
        new_text.append(filtered_string)
    new_text2=[]
    for line in new_text:
        text=re.sub(r"[^A-Za-z]"," ",line)
        new_text2.append(text)
    new_text3=[]
    for line in new_text2:
        list1=[]
        s=line.split()
        for j in s:
            a=lem.lemmatize(j)
            list1.append(a)
        new_text3.append(" ".join(list1))
    return new_text3

if "log" not in st.session_state:
    with open("Boomer.pkl","rb") as f:
        log = pickle.load(f)
    st.session_state["log"]=log

if "vec" not in st.session_state:
    with  open("vector.pkl","rb") as v:
        vec= pickle.load(v)
    st.session_state["vec"]=vec


def predict(x):
    x=cleaning(x)
    vec_test_x=st.session_state["vec"].transform(x)
    output=st.session_state["log"].predict(vec_test_x)
    
    emotion=[]
    for i in output:
        emotion.append(itoe[i])
    return emotion



#app interface

st.title("Welcome to Boomer")
st.write("Sentiment analysis for your tweets.")
text=st.text_area("",placeholder="Enter text")
submit_button=st.button("Analyze")
if submit_button:
    prediction=predict([text])
    st.write(prediction[0])

_,col,_=st.columns([0.4,0.1,0.4])
with col:
    st.subheader("OR")


upload=st.file_uploader("Upload file (Please make sure that the column for Tweets are labelled as 'text'. )")
if upload is not None:
    data= pd.read_csv(upload)
    if "text" not in data:
        st.write("Error : Invalid Data.")
    else:
        text=data.text
        prediction=predict(text)
        data["label"]=prediction
        st.write(data)
        