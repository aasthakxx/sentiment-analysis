import nltk
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download("wordnet")


etoi={'joy': 0, 'fear': 1, 'anger': 2, 'sadness': 3}
itoe={etoi[i]:i for i in etoi}
stop_words = list(stopwords.words('english'))
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

with open("Boomer.pkl","rb") as f:
    log = pickle.load(f)
with  open("vector.pkl","rb") as v:
    vec= pickle.load(v)
    
  
def predict(x):
    x=cleaning(x)
    vec_test_x=vec.transform(x)
    output=log.predict(vec_test_x)
    
    emotion=[]
    for i in output:
        emotion.append(itoe[i])
    return emotion

if __name__=="__main__":
    s=input("Enter tweet : ")
    if s=="":
        print("No value found.")
    else:
        a=predict([s])
        print(a)