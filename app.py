from bert_embedding import BertEmbedding
#from bert_serving.client import BertClient
from flask import Flask, render_template, request
import os
import json
import requests
import pickle
import joblib
import numpy as np
import pandas as pd
#import tensorflow as tf
#all packages 
import nltk 
import string 
import re
import random
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from textblob.sentiments import *
import re
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

app = Flask(__name__)
app.static_folder = 'static'

sent_bertphrase_embeddings = joblib.load('model/questionembedding.dump')
sent_bertphrase_ans_embeddings = joblib.load('model/ansembedding.dump')

stop_w = stopwords.words('english')
#bc = BertClient(ip='localhost')
df = pd.read_csv("model/20200325_counsel_chat.csv",encoding="utf-8")

def get_embeddings(texts):
    url = '127.0.0.1:8000/encode'  # Add /encode to the URL
    headers = {
        'content-type': 'application/json'
    }
    data = {
        "id": 123,
        "texts": texts,
        "is_tokenized": False
    }
    data = json.dumps(data)
    try:
        r = requests.post("http://" + url, data=data, headers=headers)
        print("Embedding server status:", r.status_code)
        print("Embedding server response:", r.text)
        r.raise_for_status()
        # Flatten the embeddings to ensure they are 2D
        embeddings = r.json().get('result', [])
        return np.array(embeddings).squeeze(axis=1)  # Remove the extra dimension
    except Exception as e:
        print("Error in get_embeddings:", e)
        return None

def predictor(userText):
    data = [userText]
    x_try = pd.DataFrame(data,columns=['text'])
    clean('text',x_try,stopwords=True)
    for index,row in x_try.iterrows():
        question = row['text']
        question_embedding = get_embeddings([question])
        if question_embedding is None:
            return "Sorry, I couldn't process your request right now."
        return retrieveAndPrintFAQAnswer(question_embedding,sent_bertphrase_embeddings,df)

def clean(column,df,stopwords=False):
  df[column] = df[column].apply(str)
  df[column] = df[column].str.lower().str.split()
  #remove stop words
  if stopwords:
    df[column]=df[column].apply(lambda x: [item for item in x if item not in stop_w])
  #remove punctuation
  df[column]=df[column].apply(lambda x: [item for item in x if item not in string.punctuation])
  df[column]=df[column].apply(lambda x: " ".join(x))

def retrieveAndPrintFAQAnswer(question_embedding, sentence_embeddings, FAQdf):
    max_sim = -1
    index_sim = -1
    valid_ans = []
    for index, faq_embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(faq_embedding, question_embedding)[0][0]
        print(f"Question {index}: Similarity = {sim}")  # Debug print
        if sim >= max_sim:
            max_sim = sim
            index_sim = index
            valid_ans.append(index_sim)

    # Calculate answer with the highest cosine similarity
    max_a_sim = -1
    answer = ""
    for ans in valid_ans:
        answer_text = FAQdf.iloc[ans, 8]  # Answer text
        answer_em = sent_bertphrase_ans_embeddings[ans]  # Answer embedding
        similarity = cosine_similarity(answer_em, question_embedding)[0][0]
        print(f"Answer {ans}: Similarity = {similarity}")  # Debug print
        if similarity > max_a_sim:
            max_a_sim = similarity
            answer = answer_text

    if max_a_sim == -1:  # If no valid answer is found
        return "I'm sorry, I couldn't find an exact answer, but I'm here to help!"
    return answer

def retrieve(sent_bertphrase_embeddings,example_query): # USE ONLY QUESTION/ANSWER EMBEDDINGS CS
    max_=-1
    max_i = -1
    for index,emb in enumerate(sent_bertphrase_embeddings):
        sim_score = cosine_similarity(emb,example_query)[0][0]
        if sim_score>max_:
            max_=sim_score
            max_i=index
    #print("\n");
    #print("Retrieved: "+str(max_)+" ",df.iloc[max_i,3])  # 3 is index for q text
    #print(df.iloc[max_i,8])    # 8 is the index for the answer text  
    return str(df.iloc[max_i,8])

def clean_text(greetings):
    greetings = greetings.lower()
    greetings = ' '.join(word.strip(string.punctuation) for word in greetings.split())
    re.sub(r'\W+', '',greetings)
    greetings = lmtzr.lemmatize(greetings)
    return greetings

def predictor(userText):
    data = [userText]
    x_try = pd.DataFrame(data,columns=['text'])
    #clean the user query
    clean('text',x_try,stopwords=True)
    
    for index,row in x_try.iterrows():
        question = row['text']
        question_embedding = get_embeddings([question])
        #question_embedding = bc.encode([question])
        return retrieveAndPrintFAQAnswer(question_embedding,sent_bertphrase_embeddings,df)
    
greetings = ['hi','hey', 'hello', 'heyy', 'hi', 'hey', 'good evening', 'good morning', 'good afternoon', 'good', 'fine', 'okay', 'great', 'could be better', 'not so great', 'very well thanks', 'fine and you', "i'm doing well", 'pleasure to meet you', 'hi whatsup']
happy_emotions = ['i feel good', 'life is good', 'life is great', "i've had a wonderful day", "i'm doing good"]
goodbyes = ['thank you', 'thank you', 'yes bye', 'bye', 'thanks and bye', 'ok thanks bye', 'goodbye', 'see ya later', 'alright thanks bye', "that's all bye", 'nice talking with you', 'i’ve gotta go', 'i’m off', 'good night', 'see ya', 'see ya later', 'catch ya later', 'adios', 'talk to you later', 'bye bye', 'all right then', 'thanks', 'thank you', 'thx', 'thx bye', 'thnks', 'thank u for ur help', 'many thanks', 'you saved my day', 'thanks a bunch', "i can't thank you enough", "you're great", 'thanks a ton', 'grateful for your help', 'i owe you one', 'thanks a million', 'really appreciate your help', 'no', 'no goodbye']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    cleanText = clean_text(str(userText))
    #check sentiment 
    blob = TextBlob(userText, analyzer=PatternAnalyzer())
    polarity = blob.sentiment.polarity

    if cleanText in greetings:
        return "Hello! How may I help you today?"
    elif polarity>0.7:
        return "That's great! Do you still have any questions for me?"
    elif cleanText in happy_emotions:
        return "That's great! Do you still have any questions for me?"  
    elif cleanText in goodbyes:
        return "Hope I was able to help you today! Take care, bye!"
    topic = predictor(userText)
    #res = random.choice(dictionary[topic])
    return topic

if __name__ == "__main__":
    app.run() 
