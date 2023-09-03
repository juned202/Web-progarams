from importlib.resources import path
from lib2to3.pgen2.tokenize import tokenize
from gensim.parsing.preprocessing import preprocess_string
import gradio as gr
from bs4 import BeautifulSoup
import nltk
import os
import sklearn
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from deep_translator import GoogleTranslator
from langdetect import detect
# from deep_translator import single_detection
import pandas as pd
import numpy as np
title = "Query Search"

LANGUAGES = ["en", "hn", "gj"]
path = r"C:\Users\juned\Downloads\business\\"

files = os.listdir(path)
Num = len(files) 

# for x in range(Num): 
#     fp = open(path + files[x],"r")
#     soup = BeautifulSoup(fp.read(), features="html.parser")
#     temp = soup.find("text").text
#     fw = open(path + files[x],"w")
#     fw.write(temp)

doc_preprocess = []
for x in range(Num):
    fb= open(path + files[x],"r")
    text=fb.read()
    doc_preprocess.append(preprocess_string(text))

cbow = Word2Vec(sentences=doc_preprocess, vector_size=100, window=5, min_count=5, workers=4, sg=0)

def translate(lang, text):
    lang_detect = detect(text)
    # print(lang_detect)
    # print(lang)
    if lang_detect == 'hi':
        text=GoogleTranslator(source='auto', target='en').translate(text)
    elif lang_detect =='gu':
       text= GoogleTranslator(source='auto', target='en').translate(text)
    else:
        text=text
    # print(text)
    preprocessing = preprocess_string(text)
    dvec = np.zeros((len(doc_preprocess),100))
    qvec = np.zeros((1,100))
    for n in range(len(doc_preprocess)):
        for token in doc_preprocess[n]:
            try:
                dvec[n,:] += cbow.wv[token]
            except:
                dvec[n,:] += np.zeros((100,))
    
    for n in range(len(doc_preprocess)):
        dvec[n,:] = dvec[n,:]/len(doc_preprocess[n])

    for token in preprocessing:
        try:
            qvec += cbow.wv[token]
        except:
            qvec += np.zeros((100,))
    for n in range(len(preprocessing)):
        qvec = qvec/len(preprocessing)

    doc_no = 0
    cosine_doc = {}
   
    for doc in dvec:
        # d_vec = doc.reshape(1,-1).T
        # q_vec = qvec.reshape(1,-1)
        # print(doc.shape)
        # print(qvec[0].shape)
        cosine = np.dot(doc,qvec[0])/(np.linalg.norm(doc)*np.linalg.norm(qvec[0]))
        cosine_doc[doc_no] = cosine
        doc_no += 1
    # final_cosine = dict(sorted(cosine_doc.items(), key=lambda x: x[1], reverse=True))
    # cosine_lst.append(final_cosine)
    d =  max(zip(cosine_doc.values(), cosine_doc.keys()))[1]
    # if text == "Indian Oil":
    # print(d)
    if lang == "en":
        fb= open(path + files[d],"r")
        text=fb.read()
        return text
    elif lang == "hn":
        fb= open(path + files[d],"r")
        text=fb.read()
        translated = GoogleTranslator(source='auto', target='hi').translate(text)
        return translated
    elif lang == "gj":
        fb= open(path + files[d],"r")
        text=fb.read()
        translated = GoogleTranslator(source='auto', target='gu').translate(text)
        return translated
    else:
        text="ERROR"
        return text
    # else:
    #     if lang == "en":
    #         fb= open(path + files[25],"r")
    #         text=fb.read()
    #         return text
    #     elif lang == "hn":
    #         return text
    #     elif lang == "gj":
    #         return text

# def eng(title, temp):
#     return "Hello " + title + "!!"

# def hind(query):
#     return "Hello in Hindi " + query + "!!"

# def guj(query):
#     return "Hello in Gujarati " + query + "!!"

iface = gr.Interface(fn=translate, inputs=[gr.inputs.Radio(LANGUAGES), gr.inputs.Textbox(
    lines=7, label="Enter Query")], title="Cross-Lingual Information Retrieval", outputs="text")
iface.launch()