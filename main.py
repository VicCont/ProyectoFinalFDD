import pyspark 
from pyspark.sql import SparkSession
import re
import os
from logit import Logit
import string
import pandas as pd
import json
import pickle
import numpy as np
import hashlib

def save_model(dir_model,model):
    pickle.dump(model, open('model.pkl','wb'))


def load_model(dir_model):
    if (os.path.isfile(dir_model)):
        return pickle.load(open('model.pkl','rb'))
    return None


def get_training_data(dir_training_data,stop_words):
    if not (os.path.isfile(dir_training_data)):
        f=open("dev-v2.0.json",encoding="utf-8")
        data=json.load(f)
        f.close()
        json_data=[]
        for grupo in data["data"]:
            for paragraph in grupo["paragraphs"]:
                qas=[x for x in paragraph["qas"]]
                dict_aux={
                    "hash_contexto":hashlib.sha256(paragraph["context"].encode('utf-8')).hexdigest(),
                    "context":[x+'.' for x in paragraph["context"][:-1].lower().split(".")],
                    "questions": [x["question"].lower() for x in qas],
                    #"ans":[x["answers"][0]['text'].lower() if len (x["answers"]) else None for x in qas],
                    "index_respuesta" : [x["answers"][0]['answer_start'] if len (x["answers"]) else None for x in qas]
                }
                json_data.append(dict_aux)
        training_data=pd.DataFrame.from_dict(json_data)
        training_data.to_pickle(dir_training_data)
        del json_data

        return training_data
    else:
        return pd.read_pickle(dir_training_data)

def get_stop_words(direccion_stopwords):
    if not (os.path.isfile(direccion_stopwords)):
        stop_words=pd.read_csv("NLTK's20of20stopwords.txt",header=None)
        stop_words.to_pickle(direccion_stopwords)
        return stop_words
    else:
        return pd.read_pickle(direccion_stopwords)
direccion_stopwords="stopword.pickle"
dir_training_data="training_data.pickle"
pattern=re.compile(r"\s{2.}")
stop_words=get_stop_words(direccion_stopwords)
stop_words=set(stop_words[0].to_list())
training_data=get_training_data(dir_training_data,stop_words)
prepare_strings=lambda x: set(re.sub(r" +", r" ", x.translate(str.maketrans('', '', string.punctuation))).split())
training_data=training_data.assign(context_jaccard=lambda df: list((list([prepare_strings(z)-stop_words for z in y]) for y in (x for x in df["context"].array))))
training_data=training_data.assign(questions_jaccard=lambda df: list((list([prepare_strings(z)-stop_words  for z in y]) for y in (x for x in df["questions"].array))))
training_data=training_data.explode(["context","context_jaccard"])
training_data=training_data[training_data["context_jaccard"] != set()]
training_data["len_cont"]= training_data["context"].str.len()
training_data["suma_acumulada"]=training_data.groupby(['hash_contexto'])["len_cont"].cumsum()-training_data['len_cont']
training_data["suma_posterior"]=training_data.groupby('hash_contexto')['suma_acumulada'].shift(-1)
training_data=training_data.drop(['len_cont',"context",'questions'],axis=1)
training_data=training_data.explode(['questions_jaccard','index_respuesta'])
training_data=training_data[training_data["index_respuesta"].notna()]
training_data['index_respuesta'] = training_data['index_respuesta'].astype('int')
training_data['responds_to_question']=(training_data.suma_acumulada<=training_data.index_respuesta) & (((training_data.suma_posterior>training_data.index_respuesta) | (training_data.suma_posterior.isnull())) )
training_data=training_data.drop(['suma_acumulada','suma_posterior','index_respuesta','hash_contexto'],axis=1)
training_data=training_data.assign(jaccard=lambda df: list(len(x.intersection(y))/len(x.union(y)) for x,y in zip(df['questions_jaccard'],df['context_jaccard'])))#training_data.set_index([['questions','index_respuesta','hash_contexto']])
training_data=training_data.drop(['context_jaccard','questions_jaccard'],axis=1)

#spark = SparkSession.builder.appName('Proyecto2FDD.com').getOrCreate()
#spark.sparkContext.setLogLevel('ERROR')
#training_data=pd.concat([suma_acumulada.set_index(['hash_contexto','context']),suma_acumulada.set_index(['hash_contexto','context'])], axis=1, join='inner')
## training_data["context"].str.replace(f"[{string.punctuation}]*",'',regex=True) quita puntuacion

modelo =load_model("model.pkl")
if (not modelo):
    modelo=Logit(training_data["jaccard"].to_numpy().reshape(-1, 1) ,training_data["responds_to_question"].to_numpy().reshape(-1, 1) )
    modelo.train()
    save_model('model.pkl', modelo)

input()
pregunta=input("Ingresa tu preguta: ")
pregunta=pregunta.lower()
pregunta=re.sub(pattern, r" ", pregunta)
oraciones=pregunta.split(".")
oraciones=[x.strip().split(" ") for x in oraciones]
print(oraciones)
