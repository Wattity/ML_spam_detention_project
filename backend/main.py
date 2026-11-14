from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd 
import numpy as np
import nltk
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.stem import SnowballStemmer,WordNetLemmatizer
from preprocessing import preprocess_text_pipeline
from preprocessing import get_message_vector
from gensim.models import Word2Vec




app=FastAPI()
class MESSAGE(BaseModel):
    message:str

#load the saved model
with open ('spam_model.joblib',"rb") as f:
    model=joblib.load(f)

#load the save wordvec:
word2_vec=Word2Vec.load("spam_word2vec.model")

@app.get("/home")
def home():
    return {"message":"Welcome to the spam detection API"}

@app.post("/predict" )
def predict_message(message:MESSAGE):
    #process the input message
    preprocessed_message=preprocess_text_pipeline(message)
    #get the word vector
    message_vec=get_message_vector(preprocessed_message,word2_vec)

    #make prediction
    prediction = model.predict(message_vec.reshape(1,-1))
    return {"prediction":"Spam" if prediction[0] == 0 else "Not spam"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",host="127.0.0.1",port=8000,reload=True)
