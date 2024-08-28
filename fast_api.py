from fastapi import FastAPI
from pydantic import BaseModel

import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import random
from datetime import datetime

app = FastAPI()

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#if torch.cuda.is_available() else "cpu") #Если на NVIDIA + CUDA "cuda" if torch.cuda.is_available() else 
model_path='model.pth'
model = torch.load(model_path, map_location=device).to(device).eval()

class Text_mod(BaseModel):
  letter: str

@app.post("/model/")
async def detect_language(text_class:Text_mod):
    text = text_class.letter
    # Токенизируем текст и преобразуем его в тензоры
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

    # Вычисляем вероятности языков
    with torch.no_grad():
        input_ids = input_ids.to(device)
        outputs = model(input_ids)
        probs = torch.softmax(outputs.logits, dim=1)

    # Получаем язык с наибольшей вероятностью
    predicted_lang = torch.argmax(probs)

    # Преобразуем индекс языка в его название
    lang_names = ['ru', 'en', 'tr']#['af', 'ar', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hr', 'hu', 'id', 'it', 'ja', 'jv', 'ka', 'kk', 'ko', 'lt', 'lv', 'mk', 'ml', 'mr', 'ms', 'my', 'nl', 'no', 'or', 'pa', 'pl', 'pt', 'ro', 'ru', 'si', 'sk', 'sl', 'so', 'sq', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'yo', 'zh']
    return {"letter": text, "predicted_lang": lang_names[predicted_lang.item()]}

class Item(BaseModel):
  name: str  
  description: str | None = None


# app = FastAPI()

class User(BaseModel):
   username: str
   letter: str

@app.post("/home/")
async def detect_len(us:User):
  # print(letter)
  return {"message": "catch", "user" :us}


@app.get("/home/")
def read_items():
    return {"message": "get"}


# def post_api(l_t, l_l)
#   print("Efficiency(test): " + str(((test(l_t, l_l) * 10000)//1)/100))#texts, labels
# print("Efficiency(trn): " + str(((test(texts, labels) * 10000)//1)/100))#texts, labels
# print(datetime.now() - start_time)
    # lang_names = ['ru', 'en', 'tr']