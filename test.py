import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import random
from datetime import datetime
import numpy as nn
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#if torch.cuda.is_available() else "cpu") #Если на NVIDIA + CUDA "cuda" if torch.cuda.is_available() else 
model_path='model.pth'
model = torch.load(model_path, map_location=device).to(device).eval()
# file_w_n = open("file_w_n.txt", "w+")
metric_list = list() #[[None], [None], [None], [None], [None]]

def detect_language(text):
    # Токенизируем текст и преобразуем его в тензоры
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

    # Вычисляем вероятности языков
    with torch.no_grad():
        input_ids = input_ids.to(device)
        outputs = model(input_ids)
        probs = torch.softmax(outputs.logits, dim=1)

    # Получаем язык с наибольшей вероятностью
    predicted_lang = torch.argmax(probs)
    model_confidence = max(nn.array(probs)[0]) - sorted([nn.array(probs)[0][0], nn.array(probs)[0][1], nn.array(probs)[0][2]])[1]

    #-----------------
    metric_list[-1].append(nn.array(probs)[0][0])# + " " + str(nn.array(probs)[0][1]) + " " + str(nn.array(probs)[0][2]) + "\n")
    metric_list[-1].append(nn.array(probs)[0][1])
    metric_list[-1].append(nn.array(probs)[0][2])
    #-----------------

    # Преобразуем индекс языка в его название
    lang_names = ['ru', 'en', 'tr']#['af', 'ar', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hr', 'hu', 'id', 'it', 'ja', 'jv', 'ka', 'kk', 'ko', 'lt', 'lv', 'mk', 'ml', 'mr', 'ms', 'my', 'nl', 'no', 'or', 'pa', 'pl', 'pt', 'ro', 'ru', 'si', 'sk', 'sl', 'so', 'sq', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'yo', 'zh']
    return predicted_lang.item(), model_confidence

def data_test(path):
  file_v = open(path).read().split("\n") # "data_test/trn/test1_0.txt"
  text_list, label_list = list(), list()
  for i in file_v:
      label_list.append(i[-1])
      text_list.append(i[:min(len(i) - 2, 511)])
  return text_list, label_list

def test(list_text, list_lab):
  start_time = datetime.now()
  file_error = open("file_error.txt", "w+")
  print("start:", start_time)
  list_success = list()
  n = 0
  for i in range(len(list_text)):
    metric_list.append([list_text[i]])#, int(list_lab[i]))
    metric_list[-1].append(int(list_lab[i]))
    output_lang, confidence_lang = detect_language(list_text[i])
    # print(metric_list)
    # if i == 2:
    #   exit()
    # print(output_lang)
    if int(output_lang) == int(list_lab[i]):
      list_success.append(1)
    else:
      list_success.append(0)
      # print(confidence_lang)
      file_error.write(list_text[i] + " " + str(list_lab[i]) + " " + str(output_lang) + " " + str(confidence_lang) + "\n")
    # if confidence_lang * 100 < 20:
    #    print(confidence_lang)
    if (i / len(list_text) * 100) >= n:
      print(str(i / len(list_text)*100) +  "% " + str(datetime.now() - start_time))
      n += 10
    # print(i / len(list_text) * 100 >= 10, n)
  file_error.close()
  print("100%", datetime.now() - start_time)
  return sum(list_success)/len(list_success)

def visualize_metric_values(metric_values):
    plt.figure(figsize=(16, 8))
    plt.plot(metric_values, label='Значение метрики на нейроне')
    plt.xlabel('Номер')
    plt.ylabel('Значение нейрона')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

l_t, l_l = data_test("data_txt/test/trans_3.txt")#data_test("data_txt/test/test1_0.txt") Nn_for_bot/Nn_for_bot/data_txt/trn/trn.txt data_txt/test/real.tx
print("Efficiency(test): " + str(((test(l_t, l_l) * 10000)//1)/100))#texts, labels

# visualize_metric_values(metric_list[4])

df = pd.DataFrame(metric_list)
df.to_pickle('pred.pkl')

# file_w_n.close()
# print("Efficiency(trn): " + str(((test(texts, labels) * 10000)//1)/100))#texts, labels
# print(datetime.now() - start_time)
    # lang_names = ['ru', 'en', 'tr']