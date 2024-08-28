import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Загружаем токенизатор и модель
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=10)

# Загружаем датасет
df = pd.read_csv("language_dataset.csv")
texts = df["text"].tolist()
labels = df["language"].tolist()
# print(labels)
# exit()
# Разбиваем датасет на тренировочный и валидационный
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Определяем класс датасета
class LanguageDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids)
        pad_len = self.max_len - len(input_ids)
        input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
        return {"input_ids": input_ids, "labels": label}


# Создаем тренировочный и валидационный датасеты
train_dataset = LanguageDataset(train_texts, train_labels, tokenizer, max_len=128)
val_dataset = LanguageDataset(val_texts, val_labels, tokenizer, max_len=128)

# Определяем параметры обучения
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Создаем тренер и обучаем модель
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# Определяем функцию для определения языка
def detect_language(text, model, tokenizer):
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    outputs = model(input_ids)
    probs = torch.softmax(outputs.logits, dim=1)
    predicted_lang = torch.argmax(probs)
    return predicted_lang.item()

# Тестируем модель
text = "Hello, how are you?"
predicted_lang = detect_language(text, model, tokenizer)
print(f"Predicted language: {predicted_lang}")
