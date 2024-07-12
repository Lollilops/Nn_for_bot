from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')

import torch

def detect_language(text):
    # Токенизируем текст и преобразуем его в тензоры
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

    # Вычисляем вероятности языков
    with torch.no_grad():
        outputs = model(input_ids)
        probs = torch.softmax(outputs.logits, dim=1)

    # Получаем язык с наибольшей вероятностью
    predicted_lang = torch.argmax(probs)

    # Преобразуем индекс языка в его название
    lang_names = ['af', 'ar', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hr', 'hu', 'id', 'it', 'ja', 'jv', 'ka', 'kk', 'ko', 'lt', 'lv', 'mk', 'ml', 'mr', 'ms', 'my', 'nl', 'no', 'or', 'pa', 'pl', 'pt', 'ro', 'ru', 'si', 'sk', 'sl', 'so', 'sq', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'yo', 'zh']
    return lang_names[predicted_lang.item()]

text = "Hello, how are you?"
print(detect_language(text))  # Выведет: en

text = "Bonjour, comment ça va?"
print(detect_language(text))  # Выведет: fr
