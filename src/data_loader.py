import os
import json
import re
import numpy as np
import pandas as pd

SELECTED_INTENTS = [
    'reminder',
    'smart_home',
    'directions',
    'play_music',
    'weather',
    'alarm',
    'calendar',
    'translate'
]

DATA_DIR = "dataset"


def load_data():
    json_path = os.path.join(DATA_DIR, "data_full.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def records_to_df(records):
        return pd.DataFrame(records, columns=['text', 'intent'])

    train_df = records_to_df(data['train'])
    val_df   = records_to_df(data['val'])
    test_df  = records_to_df(data['test'])

    print("Колонки train:", train_df.columns.tolist())
    print("Примеры интентов (первые 10):", train_df['intent'].unique()[:10])

    train_df = train_df[train_df['intent'].isin(SELECTED_INTENTS)]
    val_df   = val_df[val_df['intent'].isin(SELECTED_INTENTS)]
    test_df  = test_df[test_df['intent'].isin(SELECTED_INTENTS)]

    print(f"Train: {len(train_df)} примеров")
    print(f"Val:   {len(val_df)} примеров")
    print(f"Test:  {len(test_df)} примеров")

    if len(train_df) == 0:
        raise RuntimeError("Не найдено ни одного примера для выбранных интентов. Проверьте список SELECTED_INTENTS.")

    return train_df, val_df, test_df

def clean_text_baseline(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_bert(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def prepare_datasets(train_df, val_df, test_df):
    X_train_base = train_df['text'].apply(clean_text_baseline).values
    X_val_base   = val_df['text'].apply(clean_text_baseline).values
    X_test_base  = test_df['text'].apply(clean_text_baseline).values

    X_train_bert = train_df['text'].apply(clean_text_bert).values
    X_val_bert   = val_df['text'].apply(clean_text_bert).values
    X_test_bert  = test_df['text'].apply(clean_text_bert).values

    y_train = train_df['intent'].values
    y_val   = val_df['intent'].values
    y_test  = test_df['intent'].values

    le = {intent: idx for idx, intent in enumerate(SELECTED_INTENTS)}
    y_train_num = np.array([le[y] for y in y_train])
    y_val_num   = np.array([le[y] for y in y_val])
    y_test_num  = np.array([le[y] for y in y_test])

    return (X_train_base, X_val_base, X_test_base,
            X_train_bert, X_val_bert, X_test_bert,
            y_train_num, y_val_num, y_test_num, le)
