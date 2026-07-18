import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.models_transformers import IntentDataset

PLOT_DIR = "plots"

def show_error_examples(model, tokenizer, X_test_orig, y_test, le, model_name, device, n=15):
    print(f"\nАнализ ошибок {model_name}:")
    inv_le = {v: k for k, v in le.items()}
    texts = X_test_orig
    y_pred = []
    dataset = IntentDataset(X_test_orig, y_test, tokenizer)
    loader = DataLoader(dataset, batch_size=32)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
    y_pred = np.array(y_pred)

    errors = []
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            errors.append((texts[i], inv_le[y_test[i]], inv_le[y_pred[i]]))

    if len(errors) == 0:
        print("Ошибок нет!")
        return

    np.random.shuffle(errors)
    sample_errors = errors[:n]

    print(f"{'Текст':<50} {'Истинный':<15} {'Предсказанный':<15} {'Возможная причина'}")
    print("-" * 130)
    explanations = {
        ('weather', 'search'): "Погода – разновидность поиска информации",
        ('reminder', 'smart_home'): "Установка напоминания и управление устройством могут пересекаться",
        ('smart_home', 'reminder'): "Управление устройством иногда похоже на установку напоминания",
        ('directions', 'search'): "Навигация = поиск маршрута",
        ('play_music', 'search'): "Воспроизведение музыки – поиск контента",
        ('alarm', 'reminder'): "Будильник и напоминание близки по смыслу",
        ('reminder', 'alarm'): "Напоминание могут путать с будильником",
        ('calendar', 'reminder'): "Календарь и напоминания пересекаются",
        ('translate', 'search'): "Запрос перевода похож на поисковый запрос",
    }

    for text, true, pred in sample_errors:
        expl = explanations.get((true, pred), f"Пересечение классов '{true}' и '{pred}'")
        print(f"{text:<50} {true:<15} {pred:<15} {expl}")

    df_err = pd.DataFrame(sample_errors, columns=['Текст', 'Истинный класс', 'Предсказанный класс'])
    df_err.to_csv(os.path.join(PLOT_DIR, f"error_examples_{model_name.lower()}.csv"), index=False)
    print(f"Таблица ошибок {model_name} сохранена.")

def manual_check(model, tokenizer, le, model_name, device):
    phrases = [
        ("find me a recipe for lasagna", "search"),
        ("what's the capital of France", "search"),
        ("set an alarm for my dentist appointment", "alarm"),
        ("wake me up at 7 am", "alarm"),
        ("remind me to buy milk tomorrow at 8am", "reminder"),
        ("set a reminder to water the plants", "reminder"),
        ("turn off the lights in the living room", "smart_home"),
        ("set thermostat to 22 degrees", "smart_home"),
        ("how do I get to the nearest coffee shop", "directions"),
        ("navigate to Central Park", "directions"),
        ("play some relaxing jazz music", "play_music"),
        ("shuffle my workout playlist", "play_music"),
        ("do I need an umbrella this weekend", "weather"),
        ("what's the temperature in Chicago", "weather"),
        ("what are the latest sports news", "news"),
        ("tell me the top headlines", "news"),
        ("add meeting to my calendar for Friday", "calendar"),
        ("what's on my schedule today", "calendar"),
        ("translate hello to Spanish", "translate"),
        ("how do you say thank you in French", "translate"),
        ("look up flights to London", "search"),
    ]

    inv_le = {v: k for k, v in le.items()}
    print(f"\nРучная проверка на собственных фразах ({model_name})")
    results = []
    for text, true_label in phrases:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
        pred_label = inv_le[pred_idx]
        correct = (pred_label == true_label)
        results.append((text, true_label, pred_label, confidence, correct))
        status = "✓" if correct else "✗"
        print(f"{status} '{text}' -> {pred_label} (уверенность: {confidence:.2f})")

    df_res = pd.DataFrame(results, columns=['Текст', 'Истинный класс', 'Предсказанный класс', 'Уверенность', 'Верно?'])
    df_res.to_csv(os.path.join(PLOT_DIR, f"manual_check_{model_name.lower()}.csv"), index=False)
    print(f"Результаты ручной проверки {model_name} сохранены.")
    return df_res
