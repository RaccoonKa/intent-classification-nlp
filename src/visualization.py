import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import torch
from src.models_transformers import IntentDataset
from src.data_loader import SELECTED_INTENTS

PLOT_DIR = "plots"

def plot_basic_stats(y_train, le):
    counts = Counter(y_train)
    intents = [k for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
    values  = [counts[i] for i in intents]

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=intents, y=values, palette='viridis')
    plt.title('Распределение классов в обучающей выборке', fontsize=14)
    plt.xlabel('Интент')
    plt.ylabel('Количество примеров')
    plt.xticks(rotation=20)
    for i, v in enumerate(values):
        ax.text(i, v + 2, str(v), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "class_distribution.png"))
    plt.close()
    print("График распределения классов сохранён.")

def plot_wordclouds(train_df, le):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    stopwords = set(STOPWORDS)
    custom_stops = {'im', 'ive', 'youre', 'dont', 'cant', 'ill', 'id', 'youve', 'hes', 'shes', 'that', 'what', 'where',
                    'which', 'how', 'when', 'who', 'whom', 'can', 'will', 'want', 'need', 'find', 'give', 'tell'}
    stopwords = stopwords.union(custom_stops)

    for idx, intent in enumerate(SELECTED_INTENTS):
        if idx >= len(axes):
            break
        mask = train_df['intent'] == intent
        text = " ".join(train_df[mask]['text'].values)
        if len(text.strip()) == 0:
            text = "no data"
        wc = WordCloud(width=300, height=200, background_color='white',
                       stopwords=stopwords, max_words=50, collocations=False,
                       colormap='Dark2').generate(text)
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(intent, fontsize=10)
        axes[idx].axis("off")

    for j in range(len(SELECTED_INTENTS), len(axes)):
        axes[j].axis('off')

    plt.suptitle('Облака слов по классам интентов', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "wordclouds.png"))
    plt.close()
    print("Облака слов сохранены.")

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14)
    plt.xlabel('Предсказанный')
    plt.ylabel('Истинный')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()
    print(f"Матрица ошибок сохранена: {filename}")

def plot_classification_report_heatmap(y_true, y_pred, labels, filename):
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True
    )
    df_report = pd.DataFrame(report).transpose()
    df_plot = df_report[['precision', 'recall', 'f1-score']].drop(
        ['accuracy', 'macro avg', 'weighted avg'], errors='ignore'
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_plot.astype(float), annot=True, cmap='YlGnBu', vmin=0, vmax=1)
    plt.title('Precision, Recall, F1-score по классам', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()
    print(f"Heatmap метрик сохранён: {filename}")


def plot_training_history(history, model_name, filename):
    train_loss = []
    eval_loss = []
    eval_acc = []
    epochs_train = []
    epochs_eval = []

    for log in history:
        if 'loss' in log and 'epoch' in log and 'eval_loss' not in log:
            train_loss.append(log['loss'])
            epochs_train.append(log['epoch'])
        if 'eval_loss' in log and 'epoch' in log:
            eval_loss.append(log['eval_loss'])
            epochs_eval.append(log['epoch'])
        if 'eval_accuracy' in log and 'epoch' in log:
            eval_acc.append(log['eval_accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if train_loss:
        ax1.plot(epochs_train, train_loss, 'o-', label='Train Loss')
    if eval_loss:
        ax1.plot(epochs_eval, eval_loss, 's-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} Loss')
    ax1.legend()

    if eval_acc:
        ax2.plot(epochs_eval, eval_acc, 'D-', color='green', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()
    print(f"График обучения {model_name} сохранён: {filename}")


def plot_tsne(model, tokenizer, X_test, y_test, le, model_name, filename, device):
    print(f"Вычисление эмбеддингов для t-SNE ({model_name})...")
    model.eval()
    n_samples = min(2000, len(X_test))
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    texts_subset = [X_test[i] for i in indices]
    labels_subset = y_test[indices]

    embeddings = []
    batch_size = 64
    for i in range(0, len(texts_subset), batch_size):
        batch_texts = texts_subset[i:i + batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            if 'distilbert' in model.config.architectures[0].lower():
                outputs = model.distilbert(**encoded)
            elif 'roberta' in model.config.architectures[0].lower():
                outputs = model.roberta(**encoded)
            else:
                outputs = model.base_model(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    embeddings = np.concatenate(embeddings, axis=0)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    reduced = tsne.fit_transform(embeddings)

    inv_le = {v: k for k, v in le.items()}
    label_names = [inv_le[l] for l in labels_subset]

    plt.figure(figsize=(12, 10))
    unique_labels = sorted(set(label_names))
    palette = sns.color_palette("husl", len(unique_labels))
    for i, lab in enumerate(unique_labels):
        idx = [j for j, l in enumerate(label_names) if l == lab]
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=lab, alpha=0.6, color=palette[i])
    plt.title(f't-SNE эмбеддингов {model_name} (тестовые примеры)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()
    print(f"t-SNE визуализация для {model_name} сохранена: {filename}")
