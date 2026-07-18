import os
import warnings
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

from src.utils import device
from src.data_loader import load_data, prepare_datasets, SELECTED_INTENTS
from src.models_baseline import train_baseline
from src.models_transformers import train_distilbert, train_roberta
from src.visualization import (
    plot_basic_stats,
    plot_wordclouds,
    plot_confusion_matrix,
    plot_classification_report_heatmap,
    plot_training_history,
    plot_tsne
)
from src.evaluation import show_error_examples, manual_check

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

MODEL_DIR = "models"
PLOT_DIR = "plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def main():
    print(f"Используемое устройство: {device}")
    if device.type == "cpu":
        print("Внимание: GPU не обнаружен. Обучение моделей будет очень медленным!")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df, val_df, test_df = load_data()
    train_original_df = train_df.copy()
    test_original_texts = test_df['text'].values

    (X_train_b, X_val_b, X_test_b,
     X_train_bert, X_val_bert, X_test_bert,
     y_train, y_val, y_test, le) = prepare_datasets(train_df, val_df, test_df)

    plot_basic_stats(y_train, le)
    plot_wordclouds(train_original_df, le)

    baseline_model, vectorizer, y_pred_base = train_baseline(X_train_b, y_train, X_test_b, y_test)

    distilbert_model, distilbert_tokenizer, y_pred_distilbert, hist_distilbert = train_distilbert(
        X_train_bert, y_train, X_val_bert, y_val, X_test_bert, y_test, le, device
    )

    roberta_model, roberta_tokenizer, y_pred_roberta, hist_roberta = train_roberta(
        X_train_bert, y_train, X_val_bert, y_val, X_test_bert, y_test, le, device
    )

    labels_list = SELECTED_INTENTS

    plot_confusion_matrix(y_test, y_pred_base, labels_list, "Confusion Matrix - Baseline", "cm_baseline.png")
    plot_confusion_matrix(y_test, y_pred_distilbert, labels_list, "Confusion Matrix - DistilBERT", "cm_distilbert.png")
    plot_confusion_matrix(y_test, y_pred_roberta, labels_list, "Confusion Matrix - RoBERTa", "cm_roberta.png")

    plot_classification_report_heatmap(y_test, y_pred_base, labels_list, "metrics_heatmap_baseline.png")
    plot_classification_report_heatmap(y_test, y_pred_distilbert, labels_list, "metrics_heatmap_distilbert.png")
    plot_classification_report_heatmap(y_test, y_pred_roberta, labels_list, "metrics_heatmap_roberta.png")

    plot_training_history(hist_distilbert, "DistilBERT", "training_history_distilbert.png")
    plot_training_history(hist_roberta, "RoBERTa", "training_history_roberta.png")

    plot_tsne(distilbert_model, distilbert_tokenizer, X_test_bert, y_test, le, "DistilBERT", "tsne_distilbert.png", device)
    plot_tsne(roberta_model, roberta_tokenizer, X_test_bert, y_test, le, "RoBERTa", "tsne_roberta.png", device)

    show_error_examples(distilbert_model, distilbert_tokenizer, test_original_texts, y_test, le, "DistilBERT", device, n=15)
    show_error_examples(roberta_model, roberta_tokenizer, test_original_texts, y_test, le, "RoBERTa", device, n=15)

    manual_check(distilbert_model, distilbert_tokenizer, le, "DistilBERT", device)
    manual_check(roberta_model, roberta_tokenizer, le, "RoBERTa", device)

    acc_base = accuracy_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base, average='macro')
    acc_distilbert = accuracy_score(y_test, y_pred_distilbert)
    f1_distilbert = f1_score(y_test, y_pred_distilbert, average='macro')
    acc_roberta = accuracy_score(y_test, y_pred_roberta)
    f1_roberta = f1_score(y_test, y_pred_roberta, average='macro')

    comparison = pd.DataFrame({
        'Модель': ['Baseline (TF-IDF+LR)', 'DistilBERT', 'RoBERTa'],
        'Accuracy': [acc_base, acc_distilbert, acc_roberta],
        'Macro F1': [f1_base, f1_distilbert, f1_roberta]
    })
    print("\nИтоговое сравнение:")
    print(comparison)
    comparison.to_csv(os.path.join(PLOT_DIR, "comparison.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x='Модель', y='Accuracy', data=comparison, ax=ax[0], palette='Set2')
    ax[0].set_title('Accuracy')
    sns.barplot(x='Модель', y='Macro F1', data=comparison, ax=ax[1], palette='Set2')
    ax[1].set_title('Macro F1')
    plt.suptitle('Сравнение моделей', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "comparison_plot.png"))
    plt.close()
    print("График сравнения моделей сохранён.")

    print("\nВсе файлы результатов сохранены в папках models/ и plots/.")

if __name__ == "__main__":
    main()
