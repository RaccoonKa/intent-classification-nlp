import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score
from src.data_loader import SELECTED_INTENTS

MODEL_DIR = "models"


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class LightAugmentedIntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64, aug_prob=0.5):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.aug_prob = aug_prob

    def _augment(self, text):
        words = text.split()
        if len(words) < 2:
            return text

        operation = random.choice(['delete', 'swap', 'replace_inplace'])

        if operation == 'delete' and len(words) >= 3:
            idx = random.randint(0, len(words) - 1)
            del words[idx]

        elif operation == 'swap':
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        elif operation == 'replace_inplace':
            if len(set(words)) >= 2:
                idx = random.randint(0, len(words) - 1)
                current_word = words[idx]
                other_words = [w for w in words if w != current_word]
                if other_words:
                    words[idx] = random.choice(other_words)

        return ' '.join(words)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if random.random() < self.aug_prob:
            text = self._augment(text)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_distilbert(X_train, y_train, X_val, y_val, X_test, y_test, le, device):
    print("\nОбучение DistilBERT:")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(SELECTED_INTENTS)
    )
    model.to(device)

    train_dataset = IntentDataset(X_train, y_train, tokenizer)
    val_dataset = IntentDataset(X_val, y_val, tokenizer)
    test_dataset = IntentDataset(X_test, y_test, tokenizer)

    output_dir = os.path.join(MODEL_DIR, "distilbert_checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=1,
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        return {'accuracy': acc, 'f1_macro': f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    print("Обучение DistilBERT...")
    trainer.train()

    model.save_pretrained(os.path.join(MODEL_DIR, "distilbert-intent"))
    tokenizer.save_pretrained(os.path.join(MODEL_DIR, "distilbert-intent"))

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"DistilBERT Test Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    history = trainer.state.log_history
    return model, tokenizer, y_pred, history


def train_roberta(X_train, y_train, X_val, y_val, X_test, y_test, le, device):
    print("\nОбучение RoBERTa (dropout + легкая аугментация):")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=len(SELECTED_INTENTS),
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3
    )
    model.to(device)

    train_dataset = LightAugmentedIntentDataset(X_train, y_train, tokenizer, aug_prob=0.5)
    val_dataset = IntentDataset(X_val, y_val, tokenizer)
    test_dataset = IntentDataset(X_test, y_test, tokenizer)

    output_dir = os.path.join(MODEL_DIR, "roberta_checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=1,
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        return {'accuracy': acc, 'f1_macro': f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    print("Обучение RoBERTa...")
    trainer.train()

    model.save_pretrained(os.path.join(MODEL_DIR, "roberta-intent"))
    tokenizer.save_pretrained(os.path.join(MODEL_DIR, "roberta-intent"))

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"RoBERTa Test Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    history = trainer.state.log_history
    return model, tokenizer, y_pred, history
