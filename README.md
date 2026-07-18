# Intent Classification System

A project to classify intents for a voice assistant. 
As part of the work, the effectiveness of various approaches is compared: from classical Machine Learning 
before modern transformers.

## Description
The system is trained to recognize 8 types of user intents, including:
* `reminder`, `smart_home`, `directions`, `play_music`
* `weather`, `alarm`, `calendar`, `translate`

## Models used
Three architectures have been implemented and trained for classification:
1. **Baseline**: TF-IDF vectorizer + Logistic regression.
2. **DistilBERT**: Optimized BERT for fast classification.
3. **RoBERTa**: Enhanced architecture with light data augmentation and enhanced dropout for resilience.

## Functional
* **Visualization**: Construction of error matrices (confusion matrix), heat maps of metrics (Precision, Recall, F1) and training graphs.
* **Analysis**: Generating word clouds for each class.
* **Debugging**: Tools for displaying examples where the model is wrong, and manual verification of custom phrases.
* **t-SNE**: Visualization of embeddings of models in 2D space.

## Installation
1. Clone the repository: `git clone ...`
2. Install the dependencies: `pip install -r requirements.txt`
3. Launch training and evaluation: `python main.py`

## Results
The project implements a comparison of models for Accuracy and Macro F1.
All reports, graphs, and trained weights of the models are saved in the appropriate directories (`plots/` and `models/`).