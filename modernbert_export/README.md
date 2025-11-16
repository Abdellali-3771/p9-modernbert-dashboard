# ModernBERT — Sentiment Analysis (Projet 9)

## Scores Test
- ROC-AUC: 0.9248
- Accuracy: 0.8485
- F1-score: 0.8529
- Precision: 0.8291
- Recall: 0.8781

## Usage Python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('modernbert_export')
tokenizer = AutoTokenizer.from_pretrained('modernbert_export')

text = 'I love this!'
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
logits = model(**inputs).logits
print(torch.argmax(logits))

## Contenu du dossier export
- config.json
- pytorch_model.bin
- tokenizer.json
- tokenizer_config.json
- model_card.json
- inference.py
- README.md
