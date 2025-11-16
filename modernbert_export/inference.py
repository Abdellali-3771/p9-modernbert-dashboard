import torch, re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def preprocess_tweet(t):
    t = re.sub(r'https?://\S+|www\.\S+', '[URL]', t)
    t = re.sub(r'@\w+', '[USER]', t)
    t = re.sub(r'#(\w+)', r'\1', t)
    t = re.sub(r'(.)\1{3,}', r'\1\1', t)
    return t.strip()

def predict(text, model_path='./modernbert_export'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    enc = tok(preprocess_tweet(text), max_length=128, truncation=True,
              padding='max_length', return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}

    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

    return {'label': 'Positive' if pred == 1 else 'Negative', 'confidence': float(probs[pred])}

if __name__ == '__main__':
    import sys
    text = ' '.join(sys.argv[1:])
    print(predict(text))
