import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# ================================================================
# 🚨 MUST BE THE FIRST STREAMLIT COMMAND
# ================================================================
st.set_page_config(
    page_title="ModernBERT Sentiment – Projet 9",
    page_icon="🚀",
    layout="centered"
)

# ================================================================
# 🔥 LOAD MODEL + TOKENIZER
# ================================================================
MODEL_PATH = "modernbert_export"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ================================================================
# 🔧 PREPROCESSING IDENTIQUE AU TRAIN
# ================================================================
def preprocess_tweet(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    return text.strip()

# ================================================================
# 🔮 INFERENCE
# ================================================================
def predict_sentiment(text: str):
    processed = preprocess_tweet(text)

    encoded = tokenizer(
        processed,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**encoded).logits.cpu()
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

    return {
        "label": "Positive" if pred == 1 else "Negative",
        "class": pred,
        "confidence": float(probs[pred]),
        "probs": {
            "positive": float(probs[1]),
            "negative": float(probs[0])
        },
        "processed_text": processed
    }

# ================================================================
# 🧠 SIMPLE WORD IMPORTANCE
# ================================================================
NEGATIVE_WORDS = {
    "bad","terrible","worst","awful","hate","angry","poor","disappointed",
    "upset","sad","horrible","boring","annoying","slow"
}

def compute_word_importance(tokens):
    scores = []
    for w in tokens:
        if w.lower() in NEGATIVE_WORDS:
            scores.append((w, -0.8))
        else:
            scores.append((w, 0.4))
    return scores

# ================================================================
# 🎨 STREAMLIT UI
# ================================================================
st.title("🚀 ModernBERT – Sentiment Analysis")
st.write("Modèle fine-tuné sur **100 000 tweets** – Projet OpenClassrooms P9")

st.markdown("---")

user_text = st.text_area(
    "🔎 Texte à analyser :",
    placeholder="Ex: I love this product! It's amazing."
)

if st.button("Analyser 🔥"):
    if user_text.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        result = predict_sentiment(user_text)

        st.markdown("### 📊 Résultat")
        label = result["label"]
        conf = result["confidence"]

        if label == "Positive":
            st.success(f"😊 **Positive** (confiance : {conf:.2%})")
        else:
            st.error(f"😞 **Negative** (confiance : {conf:.2%})")

        st.markdown("### 📈 Probabilités")
        st.write(f"Positive : **{result['probs']['positive']:.3f}**")
        st.write(f"Negative : **{result['probs']['negative']:.3f}**")

        st.markdown("### 🧠 Importance des mots (approx.)")
        tokens = result["processed_text"].split()
        scores = compute_word_importance(tokens)

        for word, score in scores:
            bar_color = "red" if score < 0 else "green"
            bar_width = int(abs(score) * 80)
            st.markdown(
                f"""
                <div style="margin:4px 0;">
                    <b>{word}</b>
                    <div style="height:8px;width:{bar_width}px;background:{bar_color};"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")
st.caption("Projet 9 • ModernBERT • Déployé avec Streamlit Cloud")
