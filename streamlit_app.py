import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================================================================
# 🚨 STREAMLIT CONFIG — FIRST COMMAND
# ================================================================
st.set_page_config(
    page_title="ModernBERT Sentiment – Projet 9",
    page_icon="🚀",
    layout="centered"
)

# ================================================================
# 🔥 LOAD MODEL (CPU, no fast tokenizer)
# ================================================================
MODEL_PATH = "modernbert_export"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False  # IMPORTANT for Streamlit Cloud
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    ).to("cpu")
    model.eval()
    return tokenizer, model

with st.spinner("⏳ Chargement du modèle ModernBERT..."):
    tokenizer, model = load_model()
st.success("✅ Modèle chargé !")

# ================================================================
# 🔧 PREPROCESSING
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

    enc = tokenizer(
        processed,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

    return {
        "label": "Positive" if pred == 1 else "Negative",
        "confidence": float(probs[pred]),
        "probs": {
            "positive": float(probs[1]),
            "negative": float(probs[0])
        },
        "processed_text": processed
    }

# ================================================================
# 🎨 SIMPLE WORD IMPORTANCE
# ================================================================
NEG = {"bad","terrible","worst","awful","hate","angry","poor","disappointed",
       "upset","sad","horrible","boring","annoying","slow"}

def compute_word_importance(tokens):
    scores = []
    for w in tokens:
        if w.lower() in NEG:
            scores.append((w, -0.8))
        else:
            scores.append((w, 0.3))
    return scores

# ================================================================
# 🎨 UI
# ================================================================
st.title("🚀 ModernBERT – Analyse de Sentiment (Projet P9)")
st.write("Modèle fine-tuné sur **100 000 tweets** — compatible Streamlit Cloud")

st.markdown("---")

user_text = st.text_area(
    "🔎 Texte à analyser :",
    placeholder="Ex: I love this product!"
)

if st.button("🔥 Analyser"):
    if not user_text.strip():
        st.warning("Veuillez entrer un texte.")
    else:
        result = predict_sentiment(user_text)

        st.markdown("### 📊 Résultat")
        if result["label"] == "Positive":
            st.success(f"😊 **Positive** ({result['confidence']:.1%})")
        else:
            st.error(f"😞 **Negative** ({result['confidence']:.1%})")

        st.markdown("### 📈 Probabilités")
        st.write(result["probs"])

        st.markdown("### 🧠 Importance des mots")
        tokens = result["processed_text"].split()
        scores = compute_word_importance(tokens)

        for word, score in scores:
            color = "red" if score < 0 else "green"
            width = int(abs(score) * 80)
            st.markdown(
                f"<div><b>{word}</b> "
                f"<div style='height:8px;width:{width}px;background:{color}'></div></div>",
                unsafe_allow_html=True
            )

st.markdown("---")
st.caption("Projet 9 • Déployé sur Streamlit Cloud • ModernBERT (CPU)")
