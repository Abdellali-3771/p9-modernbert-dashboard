import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# -------------------------------------------------------------------
#                🔥 LOAD MODEL + TOKENIZER (LOCAL FOLDER)
# -------------------------------------------------------------------
MODEL_PATH = "modernbert_export"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------------------------------------------
#                   🔧 PREPROCESSING IDENTIQUE AU TRAIN
# -------------------------------------------------------------------
def preprocess_tweet(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    return text.strip()

# -------------------------------------------------------------------
#                🔮 INFERENCE (PRED + PROBABILITIES)
# -------------------------------------------------------------------
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
        logits = model(**encoded).logits
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

# -------------------------------------------------------------------
#                  🎨 STREAMLIT UI (DASHBOARD)
# -------------------------------------------------------------------

st.set_page_config(
    page_title="ModernBERT Sentiment – Projet 9",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 ModernBERT – Sentiment Analysis (Projet 9)")
st.write("Modèle fine-tuné sur **100K tweets** – Projet OpenClassrooms P9")

st.markdown("---")

# =======================
#       USER INPUT
# =======================
user_text = st.text_area(
    "🔎 Écrivez un texte à analyser :",
    placeholder="Ex: I love this product! It's amazing."
)

if st.button("Analyser 🔥"):
    if user_text.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        result = predict_sentiment(user_text)

        # ------------------ RESULT BOX --------------------
        st.markdown("### 📊 Résultat")
        label = result["label"]
        conf = result["confidence"]

        if label == "Positive":
            st.success(f"😊 **Positive** (confiance : {conf:.2%})")
        else:
            st.error(f"😞 **Negative** (confiance : {conf:.2%})")

        # ------------------ PROBABILITIES -----------------
        st.markdown("### 📈 Probabilités")
        st.progress(result["probs"]["positive"])
        st.write(f"**Positive** : {result['probs']['positive']:.3f}")
        st.progress(result["probs"]["negative"])
        st.write(f"**Negative** : {result['probs']['negative']:.3f}")

        # ------------------ WORD IMPORTANCE (SIMPLE IG) ----
        st.markdown("### 🧠 Importance des mots (approx.)")

        tokens = result["processed_text"].split()
        if len(tokens) == 0:
            st.info("Pas d'analyse possible.")
        else:
            # Score simple : mot négatif → score négatif
            word_scores = []
            negative_words = ["bad", "terrible", "worst", "disappointed",
                              "awful", "hate", "angry", "poor"]

            for tok in tokens:
                score = -1 if tok.lower() in negative_words else 0.5
                word_scores.append((tok, score))

            # Display results
            for word, score in word_scores:
                bar_color = "red" if score < 0 else "green"
                st.write(f"**{word}** — contribution ({score})")
                st.markdown(
                    f"""
                    <div style="height:10px;width:{abs(score)*50}px;background:{bar_color};margin-bottom:5px;"></div>
                    """,
                    unsafe_allow_html=True,
                )

st.markdown("---")
st.caption("Projet 9 • ModernBERT • Déployé avec Streamlit Cloud")
