import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

# ================================================================
# 🚨 CONFIG STREAMLIT (OBLIGATOIRE EN PREMIER)
# ================================================================
st.set_page_config(
    page_title="ModernBERT Sentiment – Projet 9",
    page_icon="🚀",
    layout="centered"
)

# ================================================================
# 🔧 DEBUG INFO (optionnel)
# ================================================================
with st.expander("🔧 Debug info"):
    st.write(f"Python: {sys.version}")
    st.write(f"PyTorch: {torch.__version__}")
    import transformers
    st.write(f"Transformers: {transformers.__version__}")


# ================================================================
# 🔥 CHARGEMENT DU TOKENIZER + MODÈLE
# ================================================================
MODEL_PATH = "modernbert_export"

@st.cache_resource
def load_model():
    try:
        # Tokenizer slow → évite Rust/tokenizers errors
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            use_fast=False,            # ⚠️ IMPORTANT
            trust_remote_code=True     # ⚠️ ModernBERT
        )

        # Modèle ModernBERT
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        model.eval()
        model.to("cpu")               # Streamlit Cloud = CPU

        return tokenizer, model

    except Exception as e:
        st.error(f"❌ Erreur chargement modèle/tokenizer : {e}")
        st.stop()


with st.spinner("⏳ Chargement du modèle ModernBERT..."):
    tokenizer, model = load_model()
st.success("✅ Modèle chargé !")


# ================================================================
# 🧹 PREPROCESSING (identique au training)
# ================================================================
def preprocess_tweet(text):
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ================================================================
# 🔮 PREDICTION
# ================================================================
def predict_sentiment(text):
    try:
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
            "confidence": float(probs[pred]),
            "probs": {
                "positive": float(probs[1]),
                "negative": float(probs[0])
            },
            "processed": processed
        }

    except Exception as e:
        st.error(f"❌ Erreur prédiction : {e}")
        return None


# ================================================================
# 🧠 WORD IMPORTANCE (simple lexical)
# ================================================================
NEGATIVE = {
    "bad", "terrible", "worst", "awful", "hate", "angry",
    "poor", "disappointed", "sad", "boring", "slow"
}
POSITIVE = {
    "good", "great", "best", "love", "excellent",
    "amazing", "perfect", "awesome"
}

def word_importance(tokens):
    out = []
    for w in tokens:
        wl = w.lower()
        if wl in NEGATIVE:
            out.append((w, -0.8))
        elif wl in POSITIVE:
            out.append((w, 0.8))
        else:
            out.append((w, 0.2))
    return out


# ================================================================
# 🎨 INTERFACE STREAMLIT
# ================================================================
st.title("🚀 ModernBERT – Sentiment Analysis (Projet 9)")
st.write("Modèle fine-tuné sur **100 000 tweets** — Projet P9 OpenClassrooms")
st.markdown("---")

examples = {
    "Positif": "I love this product, it's amazing!",
    "Négatif": "This is the worst experience ever.",
    "Neutre": "The movie was okay, nothing special."
}

cols = st.columns(len(examples))
preset = None
for i, (name, txt) in enumerate(examples.items()):
    if cols[i].button(name):
        preset = txt

user_text = st.text_area(
    "📝 Texte à analyser :",
    value=preset if preset else "",
    placeholder="Ex: I love this product!",
    height=110
)

if st.button("🔥 Analyser", type="primary"):
    if not user_text.strip():
        st.warning("⚠️ Veuillez entrer un texte.")
    else:
        with st.spinner("Analyse en cours..."):
            result = predict_sentiment(user_text)

        if result:
            st.markdown("### 📊 Résultat")
            label = result["label"]
            conf = result["confidence"]

            if label == "Positive":
                st.success(f"😊 **Positive** — {conf:.2%}")
            else:
                st.error(f"😞 **Negative** — {conf:.2%}")

            st.markdown("### 📈 Probabilités")
            st.write(f"Positive : {result['probs']['positive']:.3f}")
            st.write(f"Negative : {result['probs']['negative']:.3f}")

            st.markdown("### 🔧 Texte prétraité")
            st.code(result["processed"])

            st.markdown("### 🧠 Importance des mots")
            tokens = result["processed"].split()
            scores = word_importance(tokens)

            html = []
            for word, score in scores:
                if score < 0:
                    color = "#ff4444"
                elif score > 0.5:
                    color = "#44ff44"
                else:
                    color = "#cccccc"
                opacity = abs(score)
                html.append(
                    f'<span style="background:{color};opacity:{opacity};'
                    f'padding:3px;border-radius:4px;margin:2px">{word}</span>'
                )
            st.markdown(" ".join(html), unsafe_allow_html=True)

st.markdown("---")
st.caption("Projet 9 • ModernBERT • Déployé sur Streamlit Cloud")
