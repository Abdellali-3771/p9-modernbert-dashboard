import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

# ================================================================
# 🚨 STREAMLIT CONFIG — MUST BE THE FIRST STREAMLIT COMMAND
# ================================================================
st.set_page_config(
    page_title="ModernBERT Sentiment – Projet 9",
    page_icon="🚀",
    layout="centered"
)

# ================================================================
# 🔍 DEBUG INFO (à retirer en production)
# ================================================================
with st.expander("🔧 Debug Info"):
    st.write(f"Python: {sys.version}")
    st.write(f"Streamlit: {st.__version__}")
    st.write(f"PyTorch: {torch.__version__}")
    import transformers
    st.write(f"Transformers: {transformers.__version__}")

# ================================================================
# 🔥 LOAD MODEL + TOKENIZER
# ================================================================
MODEL_PATH = "modernbert_export"

@st.cache_resource
def load_model():
    """
    Charge le modèle avec gestion d'erreurs
    use_fast=False pour compatibilité Streamlit Cloud
    """
    try:
        # Tokenizer sans fast (évite erreurs Rust)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            use_fast=False,
            trust_remote_code=True  # Pour ModernBERT
        )
        
        # Modèle en mode CPU
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        model.eval()
        
        # Forcer CPU
        model = model.to('cpu')
        
        return tokenizer, model
    
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle: {e}")
        st.stop()

# Charger avec message de progression
with st.spinner("⏳ Chargement du modèle ModernBERT..."):
    tokenizer, model = load_model()
    st.success("✅ Modèle chargé!")

# ================================================================
# 🔧 PREPROCESSING IDENTIQUE AU TRAIN
# ================================================================
def preprocess_tweet(text: str) -> str:
    """Prétraitement standardisé des tweets"""
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ================================================================
# 🔮 INFERENCE
# ================================================================
def predict_sentiment(text: str):
    """
    Prédit le sentiment avec gestion d'erreurs
    """
    try:
        processed = preprocess_tweet(text)

        # Tokenization
        encoded = tokenizer(
            processed,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Inference (CPU)
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits.cpu()
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
    
    except Exception as e:
        st.error(f"❌ Erreur prédiction: {e}")
        return None

# ================================================================
# 🧠 SIMPLE WORD IMPORTANCE (LEXICAL RULES)
# ================================================================
NEGATIVE_WORDS = {
    "bad", "terrible", "worst", "awful", "hate", "angry", "poor", 
    "disappointed", "upset", "sad", "horrible", "boring", "annoying", 
    "slow", "useless", "waste", "never", "not", "no"
}

POSITIVE_WORDS = {
    "good", "great", "best", "amazing", "love", "happy", "excellent",
    "wonderful", "fantastic", "perfect", "awesome", "brilliant"
}

def compute_word_importance(tokens):
    """Calcule importance approximative des mots"""
    scores = []
    for w in tokens:
        word_lower = w.lower()
        if word_lower in NEGATIVE_WORDS:
            scores.append((w, -0.8))
        elif word_lower in POSITIVE_WORDS:
            scores.append((w, 0.8))
        else:
            scores.append((w, 0.2))
    return scores

# ================================================================
# 🎨 STREAMLIT UI
# ================================================================
st.title("🚀 ModernBERT – Sentiment Analysis")
st.markdown("""
**Modèle fine-tuné sur 100 000 tweets** – Projet OpenClassrooms P9

📊 **Performances** :
- ROC-AUC : **0.9248**
- Accuracy : **0.8485**
- F1-Score : **0.8529**
""")

st.markdown("---")

# ================================================================
# EXEMPLES PRÉ-DÉFINIS
# ================================================================
examples = {
    "Positif clair": "I love this product! It's amazing and works perfectly!",
    "Négatif clair": "This is the worst experience ever. Totally disappointed.",
    "Neutre": "The movie was okay, nothing special but not terrible either.",
    "Avec mention": "@user Thanks for the great service! Really appreciate it :)",
    "Avec hashtag": "Can't believe how bad this is... #fail #disappointed"
}

st.markdown("### 📝 Exemples rapides")
cols = st.columns(len(examples))
selected_example = None

for i, (label, text) in enumerate(examples.items()):
    if cols[i].button(label, key=f"example_{i}"):
        selected_example = text

# ================================================================
# INPUT UTILISATEUR
# ================================================================
default_text = selected_example if selected_example else ""
user_text = st.text_area(
    "🔎 Texte à analyser :",
    value=default_text,
    placeholder="Ex: I love this product! It's amazing.",
    height=100
)

# ================================================================
# ANALYSE
# ================================================================
if st.button("🔥 Analyser", type="primary"):
    if user_text.strip() == "":
        st.warning("⚠️ Veuillez entrer un texte.")
    else:
        with st.spinner("Analyse en cours..."):
            result = predict_sentiment(user_text)
        
        if result:
            # ----------------------------------------------------------
            # RESULT
            # ----------------------------------------------------------
            st.markdown("### 📊 Résultat")
            label = result["label"]
            conf = result["confidence"]

            col1, col2 = st.columns(2)
            
            with col1:
                if label == "Positive":
                    st.success(f"😊 **{label}**")
                else:
                    st.error(f"😞 **{label}**")
            
            with col2:
                st.metric("Confiance", f"{conf:.1%}")

            # ----------------------------------------------------------
            # PROBABILITIES
            # ----------------------------------------------------------
            st.markdown("### 📈 Probabilités détaillées")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Positive", 
                    f"{result['probs']['positive']:.3f}",
                    delta=f"{(result['probs']['positive'] - 0.5)*100:+.1f}%"
                )
            with col2:
                st.metric(
                    "Negative", 
                    f"{result['probs']['negative']:.3f}",
                    delta=f"{(result['probs']['negative'] - 0.5)*100:+.1f}%"
                )

            # Progress bars
            st.progress(result['probs']['positive'], text=f"Positive: {result['probs']['positive']:.1%}")
            st.progress(result['probs']['negative'], text=f"Negative: {result['probs']['negative']:.1%}")

            # ----------------------------------------------------------
            # TEXTE PREPROCESSÉ
            # ----------------------------------------------------------
            with st.expander("🔧 Texte prétraité"):
                st.code(result["processed_text"])

            # ----------------------------------------------------------
            # WORD IMPORTANCE
            # ----------------------------------------------------------
            st.markdown("### 🧠 Importance des mots (approximation)")
            st.caption("⚠️ Basé sur des règles lexicales simples (non SHAP)")

            tokens = result["processed_text"].split()
            scores = compute_word_importance(tokens)

            # Créer HTML pour affichage
            html_words = []
            for word, score in scores:
                if score < 0:
                    color = "#ff4444"  # Rouge
                    opacity = abs(score)
                elif score > 0.5:
                    color = "#44ff44"  # Vert
                    opacity = score
                else:
                    color = "#888888"  # Gris
                    opacity = 0.3
                
                html_words.append(
                    f'<span style="background-color:{color}; '
                    f'opacity:{opacity}; padding:2px 4px; margin:2px; '
                    f'border-radius:3px;">{word}</span>'
                )
            
            st.markdown(" ".join(html_words), unsafe_allow_html=True)

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown("""
### 📚 À propos

**ModernBERT-base** fine-tuné sur le dataset Sentiment140 :
- 🎯 100 000 tweets d'entraînement
- 🚀 GPU P100 (45.5 min)
- 📊 ROC-AUC : 0.9248 (+4.48% vs BERT P7)

**Projet 9** – OpenClassrooms AI Engineer  
*Étudiant : [Abdellali Touaddi]*
""")

st.caption("Déployé avec Streamlit Cloud • ModernBERT © Answer.AI")
