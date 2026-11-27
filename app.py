import streamlit as st
import pandas as pd
import joblib
import importlib
import __main__
from urllib.parse import urlparse

# Chargement propre des modules locaux
train_mod = importlib.import_module("2_train_models")
__main__.LexicalFeatureExtractor = train_mod.LexicalFeatureExtractor
from database import init_db, save_analysis, get_history

# 1. Fonction de nettoyage automatique (placée ici, avant l'interface)
def clean_input_domain(raw_input: str) -> str:
    val = raw_input.strip().lower()
    if not val.startswith(("http://", "https://")):
        val = "http://" + val
    domain = urlparse(val).netloc.split(":")[0]
    return domain[4:] if domain.startswith("www.") else domain

st.set_page_config(page_title="DomainShield AI", page_icon="🛡️", layout="wide")
init_db()

@st.cache_resource
def load_model():
    return joblib.load("models/best_domain_detector.pkl")

st.title("🛡️ DomainShield AI — Détection de Domaines Malveillants")
st.markdown("Analyse lexicale et statistique par apprentissage automatique (DGA, Phishing, Typosquatting).")

try:
    model = load_model()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# Zone de test
col1, col2 = st.columns([2, 1])
with col1:
    domain_input = st.text_input("Entrez un nom de domaine (FQDN) ou une URL à analyser :", placeholder="ex: x89qka12z98p.info ou google.com")
    if st.button("Analyser le domaine", type="primary") and domain_input.strip():
        # 2. Utilisation de la fonction de nettoyage ici
        clean_d = clean_input_domain(domain_input)
        
        proba = float(model.predict_proba([clean_d])[0][1])
        sld = clean_d.split(".")[-2] if "." in clean_d else clean_d
        entropy = train_mod.shannon_entropy(sld)

        if proba >= 0.75:
            status = "CRITIQUE / MALVEILLANT"
            st.error(f"🚨 **{status}** (`{clean_d}`) — Score de risque : **{proba*100:.2f}%** (Entropie : {entropy:.2f})")
        elif proba >= 0.50:
            status = "SUSPECT"
            st.warning(f"⚠️ **{status}** (`{clean_d}`) — Score de risque : **{proba*100:.2f}%** (Entropie : {entropy:.2f})")
        else:
            status = "LÉGITIME / SÛR"
            st.success(f"✅ **{status}** (`{clean_d}`) — Score de risque : **{proba*100:.2f}%** (Entropie : {entropy:.2f})")

        save_analysis(clean_d, proba * 100, status, entropy)

# Historique depuis SQLite
st.divider()
st.subheader("📊 Historique des analyses (Base de données SQLite)")
history = get_history(50)
if history:
    df_hist = pd.DataFrame(history, columns=["ID", "Domaine", "Score Risque (%)", "Statut", "Entropie", "Date"])
    st.dataframe(df_hist, use_container_width=True, hide_index=True)
else:
    st.info("Aucune analyse enregistrée pour le moment.")