import os
import pandas as pd
from urllib.parse import urlparse

RAW_PATH = "data/kaggle_raw.csv"
OUT_PATH = "data/domains.csv"
MAX_PER_CLASS = 10000  # 10 000 bénins + 10 000 malveillants = 20 000 domaines (rapide et précis)

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(
        f"Fichier '{RAW_PATH}' introuvable. Placez le CSV téléchargé depuis Kaggle dans le dossier 'data/' et renommez-le 'kaggle_raw.csv'."
    )

print(f"[1/4] Chargement de {RAW_PATH}...")
df = pd.read_csv(RAW_PATH, low_memory=False)
print("      Colonnes détectées :", list(df.columns))

# 1. Identification automatique de la colonne URL / Domaine
url_col = None
for candidate in ["domain", "url", "URL", "Domain", "hostname"]:
    if candidate in df.columns:
        url_col = candidate
        break
if url_col is None:
    url_col = df.columns[0]  # Première colonne par défaut

# 2. Identification automatique de la colonne Label / Type
label_col = None
for candidate in ["label", "Label", "type", "Type", "result", "Result", "class", "status"]:
    if candidate in df.columns:
        label_col = candidate
        break
if label_col is None:
    label_col = df.columns[-1]  # Dernière colonne par défaut

print(f"      Utilisation de '{url_col}' (entrée) et '{label_col}' (cible).")

# 3. Normalisation de l'étiquette en binaire (0 = Bénin, 1 = Malveillant)
def normalize_label(val):
    s = str(val).strip().lower()
    if s in ["0", "0.0", "benign", "legitimate", "safe", "good", "clean"]:
        return 0
    return 1  # phishing, malware, defacement, malicious, 1 -> 1

df["clean_label"] = df[label_col].apply(normalize_label)

# 4. Extraction propre du nom de domaine (FQDN) depuis l'URL
def extract_domain(val):
    try:
        val = str(val).strip().strip('"')
        if not val.startswith(("http://", "https://")):
            val = "http://" + val
        domain = urlparse(val).netloc.split(":")[0].lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        if "." in domain and any(c.isalpha() for c in domain):
            return domain
        return None
    except Exception:
        return None

print("[2/4] Extraction et nettoyage des noms de domaine...")
df["clean_domain"] = df[url_col].apply(extract_domain)
df_clean = df[["clean_domain", "clean_label"]].dropna()
df_clean.columns = ["domain", "label"]
df_clean = df_clean.drop_duplicates(subset=["domain"])

# 5. Équilibrage des classes (50% Bénins / 50% Malveillants)
print("[3/4] Équilibrage des classes...")
df_benign = df_clean[df_clean["label"] == 0]
df_malicious = df_clean[df_clean["label"] == 1]

n_samples = min(len(df_benign), len(df_malicious), MAX_PER_CLASS)
df_balanced = pd.concat([
    df_benign.sample(n=n_samples, random_state=42),
    df_malicious.sample(n=n_samples, random_state=42)
], ignore_index=True)

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Sauvegarde finale
print("[4/4] Sauvegarde dans data/domains.csv...")
df_balanced.to_csv(OUT_PATH, index=False)
print(f"[+] Succès ! {len(df_balanced)} domaines prêts ({n_samples} bénins, {n_samples} malveillants).")