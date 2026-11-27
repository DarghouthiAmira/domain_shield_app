# 🛡️ DomainShield AI — Détection de Domaines Malveillants par Apprentissage Automatique

Projet tutoré réalisé dans le cadre du **Master en Ingénierie des Systèmes Émergents (ISEM)** à l'**Université Virtuelle de Tunis (UVT)**.

**Auteure :** Amira Darghouthi

---

## 🎯 Objectif du projet

**DomainShield AI** est une solution de cybersécurité basée sur le *Machine Learning* permettant de détecter de manière proactive les noms de domaines malveillants (générés par algorithmes **DGA**, campagnes de **Phishing**, **Typosquatting**, ou distribution de **Malware**) uniquement à partir de leur structure lexicale et statistique, sans dépendre de listes noires statiques.

---

## 🏗️ Architecture & Pipeline Technique

Le projet s'articule autour de quatre modules principaux :

1. **Collecte & Prétraitement des Données (`1_download_data.py`)**
   - Nettoyage, extraction des noms de domaines (FQDN) et équilibrage d'un jeu de données de **20 000 domaines** (10 000 légitimes et 10 000 malveillants) issu de *MUD Malicious URLs 2026*.

2. **Ingénierie des Caractéristiques & Modélisation Hybride (`2_train_models.py`)**
   - **Extraction lexicale et statistique :** Entropie de Shannon, longueur du domaine, ratio de chiffres, ratio de voyelles, séquence maximale de consonnes consécutives, nombre de sous-domaines et présence de tirets.
   - **Vectorisation textuelle :** TF-IDF sur n-grammes de caractères (`3` à `5` caractères).
   - **Comparaison multi-algorithmes :** Évaluation comparative de **Régression Logistique**, **Random Forest** et **XGBoost** (modèle retenu et exporté).
   - **Reporting automatisé :** Génération d'un rapport d'évaluation PDF (`reports/rapport_final.pdf`) incluant les métriques (Accuracy, Precision, Recall, F1-Score, ROC-AUC) et les matrices de confusion (`reports/matrices.png`).

3. **Persistance des Données (`database.py`)**
   - Base de données locale **SQLite** (`app_history.db`) assurant la journalisation et l'audit des analyses effectuées.

4. **Application Web Interactive (`app.py` & `predict.py`)**
   - Interface utilisateur développée avec **Streamlit** permettant de tester en temps réel un nom de domaine ou une URL complète, avec calcul du score de risque, niveau d'alerte et consultation de l'historique.

---

## 📂 Structure du dépôt

```text
domain_shield_app/
├── 1_download_data.py      # Préparation et équilibrage du dataset
├── 2_train_models.py       # Pipeline ML hybride, entraînement et génération du rapport PDF
├── predict.py              # Script d'inférence en ligne de commande
├── app.py                  # Interface web interactive Streamlit
├── database.py             # Gestion de la base de données SQLite
├── reports/
│   ├── matrices.png        # Matrices de confusion comparatives
│   └── rapport_final.pdf   # Rapport d'évaluation automatique des modèles
└── .gitignore              # Exclusion des fichiers volumineux et environnements virtuels
