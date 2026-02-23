# ============================================================
# Script de nettoyage des données
# ============================================================

import pandas as pd
import numpy as np
import os


# ------------------------------------------------------------
# 1️⃣ Chargement du dataset
# ------------------------------------------------------------

# Lecture du fichier CSV
df = pd.read_csv("C:/projet/projet_ml_retail/data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")

# Copie de sécurité pour éviter de modifier l’original
df_clean = df.copy()

print("Shape initial :", df_clean.shape)  # (lignes, colonnes)


# ------------------------------------------------------------
# 2️⃣ Suppression des colonnes constantes
# ------------------------------------------------------------

# Colonnes avec une seule valeur unique (aucune information utile)
constant_cols = [
    col for col in df_clean.columns
    if df_clean[col].nunique() <= 1   # nunique() = nombre de valeurs uniques
]

df_clean.drop(columns=constant_cols, inplace=True)


# ------------------------------------------------------------
# 3️⃣ Suppression des colonnes quasi-constantes
# ------------------------------------------------------------

# Si une seule valeur représente plus de 95% des données
quasi_const_cols = [
    col for col in df_clean.columns
    if df_clean[col].value_counts(normalize=True).max() > 0.95
    # normalize=True → donne les proportions au lieu des comptes
]

df_clean.drop(columns=quasi_const_cols, inplace=True)


# ------------------------------------------------------------
# 4️⃣ Suppression colonnes non pertinentes
# ------------------------------------------------------------

cols_to_remove = [
    'CustomerID', 'LastLoginIP', 'RegistrationDate',
    'IP_privee', 'Country_encoded',
    'UniqueCountries', 'ZeroPriceCount'
]

# On vérifie que les colonnes existent avant suppression
cols_to_remove = [c for c in cols_to_remove if c in df_clean.columns]

df_clean.drop(columns=cols_to_remove, inplace=True)


# ------------------------------------------------------------
# 5️⃣ Suppression colonnes avec trop de valeurs manquantes
# ------------------------------------------------------------

threshold = 0.4  # 40% de NaN maximum autorisé

# mean() sur isna() donne le pourcentage de valeurs manquantes
cols_trop_nan = df_clean.columns[df_clean.isna().mean() > threshold]

df_clean.drop(columns=cols_trop_nan, inplace=True)


# ------------------------------------------------------------
# 6️⃣ Suppression des variables fortement corrélées
# ------------------------------------------------------------

# On garde uniquement les colonnes numériques
corr_matrix = df_clean.select_dtypes(include=np.number).corr().abs()

# On prend uniquement la partie supérieure de la matrice
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Colonnes corrélées à plus de 0.9
high_corr_cols = [
    col for col in upper_triangle.columns
    if any(upper_triangle[col] > 0.9)
]

# On protège la variable cible si elle existe
if "Churn" in high_corr_cols:
    high_corr_cols.remove("Churn")

df_clean.drop(columns=high_corr_cols, inplace=True)


print("Shape final :", df_clean.shape)


# ------------------------------------------------------------
# 7️⃣ Sauvegarde du dataset nettoyé
# ------------------------------------------------------------

# Création du dossier s'il n'existe pas
os.makedirs("../projet_ml_retail/data/processed", exist_ok=True)

# Sauvegarde en CSV
df_clean.to_csv("C:/projet/projet_ml_retail/data/processed/data_cleaned_v2.csv", index=False)

print("✔ Dataset nettoyé sauvegardé avec succès")