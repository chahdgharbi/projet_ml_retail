# ============================================================
# src/utils.py
# Fonctions utilitaires pour l'analyse des données
# ============================================================

import pandas as pd          # Manipulation des données (DataFrame)
import numpy as np           # Calculs numériques
import matplotlib.pyplot as plt  # Graphiques
import seaborn as sns        # Graphiques avancés
import os                    # Gestion des dossiers/fichiers


# ============================================================
# 1️⃣ Analyse des valeurs manquantes
# ============================================================

def analyser_nan(df):
    """
    Analyse les valeurs manquantes du DataFrame
    et retourne un résumé trié.
    """

    # Vérifier si le dataset est vide
    if df.empty:
        print("Dataset vide.")
        return None

    # Compter les valeurs manquantes par colonne
    total_nan = df.isna().sum()

    # Calculer le pourcentage de NaN
    pourcentage = (total_nan / len(df)) * 100

    # Créer un DataFrame résumé
    resume = pd.DataFrame({
        "Nb_NaN": total_nan,
        "Pourcentage (%)": pourcentage.round(2)
    })

    # Garder seulement les colonnes avec des NaN
    resume = resume[resume["Nb_NaN"] > 0]

    # Trier par pourcentage décroissant
    resume = resume.sort_values("Pourcentage (%)", ascending=False)

    return resume


# ============================================================
# 2️⃣ Détection des outliers avec la méthode IQR
# ============================================================

def detecter_outliers(df, colonne):
    """
    Détecte les valeurs extrêmes d'une colonne numérique
    en utilisant la méthode IQR.
    """

    # Vérifier que la colonne existe
    if colonne not in df.columns:
        raise ValueError("Colonne inexistante.")

    # Vérifier que la colonne est numérique
    if not pd.api.types.is_numeric_dtype(df[colonne]):
        raise TypeError("La colonne doit être numérique.")

    # Calcul des quartiles
    Q1 = df[colonne].quantile(0.25)   # 25%
    Q3 = df[colonne].quantile(0.75)   # 75%

    # Calcul de l'IQR (écart interquartile)
    IQR = Q3 - Q1

    # Définir les bornes
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR

    # Sélection des valeurs hors bornes
    outliers = df[(df[colonne] < borne_inf) | (df[colonne] > borne_sup)]

    print(f"Nombre d'outliers : {len(outliers)}")

    return len(outliers), borne_inf, borne_sup


# ============================================================
# 3️⃣ Sauvegarde d'un graphique
# ============================================================

def sauvegarder_graphique(nom_fichier, dossier="reports"):
    """
    Sauvegarde le graphique matplotlib actif
    dans un dossier.
    """

    # Créer le dossier s'il n'existe pas
    os.makedirs(dossier, exist_ok=True)

    # Construire le chemin complet
    chemin = os.path.join(dossier, nom_fichier)

    # Sauvegarder l'image
    plt.savefig(chemin, dpi=300, bbox_inches="tight")

    print(f"Graphique sauvegardé dans : {chemin}")


# ============================================================
# 4️⃣ Visualisation d'une distribution
# ============================================================

def afficher_distribution(df, colonne, bins=30):
    """
    Affiche un histogramme + boxplot
    pour analyser la distribution.
    """

    if colonne not in df.columns:
        raise ValueError("Colonne non trouvée.")

    if not pd.api.types.is_numeric_dtype(df[colonne]):
        raise TypeError("Colonne non numérique.")

    # Création de 2 graphiques côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogramme avec courbe de densité
    sns.histplot(df[colonne], bins=bins, kde=True, ax=axes[0])
    axes[0].set_title(f"Histogramme - {colonne}")

    # Boxplot pour visualiser les outliers
    sns.boxplot(x=df[colonne], ax=axes[1])
    axes[1].set_title(f"Boxplot - {colonne}")

    plt.tight_layout()  # Ajuste automatiquement les espaces
    plt.show()


# ============================================================
# 5️⃣ Résumé global du dataset
# ============================================================

def resume_dataset(df):
    """
    Affiche un résumé général du dataset.
    """

    print("=" * 50)
    print("RÉSUMÉ DU DATASET")
    print("=" * 50)

    print(f"Lignes      : {df.shape[0]}")   # Nombre de lignes
    print(f"Colonnes    : {df.shape[1]}")   # Nombre de colonnes
    print(f"Doublons    : {df.duplicated().sum()}")  # Nombre de doublons

    # Utilisation mémoire en MB
    memoire = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Mémoire     : {memoire:.2f} MB")

    print("\nTypes de colonnes :")
    print(df.dtypes.value_counts())

    print(f"\nTotal NaN : {df.isna().sum().sum()}")
    print("=" * 50)


# ============================================================
# 6️⃣ Analyse des corrélations
# ============================================================

def analyser_correlation(df, seuil=0.8, heatmap=False):
    """
    Affiche les paires de variables fortement corrélées.
    """

    # Sélectionner uniquement les colonnes numériques
    df_num = df.select_dtypes(include="number")

    if df_num.empty:
        print("Aucune colonne numérique.")
        return None

    # Calcul de la matrice de corrélation
    corr = df_num.corr().abs()  # .abs() pour valeur absolue

    # Option : afficher heatmap
    #Une heatmap (carte de chaleur) est un graphique qui représente des valeurs numériques avec des couleurs.
    #Plus la valeur est grande → plus la couleur est intense
    #Plus la valeur est petite → couleur plus claire
    if heatmap:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, cmap="coolwarm")
        plt.title("Matrice de corrélation")
        plt.show()

    paires = []

    # Parcourir la matrice
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] >= seuil:
                paires.append({
                    "Variable 1": corr.columns[i],
                    "Variable 2": corr.columns[j],
                    "Corrélation": round(corr.iloc[i, j], 3)
                })

    if paires:
        result = pd.DataFrame(paires).sort_values("Corrélation", ascending=False)
        return result
    else:
        print("Aucune corrélation forte détectée.")
        return None