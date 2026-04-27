"""
utils.py — Version 2
Utilitaires encapsulés dans deux classes distinctes :
  - DataIO   : chargement et sauvegarde (données + modèles)
  - Plotter  : toutes les visualisations
Logger Python à la place des print().
"""

import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Logger ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# I/O — Chargement & Sauvegarde
# ══════════════════════════════════════════════════════════════
class DataIO:
    """Centralise toutes les opérations de lecture/écriture."""

    DOSSIER_MODELS   = 'models'
    DOSSIER_DATA     = 'data/train_test'
    DOSSIER_REPORTS  = 'reports'

    # ── Données ───────────────────────────────────────────────
    @staticmethod
    def charger_csv(chemin: str) -> pd.DataFrame:
        """Lit un fichier CSV et retourne un DataFrame."""
        df = pd.read_csv(chemin)
        log.info("CSV chargé : %s — %d lignes × %d colonnes",
                 chemin, df.shape[0], df.shape[1])
        return df

    @classmethod
    def charger_train_test(
        cls, dossier: str | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Charge les quatre fichiers du split train/test."""
        rep = dossier or cls.DOSSIER_DATA
        X_train = pd.read_csv(f'{rep}/X_train.csv')
        X_test  = pd.read_csv(f'{rep}/X_test.csv')
        y_train = pd.read_csv(f'{rep}/y_train.csv').squeeze()
        y_test  = pd.read_csv(f'{rep}/y_test.csv').squeeze()
        log.info("Train : %s | Test : %s", X_train.shape, X_test.shape)
        return X_train, X_test, y_train, y_test

    # ── Modèles ───────────────────────────────────────────────
    @classmethod
    def charger_modele(cls, nom_fichier: str, dossier: str | None = None):
        """Désérialise un artefact joblib."""
        chemin = f'{dossier or cls.DOSSIER_MODELS}/{nom_fichier}'
        artefact = joblib.load(chemin)
        log.info("Modèle chargé : %s", chemin)
        return artefact

    @classmethod
    def sauvegarder_modele(cls, modele, nom_fichier: str,
                           dossier: str | None = None) -> None:
        """Sérialise un modèle avec joblib."""
        rep = dossier or cls.DOSSIER_MODELS
        os.makedirs(rep, exist_ok=True)
        chemin = f'{rep}/{nom_fichier}'
        joblib.dump(modele, chemin)
        log.info("Modèle sauvegardé : %s", chemin)

    @classmethod
    def sauvegarder_figure(cls, nom_fichier: str,
                           dossier: str | None = None, dpi: int = 150) -> None:
        """Enregistre la figure matplotlib courante puis la ferme."""
        rep = dossier or cls.DOSSIER_REPORTS
        os.makedirs(rep, exist_ok=True)
        chemin = f'{rep}/{nom_fichier}'
        plt.savefig(chemin, bbox_inches='tight', dpi=dpi)
        plt.close()
        log.info("Figure sauvegardée : %s", chemin)


# ── Fonctions libres pour la rétrocompatibilité ──────────────────────────────
def charger_train_test(dossier: str | None = None):
    return DataIO.charger_train_test(dossier)

def sauvegarder_modele(modele, nom_fichier: str, dossier: str | None = None):
    DataIO.sauvegarder_modele(modele, nom_fichier, dossier)

def sauvegarder_figure(nom_fichier: str, dossier: str | None = None, dpi: int = 150):
    DataIO.sauvegarder_figure(nom_fichier, dossier, dpi)


# ══════════════════════════════════════════════════════════════
# Visualisations
# ══════════════════════════════════════════════════════════════
class Plotter:
    """Génère et sauvegarde les visualisations du projet."""

    def __init__(self, dossier_reports: str = 'reports', dpi: int = 150):
        self.dossier = dossier_reports
        self.dpi = dpi

    def _sauvegarder(self, nom: str) -> None:
        DataIO.sauvegarder_figure(nom, dossier=self.dossier, dpi=self.dpi)

    def importance_features(
        self,
        modele,
        noms_features: list[str],
        top_n: int = 20,
        couleur: str = 'steelblue',
    ) -> None:
        """Bar chart des `top_n` features les plus importantes."""
        importances = (
            pd.Series(modele.feature_importances_, index=noms_features)
            .sort_values(ascending=False)
            .head(top_n)
        )
        plt.figure(figsize=(10, 6))
        importances.plot(kind='bar', color=couleur, edgecolor='white')
        plt.title(f'Top {top_n} features les plus importantes')
        plt.ylabel('Importance')
        plt.tight_layout()
        self._sauvegarder('feature_importance.png')

    def distribution_cible(
        self,
        y: pd.Series,
        titre: str = 'Distribution de Churn',
    ) -> None:
        """Camembert Fidèle / Churner."""
        counts = y.value_counts()
        labels = [
            f'Fidèle (0)\n{counts.get(0, 0)}',
            f'Churner (1)\n{counts.get(1, 0)}',
        ]
        plt.figure(figsize=(5, 5))
        plt.pie(counts, labels=labels, autopct='%1.1f%%',
                colors=['#4C72B0', '#DD8452'], startangle=90)
        plt.title(titre)
        plt.tight_layout()
        self._sauvegarder('distribution_churn.png')

    def heatmap_correlation(
        self,
        df: pd.DataFrame,
        top_n: int = 20,
    ) -> None:
        """
        Heatmap de corrélation sur les `top_n` colonnes numériques
        les plus corrélées à 'Churn' (si présente, sinon toutes).
        """
        numeriques = df.select_dtypes(include=[np.number])
        if 'Churn' in numeriques.columns:
            top_cols = (
                numeriques.corr()['Churn'].abs()
                .sort_values(ascending=False)
                .head(top_n)
                .index.tolist()
            )
        else:
            top_cols = numeriques.columns[:top_n].tolist()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            numeriques[top_cols].corr(),
            annot=False, cmap='coolwarm', center=0,
            linewidths=0.3, square=True,
        )
        plt.title(f'Matrice de corrélation (top {top_n} features)')
        plt.tight_layout()
        self._sauvegarder('heatmap_correlation.png')
