"""
preprocessing.py – Version 2
Pipeline de préparation des données pour la prédiction du churn.
Architecture orientée objet : toutes les étapes sont encapsulées
dans la classe RetailPreprocessor.
"""

import logging
import os

import ipaddress
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ── Configuration du logger ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class RetailPreprocessor:
    """
    Encapsule l'ensemble du pipeline de prétraitement :
      - Nettoyage (aberrantes, inutiles, leakage)
      - Parsing (dates, IP)
      - Feature engineering
      - Réduction de la multicolinéarité
      - Encodage catégoriel
      - Split / Imputation / Standardisation
    """

    # Colonnes fixes à supprimer dès l'entrée (variance nulle ou simple ID)
    COLS_INUTILES = ['CustomerID', 'NewsletterSubscribed']

    # Features à exclure après le feature engineering (data leakage)
    COLS_LEAKAGE = [
        'ChurnRiskCategory', 'CustomerType', 'LoyaltyLevel',
        'SpendingCategory', 'RFMSegment', 'AccountStatus',
        'ReturnRatio', 'NegQtyCount', 'ZeroPriceCount',
        'CancelledTransactions', 'CustomerTenureDays',
        'FirstPurchase', 'Age', 'SupportTicketsCount',
        'SatisfactionScore', 'Recency', 'TenureRatio',
    ]

    # Encodage ordinal : ordre explicite des catégories
    ORDINAL_MAP = {
        'AgeCategory': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Inconnu'],
        'BasketSizeCategory': ['Petit', 'Moyen', 'Grand', 'Inconnu'],
        'PreferredTimeOfDay': ['Matin', 'Midi', 'Après-midi', 'Soir', 'Nuit'],
    }

    # Colonnes One-Hot (nominales sans ordre) – Country est traitée séparément
    COLS_ONEHOT = ['FavoriteSeason', 'Region', 'WeekendPreference',
                   'ProductDiversity', 'Gender']

    def __init__(self, seuil_corr: float = 0.8, test_size: float = 0.2,
                 random_state: int = 42):
        self.seuil_corr   = seuil_corr
        self.test_size    = test_size
        self.random_state = random_state

        # Artefacts appris sur le train (utiles pour l'inférence en production)
        self.scaler_: StandardScaler | None = None
        self.mediane_: pd.Series | None = None
        self.ohe_country_: OneHotEncoder | None = None

    # ── 1. Chargement ─────────────────────────────────────────────────────────
    def charger(self, chemin: str) -> pd.DataFrame:
        df = pd.read_csv(chemin)
        log.info("Fichier lu : %d lignes × %d colonnes", df.shape[0], df.shape[1])
        return df

    # ── 2. Suppression des colonnes non informatives ───────────────────────────
    def nettoyer_inutiles(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=self.COLS_INUTILES, errors='ignore')
        log.info("Colonnes non informatives retirées : %s", self.COLS_INUTILES)
        return df

    # ── 3. Correction des valeurs aberrantes ──────────────────────────────────
    def corriger_aberrantes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remplace les valeurs-sentinelles hors-plage par NaN :
          SupportTicketsCount : {-1, 999}
          SatisfactionScore   : {-1, 99}
        """
        regles = {
            'SupportTicketsCount': {999, -1},
            'SatisfactionScore':   {99,  -1},
        }
        for col, valeurs in regles.items():
            if col in df.columns:
                df[col] = df[col].replace({v: np.nan for v in valeurs})
        log.info("Valeurs aberrantes → NaN pour : %s", list(regles.keys()))
        return df

    # ── 4. Parsing de RegistrationDate ────────────────────────────────────────
    def parser_date_inscription(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convertit la date au format texte (JJ/MM/AA ou ISO) en 4 features :
        RegYear, RegMonth, RegDay, RegWeekday.
        """
        serie_date = pd.to_datetime(
            df['RegistrationDate'], dayfirst=True, errors='coerce'
        )
        df['RegYear']    = serie_date.dt.year
        df['RegMonth']   = serie_date.dt.month
        df['RegDay']     = serie_date.dt.day
        df['RegWeekday'] = serie_date.dt.weekday   # 0 = Lundi, 6 = Dimanche
        df = df.drop(columns=['RegistrationDate'])
        log.info("RegistrationDate décomposée → RegYear, RegMonth, RegDay, RegWeekday")
        return df

    # ── 5. Feature engineering depuis LastLoginIP ─────────────────────────────
    def transformer_ip(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée IsPrivateIP (1 = réseau interne, 0 = IP publique ou invalide)."""
        def _flag_prive(ip: str) -> int:
            try:
                return int(ipaddress.ip_address(str(ip)).is_private)
            except ValueError:
                return 0

        df['IsPrivateIP'] = df['LastLoginIP'].apply(_flag_prive)
        df = df.drop(columns=['LastLoginIP'])
        log.info("LastLoginIP → IsPrivateIP")
        return df

    # ── 6. Feature engineering métier ─────────────────────────────────────────
    def enrichir_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trois indicateurs construits à partir des variables RFM :
          MonetaryPerDay  : intensité de dépense journalière
          AvgBasketValue  : valeur moyenne du panier
          TenureRatio     : inactivité relative à l'ancienneté
        """
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
        df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']
        df['TenureRatio']    = df['Recency'] / (df['CustomerTenureDays'] + 1)
        log.info("Features créées : MonetaryPerDay, AvgBasketValue, TenureRatio")
        return df

    # ── 7. Suppression du data leakage ────────────────────────────────────────
    def supprimer_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=self.COLS_LEAKAGE, errors='ignore')
        log.info("Data leakage : %d colonnes supprimées", len(self.COLS_LEAKAGE))
        return df

    # ── 8. Réduction de la multicolinéarité ───────────────────────────────────
    def reduire_multicolinearite(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pour chaque paire de features numériques dont |r| > seuil_corr,
        retire la première (indice le plus élevé dans la matrice).
        """
        cols_num = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != 'Churn']
        if not cols_num:
            log.warning("Aucune colonne numérique disponible pour l'analyse de corrélation")
            return df

        mat = df[cols_num].corr().abs()
        a_retirer: set[str] = set()
        for i in range(len(mat.columns)):
            for j in range(i):
                if mat.iloc[i, j] > self.seuil_corr:
                    col_redondante = mat.columns[i]
                    a_retirer.add(col_redondante)
                    log.info("  Corrélation élevée : %s ↔ %s (r=%.2f) → suppression de %s",
                             col_redondante, mat.columns[j],
                             mat.iloc[i, j], col_redondante)

        df = df.drop(columns=list(a_retirer))
        log.info("Multicolinéarité : %d colonne(s) supprimée(s)", len(a_retirer))
        return df

    # ── 9. Encodage catégoriel (hors Country) ─────────────────────────────────
    def encoder_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordinal pour les variables ordonnées, One-Hot pour les nominales."""
        # Ordinal
        for col, ordre in self.ORDINAL_MAP.items():
            if col in df.columns:
                enc = OrdinalEncoder(
                    categories=[ordre],
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                )
                df[col] = enc.fit_transform(df[[col]])
                log.info("  Ordinal → %s", col)

        # One-Hot
        presents = [c for c in self.COLS_ONEHOT if c in df.columns]
        df = pd.get_dummies(df, columns=presents, drop_first=False)
        log.info("One-Hot → %s", presents)
        return df

    # ── 10. Split / Country OHE / Imputation / Scaling ────────────────────────
    def split_et_transformer(self, df: pd.DataFrame):
        """
        Étape finale :
          - Séparation stratifiée (80 % train, 20 % test)
          - One-Hot de Country après split (évite la fuite)
          - Imputation par la médiane du train
          - Standardisation (StandardScaler fitté sur le train)
          - Sauvegarde des CSV et artefacts
        """
        X = df.drop(columns=['Churn'])
        y = df['Churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        log.info("Split → train %s | test %s", X_train.shape, X_test.shape)

        # One-Hot de Country
        if 'Country' in X_train.columns:
            pool = pd.concat([X_train[['Country']], X_test[['Country']]])
            self.ohe_country_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.ohe_country_.fit(pool[['Country']])

            country_cols = [f'Country_{c}' for c in self.ohe_country_.categories_[0]]
            train_c = pd.DataFrame(self.ohe_country_.transform(X_train[['Country']]),
                                   columns=country_cols)
            test_c  = pd.DataFrame(self.ohe_country_.transform(X_test[['Country']]),
                                   columns=country_cols)

            X_train = pd.concat(
                [X_train.drop(columns=['Country']).reset_index(drop=True), train_c], axis=1
            )
            X_test = pd.concat(
                [X_test.drop(columns=['Country']).reset_index(drop=True), test_c], axis=1
            )
            log.info("One-Hot Country : %d modalités", len(self.ohe_country_.categories_[0]))
        else:
            log.warning("Colonne 'Country' absente")

        # Imputation
        self.mediane_ = X_train.median()
        X_train = X_train.fillna(self.mediane_)
        X_test  = X_test.fillna(self.mediane_)
        log.info("Imputation par la médiane du train")

        # Standardisation
        self.scaler_ = StandardScaler()
        X_train_sc = pd.DataFrame(
            self.scaler_.fit_transform(X_train), columns=X_train.columns
        )
        X_test_sc = pd.DataFrame(
            self.scaler_.transform(X_test), columns=X_test.columns
        )
        log.info("Standardisation appliquée")

        # Sauvegarde
        self._sauvegarder(X_train_sc, X_test_sc, y_train, y_test)

        return X_train_sc, X_test_sc, y_train, y_test

    # ── Méthodes utilitaires ───────────────────────────────────────────────────
    def _sauvegarder(self, X_train, X_test, y_train, y_test) -> None:
        os.makedirs('data/train_test', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        X_train.to_csv('data/train_test/X_train.csv', index=False)
        X_test.to_csv('data/train_test/X_test.csv',   index=False)
        y_train.to_csv('data/train_test/y_train.csv', index=False)
        y_test.to_csv('data/train_test/y_test.csv',   index=False)

        joblib.dump(self.scaler_,  'models/scaler.pkl')
        joblib.dump(self.mediane_, 'models/mediane_train.pkl')
        if self.ohe_country_:
            joblib.dump(self.ohe_country_, 'models/ohe_country.pkl')

        log.info("Artefacts sauvegardés dans data/train_test/ et models/")

    def executer_pipeline(self, chemin: str):
        """Lance l'intégralité du pipeline en une seule instruction."""
        df = self.charger(chemin)
        df = self.nettoyer_inutiles(df)
        df = self.corriger_aberrantes(df)
        df = self.parser_date_inscription(df)
        df = self.transformer_ip(df)
        df = self.enrichir_features(df)
        df = self.supprimer_leakage(df)
        df = self.reduire_multicolinearite(df)
        df = self.encoder_categories(df)

        os.makedirs('data/processed', exist_ok=True)
        df.to_csv('data/processed/data_clean.csv', index=False)
        log.info("data_clean.csv enregistré — shape : %s", df.shape)

        return self.split_et_transformer(df)


# ── Point d'entrée ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    preprocessor = RetailPreprocessor(seuil_corr=0.8, test_size=0.2, random_state=42)

    X_train, X_test, y_train, y_test = preprocessor.executer_pipeline(
        chemin='data/raw/retail_customers_COMPLETE_CATEGORICAL.csv'
    )

    log.info("Pipeline terminé — train : %s | test : %s",
             X_train.shape, X_test.shape)
