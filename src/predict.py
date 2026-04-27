import logging
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import charger_train_test

# ── Logger ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Colonnes à exclure (leakage résiduel éventuel)
COLS_EXCLURE = ['ChurnRiskCategory', 'CustomerType_Perdu']

# Labels des segments K-Means
LABELS_SEGMENTS = {
    0: 'Segment A — Clients Premium',
    1: 'Segment B — Clients Réguliers',
    2: 'Segment C — Clients Occasionnels',
    3: 'Segment D — Clients Inactifs',
}


class PredictionPipeline:
    """
    Encapsule le chargement des artefacts et toutes les tâches d'inférence :
      - Prédiction du churn (Random Forest)
      - Attribution du segment (K-Means via ACP)
      - Évaluation complète sur X_test
      - Démonstration sur un client fictif
    Les artefacts sont chargés une seule fois à l'instanciation.
    """

    def __init__(self, dossier_models: str = 'models',
                 dossier_data: str = 'data/train_test'):
        self.dossier_models = dossier_models
        self.dossier_data   = dossier_data

        # Artefacts chargés lors de l'initialisation
        self.scaler_     = None
        self.rf_         = None
        self.xgb_        = None
        self.stacking_   = None
        self.kmeans_     = None
        self.pca_        = None
        self.reg_        = None
        self.reg_scaler_ = None

        self._charger_artefacts()

    # ── Chargement ────────────────────────────────────────────────────────
    def _charger_artefacts(self) -> None:
        """Charge tous les modèles et scalers depuis `dossier_models`."""
        def _load(nom):
            return joblib.load(f'{self.dossier_models}/{nom}')

        self.scaler_     = _load('scaler.pkl')
        self.rf_         = _load('random_forest.pkl')
        self.xgb_        = _load('xgboost.pkl')
        self.stacking_   = _load('stacking.pkl')
        self.kmeans_     = _load('kmeans.pkl')
        self.pca_        = _load('pca.pkl')
        self.reg_        = _load('regression_xgboost_optimized.pkl')
        self.reg_scaler_ = _load('scaler_regression.pkl')

        log.info("Artefacts chargés depuis '%s' : scaler, rf, xgb, stacking, "
                 "kmeans, pca, regression_xgboost, scaler_regression",
                 self.dossier_models)

    # ── Pré-traitement local ──────────────────────────────────────────────
    def _preparer_client(self, client_df: pd.DataFrame) -> np.ndarray:
        """
        Supprime les colonnes à risque de leakage, re-standardise
        en utilisant les statistiques de X_train.
        """
        X_train = pd.read_csv(f'{self.dossier_data}/X_train.csv')
        X_train = X_train.drop(columns=COLS_EXCLURE, errors='ignore')
        client  = client_df.drop(columns=COLS_EXCLURE, errors='ignore')

        sc = StandardScaler()
        sc.fit(X_train)
        return sc.transform(client)

    # ── Prédiction Churn ─────────────────────────────────────────────────
    def predire_churn(self, client_df: pd.DataFrame) -> dict:
        """
        Retourne la prédiction de churn (RF) avec probabilités.
        Format : {'churn_predit', 'label', 'prob_fidele', 'prob_churner'}
        """
        client_sc    = self._preparer_client(client_df)
        prediction   = self.rf_.predict(client_sc)[0]
        probabilites = self.rf_.predict_proba(client_sc)[0]

        return {
            'churn_predit':  int(prediction),
            'label':         'Churner' if prediction == 1 else 'Fidèle',
            'prob_fidele':   round(float(probabilites[0]) * 100, 1),
            'prob_churner':  round(float(probabilites[1]) * 100, 1),
        }

    # ── Prédiction Segment ────────────────────────────────────────────────
    def predire_segment(self, client_df: pd.DataFrame) -> dict:
        """
        Projette le client dans l'espace ACP puis retourne son segment K-Means.
        Format : {'segment_id', 'segment_label'}
        """
        client_sc  = self.scaler_.transform(client_df)
        client_pca = self.pca_.transform(client_sc)
        segment    = int(self.kmeans_.predict(client_pca)[0])

        return {
            'segment_id':    segment,
            'segment_label': LABELS_SEGMENTS.get(segment, f'Segment {segment}'),
        }

    # ── Évaluation sur X_test ─────────────────────────────────────────────
    def evaluer_test(self) -> pd.DataFrame:
        """
        Applique le Random Forest sur l'ensemble du test set,
        calcule l'accuracy et sauvegarde un CSV de résultats.
        """
        log.info("=" * 50)
        log.info("PRÉDICTIONS SUR X_TEST (Random Forest)")
        log.info("=" * 50)

        X_test  = pd.read_csv(f'{self.dossier_data}/X_test.csv')
        y_test  = pd.read_csv(f'{self.dossier_data}/y_test.csv').squeeze()
        X_train = pd.read_csv(f'{self.dossier_data}/X_train.csv')

        X_test_clf  = X_test.drop(columns=COLS_EXCLURE,  errors='ignore')
        X_train_clf = X_train.drop(columns=COLS_EXCLURE, errors='ignore')

        sc = StandardScaler()
        sc.fit(X_train_clf)
        X_test_sc = sc.transform(X_test_clf)

        predictions  = self.rf_.predict(X_test_sc)
        probabilites = self.rf_.predict_proba(X_test_sc)

        rapport = pd.DataFrame({
            'Churn_Réel':    y_test.values,
            'Churn_Prédit':  predictions,
            'Prob_Fidèle':   (probabilites[:, 0] * 100).round(1),
            'Prob_Churner':  (probabilites[:, 1] * 100).round(1),
        })

        print(f"\n Aperçu des 10 premières prédictions :")
        print(rapport.head(10).to_string(index=False))

        nb_ok = (rapport['Churn_Réel'] == rapport['Churn_Prédit']).sum()
        total = len(rapport)
        log.info("Prédictions correctes : %d/%d (%.1f %%)",
                 nb_ok, total, nb_ok / total * 100)

        os.makedirs('reports', exist_ok=True)
        rapport.to_csv('reports/predictions_test.csv', index=False)
        log.info("Rapport CSV sauvegardé : reports/predictions_test.csv")
        return rapport

    # ── Démo client fictif ────────────────────────────────────────────────
    def demo_client(self) -> None:
        """
        Construit un client fictif à partir des moyennes de X_train
        et affiche son risque de churn ainsi que son segment.
        """
        log.info("=" * 50)
        log.info("DÉMO — CLIENT FICTIF (moyennes du train)")
        log.info("=" * 50)

        X_train = pd.read_csv(f'{self.dossier_data}/X_train.csv')
        client  = pd.DataFrame([X_train.mean()], columns=X_train.columns)

        res_churn   = self.predire_churn(client.copy())
        res_segment = self.predire_segment(client.copy())

        print(f"\n👤 Profil client fictif (moyennes X_train) :")
        print(f"   Risque churn  : {res_churn['label']}")
        print(f"   Prob. Fidèle  : {res_churn['prob_fidele']} %")
        print(f"   Prob. Churner : {res_churn['prob_churner']} %")
        print(f"   Segment       : {res_segment['segment_label']}")

    # ── Orchestration complète ────────────────────────────────────────────
    def run(self) -> None:
        """Lance l'évaluation sur X_test puis la démo client fictif."""
        self.evaluer_test()
        self.demo_client()
        log.info("predict.py terminé avec succès.")


# ──────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = PredictionPipeline(
        dossier_models='models',
        dossier_data='data/train_test',
    )
    pipeline.run()
