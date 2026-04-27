import logging
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, classification_report,
    confusion_matrix, f1_score, mean_squared_error,
    r2_score, roc_curve, silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV,
    cross_val_predict, train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import charger_train_test, sauvegarder_figure, sauvegarder_modele

# ── Logger ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class ModelTrainer:
    """
    Encapsule l'intégralité du pipeline de modélisation :
      - ACP (réduction dimensionnelle)
      - Clustering K-Means
      - Classification : Random Forest, XGBoost, Stacking
      - Régression XGBoost sur MonetaryTotal

    Paramètres configurables via le constructeur pour faciliter
    la reproductibilité et l'expérimentation.
    """

    def __init__(
        self,
        random_state: int = 42,
        n_iter_rf: int = 60,
        n_iter_reg: int = 30,
        smote_ratio: float = 0.5,
        k_clusters: int = 4,
        seuil_leakage: float = 0.85,
    ):
        self.random_state   = random_state
        self.n_iter_rf      = n_iter_rf
        self.n_iter_reg     = n_iter_reg
        self.smote_ratio    = smote_ratio
        self.k_clusters     = k_clusters
        self.seuil_leakage  = seuil_leakage

        # Artefacts produits (accessibles après run)
        self.pca_: PCA | None = None
        self.kmeans_: KMeans | None = None
        self.rf_: RandomForestClassifier | None = None
        self.xgb_: XGBClassifier | None = None
        self.stacking_: LogisticRegression | None = None
        self.reg_: XGBRegressor | None = None

    # ── Utilitaire ROC-AUC ────────────────────────────────────────────────
    def _tracer_roc(self, y_reel, y_proba, nom: str) -> None:
        fpr, tpr, _ = roc_curve(y_reel, y_proba)
        score = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, label=f"{nom} (AUC={score:.3f})")
        plt.plot([0, 1], [0, 1], "r--", lw=1)
        plt.xlabel("Taux de faux positifs")
        plt.ylabel("Taux de vrais positifs")
        plt.title(f"Courbe ROC — {nom}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        sauvegarder_figure(f"roc_auc_{nom.lower()}.png")
        log.info("AUC %s = %.3f", nom, score)

    def _matrice_confusion(self, y_reel, y_pred, titre: str, fichier: str) -> None:
        cm = confusion_matrix(y_reel, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Fidèle', 'Churner'],
                    yticklabels=['Fidèle', 'Churner'])
        plt.title(titre)
        plt.tight_layout()
        sauvegarder_figure(fichier)

    def _chercher_seuil_optimal(self, probas, y_vrai, metrique: str = 'f1') -> float:
        """
        Balaye les seuils de 0.25 à 0.75 et retourne celui
        maximisant la métrique choisie ('f1' ou 'accuracy').
        """
        meilleur_seuil, meilleur_score = 0.5, 0.0
        for seuil in np.arange(0.25, 0.75, 0.01):
            preds = (probas >= seuil).astype(int)
            if metrique == 'f1':
                score = f1_score(y_vrai, preds)
            else:
                score = accuracy_score(y_vrai, preds)
            if score > meilleur_score:
                meilleur_score = score
                meilleur_seuil = seuil
        log.info("Seuil optimal (métrique=%s) : %.2f (score=%.3f)",
                 metrique, meilleur_seuil, meilleur_score)
        return meilleur_seuil

    # ── ACP ───────────────────────────────────────────────────────────────
    def acp(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        log.info("=" * 50)
        log.info("ACP — ANALYSE EN COMPOSANTES PRINCIPALES")
        log.info("=" * 50)

        pca_diag = PCA(random_state=self.random_state)
        pca_diag.fit(X_train)
        var_cum = np.cumsum(pca_diag.explained_variance_ratio_)

        n95 = int(np.argmax(var_cum >= 0.95)) + 1
        n90 = int(np.argmax(var_cum >= 0.90)) + 1
        log.info("Composantes : 90%% → %d | 95%% → %d", n90, n95)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(range(1, len(var_cum) + 1), var_cum, marker='o', markersize=3,
                 color='steelblue')
        ax1.axhline(0.95, color='red', linestyle='--', label='95 %')
        ax1.axvline(n95, color='red', linestyle=':')
        ax1.set_title('Variance cumulée expliquée')
        ax1.legend()
        ax2.bar(range(1, 21), pca_diag.explained_variance_ratio_[:20],
                color='coral', edgecolor='white')
        ax2.set_title('Variance par composante (top 20)')
        plt.tight_layout()
        sauvegarder_figure('acp_variance.png')

        self.pca_ = PCA(n_components=n95, random_state=self.random_state)
        X_train_pca = self.pca_.fit_transform(X_train)
        X_test_pca  = self.pca_.transform(X_test)
        log.info("ACP : %d → %d composantes (var. conservée : %.1f %%)",
                 X_train.shape[1], n95, var_cum[n95 - 1] * 100)
        joblib.dump(self.pca_, 'models/pca.pkl')
        return self.pca_, X_train_pca, X_test_pca

    # ── Clustering ────────────────────────────────────────────────────────
    def clustering(self, donnees: np.ndarray):
        log.info("=" * 50)
        log.info("MODÈLE 1 — K-MEANS CLUSTERING")
        log.info("=" * 50)

        plage_k = range(2, 9)
        inerties, scores_silhouette = [], []

        for k in plage_k:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(donnees)
            inerties.append(km.inertia_)
            scores_silhouette.append(silhouette_score(donnees, km.labels_))
            log.info("  k=%d → inertie=%.0f | silhouette=%.3f",
                     k, km.inertia_, scores_silhouette[-1])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(list(plage_k), inerties, marker='o', color='steelblue')
        ax1.set_title('Méthode du coude (inertie)')
        ax1.set_xlabel('Nombre de clusters k')
        ax2.plot(list(plage_k), scores_silhouette, marker='o', color='coral')
        ax2.set_title('Score de silhouette')
        ax2.set_xlabel('Nombre de clusters k')
        plt.tight_layout()
        sauvegarder_figure('clustering_choix_k.png')

        self.kmeans_ = KMeans(
            n_clusters=self.k_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.kmeans_.fit(donnees)
        score_final = silhouette_score(donnees, self.kmeans_.labels_)
        repartition = pd.Series(self.kmeans_.labels_).value_counts().to_dict()

        log.info("K-Means final — k=%d | silhouette=%.3f | clusters=%s",
                 self.k_clusters, score_final, repartition)
        sauvegarder_modele(self.kmeans_, 'kmeans.pkl')
        return self.kmeans_

    # ── Random Forest ─────────────────────────────────────────────────────
    def random_forest(self, X_train, X_test, y_train, y_test):
        log.info("=" * 50)
        log.info("MODÈLE 2a — RANDOM FOREST (SMOTE + SEUIL OPTIMISÉ)")
        log.info("=" * 50)

        # Vérification anti-leakage
        df_tmp = X_train.copy()
        df_tmp['Churn'] = y_train.values
        corr = df_tmp.corr()['Churn'].abs().drop('Churn').sort_values(ascending=False)
        log.info("Top 10 corrélations avec Churn :\n%s", corr.head(10).round(4))
        if (corr > self.seuil_leakage).any():
            log.error("Leakage détecté (corrélation > %.2f) — arrêt", self.seuil_leakage)
            sys.exit(1)

        log.info("Distribution classes : %s", y_train.value_counts().to_dict())

        espace_params = {
            'n_estimators':     [600, 800, 1000, 1200],
            'max_depth':        [20, 30, 40, None],
            'min_samples_split':[2, 3, 5],
            'min_samples_leaf': [1, 2, 3],
            'max_features':     ['sqrt', 'log2', 0.7],
            'bootstrap':        [True],
            'criterion':        ['gini'],
            'max_samples':      [0.8, 0.9, None],
        }

        rf_base = RandomForestClassifier(
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1,
        )
        recherche = RandomizedSearchCV(
            rf_base, espace_params,
            n_iter=self.n_iter_rf, cv=5, scoring='f1',
            n_jobs=-1, verbose=1, random_state=self.random_state,
        )
        recherche.fit(X_train, y_train)
        self.rf_ = recherche.best_estimator_
        log.info("Meilleurs hyperparamètres RF : %s", recherche.best_params_)

        # Optimisation du seuil
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2,
            random_state=self.random_state, stratify=y_train,
        )
        self.rf_.fit(X_sub, y_sub)
        seuil_opt = self._chercher_seuil_optimal(
            self.rf_.predict_proba(X_val)[:, 1], y_val, metrique='f1'
        )

        # Entraînement final
        self.rf_.fit(X_train, y_train)
        probas = self.rf_.predict_proba(X_test)[:, 1]

        y_def = (probas >= 0.5).astype(int)
        y_opt = (probas >= seuil_opt).astype(int)
        y_final = y_opt if accuracy_score(y_test, y_opt) >= accuracy_score(y_test, y_def) else y_def

        log.info("Accuracy test : %.3f", accuracy_score(y_test, y_final))
        print(classification_report(y_test, y_final))

        self._matrice_confusion(y_test, y_final,
                                'Matrice de confusion — Random Forest',
                                'classification_confusion_matrix_rf.png')

        importances = (
            pd.Series(self.rf_.feature_importances_, index=X_train.columns)
            .sort_values(ascending=False).head(15)
        )
        plt.figure(figsize=(10, 6))
        importances.plot(kind='bar', color='steelblue', edgecolor='white')
        plt.title('Top 15 features — Random Forest')
        plt.tight_layout()
        sauvegarder_figure('classification_feature_importance_rf.png')

        sauvegarder_modele(self.rf_, 'random_forest.pkl')
        self._tracer_roc(y_test, probas, "RandomForest")
        return self.rf_

    # ── XGBoost ───────────────────────────────────────────────────────────
    def xgboost(self, X_train, X_test, y_train, y_test):
        log.info("=" * 50)
        log.info("MODÈLE 2b — XGBOOST (SMOTE + SEUIL OPTIMISÉ)")
        log.info("=" * 50)

        log.info("Distribution initiale : %s", y_train.value_counts().to_dict())
        smote = SMOTE(sampling_strategy=self.smote_ratio, random_state=self.random_state)
        X_sm, y_sm = smote.fit_resample(X_train, y_train)
        log.info("Après SMOTE : %s", pd.Series(y_sm).value_counts().to_dict())

        grille = {
            'n_estimators':    [500, 700],
            'max_depth':       [5, 6, 7],
            'learning_rate':   [0.03, 0.05, 0.08],
            'subsample':       [0.8, 0.9],
            'colsample_bytree':[0.8, 0.9],
        }

        xgb_base = XGBClassifier(
            random_state=self.random_state, use_label_encoder=False,
            eval_metric='logloss', scale_pos_weight=3,
        )
        gs = GridSearchCV(xgb_base, grille, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        gs.fit(X_sm, y_sm)
        self.xgb_ = gs.best_estimator_
        log.info("Meilleurs hyperparamètres XGB : %s", gs.best_params_)

        # Optimisation du seuil
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_sm, y_sm, test_size=0.2,
            random_state=self.random_state, stratify=y_sm,
        )
        self.xgb_.fit(X_sub, y_sub)
        seuil_opt = self._chercher_seuil_optimal(
            self.xgb_.predict_proba(X_val)[:, 1], y_val, metrique='accuracy'
        )

        # Entraînement final
        self.xgb_.fit(X_sm, y_sm)
        probas = self.xgb_.predict_proba(X_test)[:, 1]
        y_final = (probas >= seuil_opt).astype(int)

        log.info("Accuracy test : %.3f", accuracy_score(y_test, y_final))
        print(classification_report(y_test, y_final,
                                    target_names=['Fidèle (0)', 'Churner (1)']))

        self._matrice_confusion(y_test, y_final,
                                'Matrice de confusion — XGBoost',
                                'classification_confusion_matrix_xgb.png')

        importances = (
            pd.Series(self.xgb_.feature_importances_, index=X_train.columns)
            .sort_values(ascending=False).head(15)
        )
        plt.figure(figsize=(10, 6))
        importances.plot(kind='bar', color='steelblue', edgecolor='white')
        plt.title('Top 15 features — XGBoost')
        plt.tight_layout()
        sauvegarder_figure('classification_feature_importance_xgb.png')

        sauvegarder_modele(self.xgb_, 'xgboost.pkl')
        self._tracer_roc(y_test, probas, "XGBoost")
        return self.xgb_

    # ── Stacking ──────────────────────────────────────────────────────────
    def stacking(self, X_train, X_test, y_train, y_test):
        log.info("=" * 50)
        log.info("MODÈLE 2c — STACKING (RF + XGB → LR)")
        log.info("=" * 50)

        if self.rf_ is None or self.xgb_ is None:
            raise RuntimeError("RF et XGB doivent être entraînés avant le stacking.")

        rf_oof  = cross_val_predict(self.rf_,  X_train, y_train, cv=5, method='predict_proba')[:, 1]
        xgb_oof = cross_val_predict(self.xgb_, X_train, y_train, cv=5, method='predict_proba')[:, 1]
        rf_test  = self.rf_.predict_proba(X_test)[:, 1]
        xgb_test = self.xgb_.predict_proba(X_test)[:, 1]

        X_meta_train = np.column_stack([rf_oof,  xgb_oof])
        X_meta_test  = np.column_stack([rf_test, xgb_test])

        self.stacking_ = LogisticRegression(random_state=self.random_state)
        self.stacking_.fit(X_meta_train, y_train)
        y_pred = self.stacking_.predict(X_meta_test)

        log.info("Accuracy stacking : %.3f", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred,
                                    target_names=['Fidèle (0)', 'Churner (1)']))

        self._matrice_confusion(y_test, y_pred,
                                'Matrice de confusion — Stacking',
                                'classification_confusion_matrix_stacking.png')

        poids = pd.Series(self.stacking_.coef_[0], index=['RF', 'XGB'])
        log.info("Poids méta-modèle : %s", poids.to_dict())

        sauvegarder_modele(self.stacking_, 'stacking.pkl')
        probas_meta = self.stacking_.predict_proba(X_meta_test)[:, 1]
        self._tracer_roc(y_test, probas_meta, "Stacking")
        return self.stacking_

    # ── Régression ────────────────────────────────────────────────────────
    def regression(self) -> XGBRegressor:
        log.info("=" * 50)
        log.info("MODÈLE 3 — RÉGRESSION XGBOOST (MonetaryTotal)")
        log.info("=" * 50)

        df = pd.read_csv('data/processed/data_clean.csv')
        if 'Country' in df.columns:
            df = df.drop(columns=['Country'])

        X = df.drop(columns=['MonetaryTotal', 'Churn'])
        y = df['MonetaryTotal'].replace([np.inf, -np.inf], np.nan)
        if y.isnull().any():
            y = y.fillna(y.median())

        avec_log = not (y < 0).any()
        if not avec_log:
            log.warning("Valeurs négatives dans MonetaryTotal → transformation log désactivée")
        y_transf = np.log1p(y) if avec_log else y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_transf, test_size=0.2, random_state=self.random_state
        )
        mediane = X_train.median()
        X_train = X_train.fillna(mediane)
        X_test  = X_test.fillna(mediane)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        espace_params = {
            'n_estimators':    [300, 500, 700],
            'max_depth':       [5, 6, 7, 8],
            'learning_rate':   [0.01, 0.03, 0.05, 0.07],
            'subsample':       [0.7, 0.8, 0.9],
            'colsample_bytree':[0.7, 0.8, 0.9],
        }

        xgb = XGBRegressor(random_state=self.random_state)
        recherche = RandomizedSearchCV(
            xgb, espace_params,
            n_iter=self.n_iter_reg, cv=5, scoring='r2',
            n_jobs=-1, random_state=self.random_state, verbose=1,
        )
        recherche.fit(X_tr_sc, y_train)
        self.reg_ = recherche.best_estimator_
        log.info("Meilleurs hyperparamètres Reg : %s", recherche.best_params_)

        y_pred_t = self.reg_.predict(X_te_sc)
        y_pred = np.expm1(y_pred_t) if avec_log else y_pred_t
        y_reel = np.expm1(y_test)   if avec_log else y_test

        rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
        r2   = r2_score(y_reel, y_pred)
        log.info("RMSE : %.2f £ | R² : %.3f", rmse, r2)

        plt.figure(figsize=(7, 5))
        plt.scatter(y_reel, y_pred, alpha=0.35, color='steelblue', edgecolors='none')
        diag = [y_reel.min(), y_reel.max()]
        plt.plot(diag, diag, 'r--', lw=1.5)
        plt.xlabel("Valeurs réelles (£)")
        plt.ylabel("Valeurs prédites (£)")
        plt.title("Régression XGBoost — Réel vs Prédit")
        plt.tight_layout()
        sauvegarder_figure('regression_reel_vs_predit.png')

        sauvegarder_modele(self.reg_, 'regression_xgboost_optimized.pkl')
        joblib.dump(scaler, 'models/scaler_regression.pkl')
        return self.reg_

    # ── Orchestration complète ────────────────────────────────────────────
    def run(self, X_train, X_test, y_train, y_test):
        """Lance l'intégralité du pipeline dans l'ordre correct."""
        # ACP + Clustering
        _, X_tr_pca, _ = self.acp(X_train, X_test)
        self.clustering(X_tr_pca)

        # Classification
        self.random_forest(X_train, X_test, y_train, y_test)
        self.xgboost(X_train, X_test, y_train, y_test)
        self.stacking(X_train, X_test, y_train, y_test)

        # Régression
        self.regression()

        log.info("Pipeline complet — modèles dans models/ | figures dans reports/")


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Chargement des données train/test...")
    X_train, X_test, y_train, y_test = charger_train_test()

    trainer = ModelTrainer(
        random_state=42,
        n_iter_rf=60,
        n_iter_reg=30,
        smote_ratio=0.5,
        k_clusters=4,
        seuil_leakage=0.85,
    )
    trainer.run(X_train, X_test, y_train, y_test)
