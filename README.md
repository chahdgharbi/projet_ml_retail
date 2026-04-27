# projet_ml_retail
# Projet ML Retail — Analyse Comportementale Clientèle

> **Module** : Atelier Machine Learning — GI2  
> **Année** : 2025–2026  
> **Auteure** : Chahd Gharbi  
> **Encadrante** : Mme Fadoua Drira  
> **Dépôt** : [github.com/chahdgharbi/projet_ml_retail](https://github.com/chahdgharbi/projet_ml_retail)

---

## 1. Présentation

Ce projet applique une chaîne complète de traitement en Data Science sur un jeu de données e-commerce réel, comprenant **4 372 clients** décrits par **52 variables**. La base est volontairement imparfaite (valeurs manquantes, outliers, data leakage) afin de couvrir l'intégralité du workflow :

```
Exploration  ──►  Préparation  ──►  Modélisation  ──►  Évaluation  ──►  Déploiement
```

L'entreprise cherche à **réduire le churn**, **segmenter sa clientèle** et **optimiser son chiffre d'affaires**.

---

## 2. Organisation du projet

```
projet_ml_retail/
│
├── data/
│   ├── raw/                          # Données brutes originales (CSV)
│   ├── processed/                    # Données nettoyées — data_clean.csv
│   └── train_test/                   # X_train, X_test, y_train, y_test
│
├── notebooks/
│   └── prototypage.ipynb             # EDA + premières expérimentations
│
├── src/
│   ├── preprocessing.py              # Pipeline complet de préparation
│   ├── train_model.py                # Entraînement de tous les modèles
│   ├── predict.py                    # Prédictions sur nouvelles données
│   └── utils.py                      # Fonctions utilitaires partagées
│
├── models/                           # Modèles sérialisés (.pkl)
│   ├── scaler.pkl / scaler_regression.pkl
│   ├── mediane_train.pkl
│   ├── pca.pkl / kmeans.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl / stacking.pkl
│   └── regression_xgboost_optimized.pkl
│
├── app/
│   ├── app.py                        # Application web Flask
│   ├── static/script.js
│   └── templates/index.html          # Interface utilisateur
│
├── reports/                          # Graphiques et visualisations (.png)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 3. Installation & configuration

**Prérequis** : Python 3.10+

```bash
# Cloner le dépôt
git clone https://github.com/chahdgharbi/projet_ml_retail.git
cd projet_ml_retail

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## 4. Lancement du pipeline

Les étapes sont à exécuter dans l'ordre suivant :

**Étape 0 — Exploration & prototypage**
```bash
jupyter notebook notebooks/prototypage.ipynb
```
*Point de départ obligatoire.* Contient l'analyse exploratoire (EDA), la détection des anomalies et les premières visualisations. À exécuter avant tout script.

---

**Étape 1 — Préparation des données**
```bash
python src/preprocessing.py
```
Ce script enchaîne dans l'ordre : suppression des colonnes inutiles · correction des valeurs aberrantes · parsing de `RegistrationDate` (3 formats) · transformation de `LastLoginIP` → `IsPrivateIP` · feature engineering · suppression du data leakage (seuil r > 0.85) · suppression de la multicolinéarité (seuil r > 0.8) · encodage · split stratifié 80/20 · imputation par médiane · standardisation `StandardScaler`.

*Fichiers produits :* `data/processed/data_clean.csv` · `data/train_test/*.csv` · `models/scaler.pkl` · `models/mediane_train.pkl`

---

**Étape 2 — Entraînement des modèles**
```bash
python src/train_model.py
```
Entraîne successivement : ACP (56 composantes, 95.5% variance) · K-Means k=4 · Random Forest (SMOTE + RandomizedSearchCV) · XGBoost (SMOTE + GridSearchCV) · Stacking RF+XGB · XGBoost Regressor.

*Fichiers produits :* tous les `.pkl` dans `models/` · matrices de confusion, courbes ROC, importances des features dans `reports/`

---

**Étape 3 — Génération des prédictions**
```bash
python src/predict.py
```
Charge les modèles sauvegardés et génère `reports/predictions_test.csv` sur `X_test`.

---

**Étape 4 — Application web**
```bash
python app/app.py
```
Interface disponible sur : `http://127.0.0.1:5000`

---

## 5. Résultats

### 5.1 Classification — Prédiction du churn

| Modèle | Accuracy | F1 (Churner) | Rappel | AUC-ROC | Seuil optimal |
|--------|----------|--------------|--------|---------|---------------|
| Random Forest | 94.5% | 0.92 | 0.89 | **0.985** | 0.47 |
| **XGBoost** | **97.1%** | **0.96** | 0.92 | — | 0.60 |
| Stacking (RF + XGB) | 97.0% | 0.95 | **0.93** | — | 0.50 |

### 5.2 Régression & clustering

| Tâche | Modèle | Métrique |
|-------|--------|----------|
| Prédiction `MonetaryTotal` | XGBoost Regressor | R² = 0.545 · RMSE = 7 648 £ |
| Segmentation | K-Means (k=4, ACP) | Silhouette = 0.086 |
| Réduction dimensionnelle | ACP | 95.5% variance en 56 composantes |

---

## 6. Traitement des problèmes de qualité

| Problème | Variables concernées | Solution appliquée |
|----------|---------------------|--------------------|
| Valeurs manquantes | `Age` (30%), `AvgDaysBetweenPurchases` | Imputation médiane (fit sur train uniquement) |
| Valeurs aberrantes | `SupportTicketsCount` (999, -1), `SatisfactionScore` (99, -1) | Remplacement par `NaN` puis imputation |
| Formats hétérogènes | `RegistrationDate` (formats UK / ISO / US) | `pd.to_datetime(dayfirst=True)` |
| Variable sans variance | `NewsletterSubscribed` (toujours "Yes") | Suppression |
| Variable brute | `LastLoginIP` | Transformation en booléen `IsPrivateIP` |
| Déséquilibre des classes | Churn : 33% / 67% | SMOTE `sampling_strategy=0.5` |
| Multicolinéarité | 11 paires (ex. `MonetaryMin` ↔ `MonetaryStd` : r=0.97) | Suppression au seuil r > 0.8 |
| Data leakage | `ChurnRiskCategory`, `CustomerType`, `LoyaltyLevel`, `RFMSegment`, `AccountStatus`, `Recency` | Suppression + vérification automatique |

> ⚠️ Sans correction du data leakage, le Random Forest atteignait **100% d'accuracy** — score artificiel non généralisable. Après correction : **94.5%**, résultat réaliste.

---

## 7. Segmentation K-Means (k=4)

| Cluster | Profil client | Caractéristiques |
|---------|--------------|-----------------|
| **A** — Premium | Meilleurs clients | Haute fréquence · montant élevé · faible récence |
| **B** — Réguliers | Clients stables | Fréquence et montant modérés |
| **C** — Occasionnels | Potentiel de réactivation | Fréquence faible · acheteurs ponctuels |
| **D** — Inactifs | Risque élevé de churn | Très longue récence · faible engagement |

Score de silhouette : **0.086** contre 0.038 sans ACP → amélioration de **+127%**.

---

## 8. Stack technique

| Catégorie | Librairies / Outils |
|-----------|---------------------|
| Langage | Python 3.10 |
| Manipulation des données | `pandas` · `numpy` |
| Machine Learning | `scikit-learn` · `xgboost` · `imbalanced-learn` |
| Visualisation | `matplotlib` · `seaborn` · `Chart.js` |
| Déploiement | `Flask` |
| Sérialisation | `joblib` |

---

## 9. Interface web — fonctionnalités

L'application Flask permet de saisir 4 caractéristiques client et retourne en temps réel :

| Sortie | Détail |
|--------|--------|
| Prédiction churn | Fidèle / Churner pour RF, XGBoost et Stacking avec probabilités |
| Segment K-Means | Cluster A / B / C / D avec description du profil |
| Valeur estimée | Montant total prédit par XGBoost Regressor |
| Dashboard | Accuracy · F1 · Rappel · AUC-ROC · Top 10 features interactif |

---

*Projet réalisé dans un cadre pédagogique — libre d'utilisation pour l'apprentissage.*