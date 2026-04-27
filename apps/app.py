from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# ============================================================
# Chargement des modèles
# ============================================================
rf         = joblib.load('models/random_forest.pkl')
xgb_clf    = joblib.load('models/xgboost.pkl')
stacking   = joblib.load('models/stacking.pkl')
scaler     = joblib.load('models/scaler.pkl')
pca        = joblib.load('models/pca.pkl')
kmeans     = joblib.load('models/kmeans.pkl')
reg_model  = joblib.load('models/regression_xgboost_optimized.pkl')
reg_scaler = joblib.load('models/scaler_regression.pkl')

X_train = pd.read_csv('data/train_test/X_train.csv')
X_test  = pd.read_csv('data/train_test/X_test.csv')
y_test  = pd.read_csv('data/train_test/y_test.csv').squeeze()

feature_names = X_train.columns.tolist()
mean_values   = X_train.mean().to_dict()

df_reg = pd.read_csv('data/processed/data_clean.csv')
if 'Country' in df_reg.columns:
    df_reg = df_reg.drop(columns=['Country'])
X_reg       = df_reg.drop(columns=['MonetaryTotal', 'Churn'])
reg_median  = X_reg.median().to_dict()
reg_columns = X_reg.columns.tolist()

# Labels des segments K-Means (cohérents avec predict.py)
SEGMENT_LABELS = {
    0: 'Premium',
    1: 'Réguliers',
    2: 'Occasionnels',
    3: 'Inactifs',
}

def scale_value(col, val):
    idx = feature_names.index(col)
    return (val - scaler.mean_[idx]) / scaler.scale_[idx]

# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/metrics')
def metrics():
    """Métriques du XGBoost avec seuil optimisé (0.60)"""
    try:
        y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]
        y_pred_xgb  = (y_proba_xgb >= 0.60).astype(int)
        return jsonify({
            'accuracy' : float(round(accuracy_score(y_test, y_pred_xgb), 3)),
            'f1'       : float(round(f1_score(y_test, y_pred_xgb), 3)),
            'recall'   : float(round(recall_score(y_test, y_pred_xgb), 3)),
            'roc_auc'  : float(round(roc_auc_score(y_test, y_proba_xgb), 3)),
            'n_clients': int(len(X_train) + len(X_test)),
            'churn_rate': float(round(float(y_test.mean()) * 100, 1))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feature_importance')
def feature_importance():
    """Top 10 features importantes du modèle XGBoost"""
    try:
        importances = xgb_clf.feature_importances_
        feat_imp    = dict(zip(feature_names, importances))
        sorted_imp  = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        return jsonify({
            'labels': [item[0] for item in sorted_imp],
            'values': [float(item[1]) for item in sorted_imp]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        frequency         = float(data['frequency'])
        monetary          = float(data['monetary'])
        favorite_season   = data['favorite_season']
        product_diversity = data['product_diversity']

        # Construire le vecteur client à partir des moyennes
        client = mean_values.copy()
        for col in feature_names:
            if col.startswith('FavoriteSeason_') or col.startswith('ProductDiversity_'):
                client[col] = 0
        if f'FavoriteSeason_{favorite_season}' in client:
            client[f'FavoriteSeason_{favorite_season}'] = 1
        if f'ProductDiversity_{product_diversity}' in client:
            client[f'ProductDiversity_{product_diversity}'] = 1
        if 'Frequency' in feature_names:
            client['Frequency'] = scale_value('Frequency', frequency)
        if 'MonetaryTotal' in feature_names:
            client['MonetaryTotal'] = scale_value('MonetaryTotal', monetary)

        client_df = pd.DataFrame([client], columns=feature_names)

        # Random Forest
        proba_rf      = rf.predict_proba(client_df)[0]
        prob_rf_churn = float(round(proba_rf[1] * 100, 1))
        prob_rf_fidele = float(round(proba_rf[0] * 100, 1))

        # XGBoost
        proba_xgb       = xgb_clf.predict_proba(client_df)[0]
        prob_xgb_churn  = float(round(proba_xgb[1] * 100, 1))
        prob_xgb_fidele = float(round(proba_xgb[0] * 100, 1))

        # Stacking
        meta      = np.array([[proba_rf[1], proba_xgb[1]]])
        stk_pred  = stacking.predict(meta)[0]
        stk_proba = stacking.predict_proba(meta)[0]
        prob_stk_churn  = float(round(stk_proba[1] * 100, 1))
        prob_stk_fidele = float(round(stk_proba[0] * 100, 1))

        # Segmentation K-Means (via PCA)
        try:
            client_scaled = scaler.transform(client_df)
            client_pca    = pca.transform(client_scaled)
            seg_id        = int(kmeans.predict(client_pca)[0])
            segment_label = SEGMENT_LABELS.get(seg_id, f'Segment {seg_id}')
        except Exception:
            seg_id        = -1
            segment_label = 'Non disponible'

        # Segment dérivé du risque churn (logique métier complémentaire)
        prob_final = prob_stk_churn
        if prob_final < 25:
            risk_segment = 'Premium'
        elif prob_final < 50:
            risk_segment = 'Réguliers'
        elif prob_final < 75:
            risk_segment = 'Occasionnels'
        else:
            risk_segment = 'Inactifs'

        # Régression MonetaryTotal
        try:
            reg_row = {col: reg_median.get(col, 0) for col in reg_columns}
            if 'Frequency' in reg_row:
                reg_row['Frequency'] = frequency
            for col in reg_columns:
                if col.startswith('FavoriteSeason_'):
                    reg_row[col] = 0
                if col.startswith('ProductDiversity_'):
                    reg_row[col] = 0
            if f'FavoriteSeason_{favorite_season}' in reg_row:
                reg_row[f'FavoriteSeason_{favorite_season}'] = 1
            if f'ProductDiversity_{product_diversity}' in reg_row:
                reg_row[f'ProductDiversity_{product_diversity}'] = 1
            reg_df        = pd.DataFrame([reg_row], columns=reg_columns)
            reg_scaled    = reg_scaler.transform(reg_df)
            pred_val      = float(reg_model.predict(reg_scaled)[0])
            monetary_pred = float(round(pred_val, 2)) if pred_val >= 0 else 0.0
        except Exception:
            monetary_pred = 0.0

        return jsonify({
            'rf':  {'churn': prob_rf_churn,  'fidele': prob_rf_fidele},
            'xgb': {'churn': prob_xgb_churn, 'fidele': prob_xgb_fidele},
            'stacking': {
                'churn':  prob_stk_churn,
                'fidele': prob_stk_fidele,
                'label':  'Churner' if stk_pred == 1 else 'Fidèle'
            },
            'segment':      segment_label,   # segment K-Means
            'risk_segment': risk_segment,    # segment par niveau de risque
            'seg_id':       seg_id,
            'monetary_pred': monetary_pred,
            'probability':   prob_stk_churn
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
