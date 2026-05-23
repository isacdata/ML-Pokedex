import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

def objective(
    trial,
    X_train,
    y_train,
    objective_metric='f1_score',
    random_state=42
    ):
    param = {
        # --- Estrutura e Aprendizado ---
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        
        # --- Controle da Árvore ---
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        # num_leaves deve ser menor que 2^(max_depth). Optuna vai buscar valores coerentes:
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        
        # --- Amostragem (Evita Overfitting) ---
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        
        # --- Regularização (Penaliza modelos complexos) ---
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True), # L1
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), # L2
        
        # ======================================================
        # 2. PARÂMETROS FIXOS (O Optuna NÃO vai testar)
        # ======================================================
        'objective': 'multiclass', # ou 'multiclass' dependendo do seu problema
        'random_state': random_state,    # Garante reprodutibilidade
        'n_jobs': -1,          # Usa todos os núcleos do seu processador
        'importance_type': 'split', 
        'class_weight': 'balanced', # Útil se as suas classes estiverem desbalanceadas
        'verbose': -1 # Para retirar warnings
    }

    # Utilização do time series split

    tscv = TimeSeriesSplit(n_splits=8)
    scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = LGBMClassifier(**param)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)

        score_accuracy = accuracy_score(y_val, y_pred)
        score_f1 = f1_score(y_val, y_pred, average='weighted')
        score_recall = recall_score(y_val, y_pred, average='weighted')
        score_precision = precision_score(y_val, y_pred, average='weighted')

        if objective_metric == 'f1_score':
            scores.append(score_f1)
        elif objective_metric == 'accuracy':
            scores.append(score_accuracy)
        elif objective_metric == 'recall':
            scores.append(score_recall)
        elif objective_metric == 'precision':
            scores.append(score_precision)
        elif objective_metric == 'roc_auc':
            scores.append(roc_auc_score(y_val, y_pred))

    return sum(scores) / len(scores)