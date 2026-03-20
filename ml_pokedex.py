import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import MaxAbsScaler, QuantileTransformer, PowerTransformer
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, normaltest
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import seaborn as sns

def apply_scaling(df, columns, plot=False):
    scalers = {
        "Original": None,
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "MaxAbsScaler": MaxAbsScaler(),
        "QuantileTransformer": QuantileTransformer(output_distribution="normal"),
        "PowerTransformer": PowerTransformer(method="yeo-johnson")
    }
    
    results = []

    for col in columns:
        data_clean = df[col].dropna().values.reshape(-1, 1)
        
        for name, scaler in scalers.items():
            # Transforma os dados
            transformed = data_clean if scaler is None else scaler.fit_transform(data_clean)
            transformed = transformed.flatten() # Volta para 1D para o teste estatístico
            
            # Teste de Hipótese (D'Agostino's K^2)
            # H0: A amostra vem de uma distribuição normal
            # p > 0.05: Não rejeitamos H0 (é normal)
            stat, p_value = normaltest(transformed) if len(transformed) >= 8 else (0, 0)
            
            results.append({
                "coluna": col,
                "transformacao": name,
                "p_value": p_value,
                "is_normal": p_value > 0.05
            })

            if plot:
                plt.figure(figsize=(4, 3))
                plt.hist(transformed, bins=30, color='skyblue', edgecolor='black')
                plt.title(f"{col}\n{name} (p={p_value:.4f})")
                plt.show()

    # Cria o DataFrame de resultados
    df_results = pd.DataFrame(results)
    
    # Identifica a melhor transformação por coluna (maior p-value)
    best_transforms = df_results.sort_values("p_value", ascending=False).drop_duplicates("coluna")
    
    return df_results, best_transforms

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

def quality_processor(df, columns, num_strategy='median', cat_strategy='most_frequent', encoding='label', knn_neighbors=5):
    """
    Processador completo de ML-Pokedex.
    
    Estratégias Numéricas: 'mean', 'median', 'most_frequent', 'constant', 'knn'
    Estratégias Categóricas: 'most_frequent', 'constant'
    Encoders: 'label', 'onehot', 'ordinal'
    """
    df_clean = df.copy()
    report_data = []
    
    # 1. Definição dos Imputers Numéricos
    if num_strategy == 'knn':
        num_imputer = KNNImputer(n_neighbors=knn_neighbors)
    else:
        num_imputer = SimpleImputer(strategy=num_strategy)
        
    # 2. Definição do Imputer Categórico
    cat_imputer = SimpleImputer(strategy=cat_strategy)

    for col in columns:
        # --- Métricas ANTES ---
        nulls_before = df[col].isnull().sum()
        dtype = df[col].dtype
        is_numeric = np.issubdtype(dtype, np.number)
        val_before = df[col].mean() if is_numeric else df[col].mode()[0]
        
        # --- EXECUÇÃO DA IMPUTAÇÃO ---
        if is_numeric:
            # Imputação Numérica
            df_clean[[col]] = num_imputer.fit_transform(df_clean[[col]])
            applied_impute = num_strategy
        else:
            # Imputação Categórica
            df_clean[[col]] = cat_imputer.fit_transform(df_clean[[col]].astype(str))
            applied_impute = cat_strategy
            
            # --- EXECUÇÃO DO ENCODING ---
            if encoding == 'label':
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col])
            if encoding == 'ordinal':
                oe = OrdinalEncoder()
                df_clean[col] = oe.fit_transform(df_clean[[col]])
            elif encoding == 'onehot':
                # Nota: OneHot gera múltiplas colunas, aqui fazemos um dummy básico para manter no DF
                df_clean = pd.get_dummies(df_clean, columns=[col], prefix=col)
                # Para o relatório, marcamos que a coluna original foi expandida
                applied_impute += " + OneHot"

        # --- Métricas DEPOIS ---
        # (Para OneHot, checamos se a coluna base ainda existe ou se pegamos a média das novas)
        if col in df_clean.columns:
            nulls_after = df_clean[col].isnull().sum()
            val_after = df_clean[col].mean() if is_numeric or encoding == 'label' else "N/A (OneHot)"
        else:
            nulls_after = 0
            val_after = "Expanded to Dummies"

        report_data.append({
            "Coluna": col,
            "Imputação": applied_impute,
            "Encoder": encoding if not is_numeric else "N/A",
            "Nulls (Antes)": nulls_before,
            "Nulls (Depois)": nulls_after,
            "Ref Média (Antes)": val_before,
            "Ref Média (Depois)": val_after
        })

    return df_clean, pd.DataFrame(report_data)

def feature_selector(X, y, k=10, variance_threshold=0.0, correlation_threshold=0.9, plot=True, regressor=False):
    """
    Seleciona as melhores variáveis usando 3 filtros:
    1. Variância (remove constantes)
    2. Correlação (remove redundantes)
    3. Model-Based Importance (Random Forest)
    """
    X_selection = X.copy()
    initial_features = X_selection.columns.tolist()
    
    # --- 1. Filtro de Variância (Remove colunas que não mudam) ---
    selector_var = VarianceThreshold(threshold=variance_threshold)
    X_selection = pd.DataFrame(selector_var.fit_transform(X_selection), 
                               columns=X_selection.columns[selector_var.get_support()])
    
    # --- 2. Filtro de Correlação (Remove redundância) ---
    corr_matrix = X_selection.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    X_selection = X_selection.drop(columns=to_drop)

    if regressor == True:
        # --- 3. Importância via Random Forest (Captura relações não-lineares) ---
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_selection, y)
        
        importances = pd.Series(model.feature_importances_, index=X_selection.columns)
        best_features = importances.sort_values(ascending=False).head(k).index.tolist()
        
        # --- Relatório Final ---
        removed_variance = list(set(initial_features) - set(selector_var.feature_names_in_))
        report = {
            "Original": len(initial_features),
            "Removidas (Variância)": len(removed_variance),
            "Removidas (Correlação)": len(to_drop),
            "Selecionadas Final (K)": len(best_features)
        }

        if plot:
            plt.figure(figsize=(10, 6))
            importances.sort_values(ascending=True).tail(k).plot(kind='barh', color='teal')
            plt.title(f"Top {k} Features - Importância Relativa")
            plt.xlabel("Score de Importância")
            plt.show()

    elif regressor == False:
        # --- 3. Importância via Random Forest (Captura relações não-lineares) ---
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_selection, y)
        
        importances = pd.Series(model.feature_importances_, index=X_selection.columns)
        best_features = importances.sort_values(ascending=False).head(k).index.tolist()
        
        # --- Relatório Final ---
        removed_variance = list(set(initial_features) - set(selector_var.feature_names_in_))
        report = {
            "Original": len(initial_features),
            "Removidas (Variância)": len(removed_variance),
            "Removidas (Correlação)": len(to_drop),
            "Selecionadas Final (K)": len(best_features)
        }

        if plot:
            plt.figure(figsize=(10, 6))
            importances.sort_values(ascending=True).tail(k).plot(kind='barh', color='teal')
            plt.title(f"Top {k} Features - Importância Relativa")
            plt.xlabel("Score de Importância")
            plt.show()

    return X_selection[best_features], pd.DataFrame([report]), importances.sort_values(ascending=False)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

def summarize_model_performance(
    y_test, 
    y_pred, 
    y_proba=None, 
    multi_class=False, 
    threshold_used=0.5,
    class_names=None,
    n_bins=10 # NOVO: Quantidade de quebras (10 = decis, 4 = quartis, etc.)
):
    """
    Calcula métricas, gera gráficos (Matriz de Confusão, Curva ROC e Curva PR) 
    e tabela de KS/Gini por quantis.
    """
    results = {}

    if class_names is not None:
        class_names = list(class_names)
    
    # --- Cálculo de Métricas Iniciais ---
    results["Accuracy"] = accuracy_score(y_test, y_pred)
    avg_type = 'weighted' if multi_class else 'binary'
    results["Precision"] = precision_score(y_test, y_pred, average=avg_type, zero_division=0)
    results["Recall"] = recall_score(y_test, y_pred, average=avg_type, zero_division=0)
    results["F1 Score"] = f1_score(y_test, y_pred, average=avg_type, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    results["Confusion Matrix"] = cm

    # --- Setup dos Gráficos ---
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Plot da Matriz de Confusão
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], 
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    ax[0].set_title('Matriz de Confusão')
    ax[0].set_xlabel('Predito')
    ax[0].set_ylabel('Real')

    # 2. ROC, PR, Gini e KS
    if y_proba is not None:
        if multi_class:
            # Lógica Multiclasse (One-vs-Rest)
            classes = np.unique(y_test)
            y_test_bin = label_binarize(y_test, classes=classes)
            n_classes = len(classes)
            
            fpr, tpr, roc_auc = {}, {}, {}
            precisions, recalls, pr_auc = {}, {}, {}
            
            for i in range(n_classes):
                # ROC
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                ax[1].plot(fpr[i], tpr[i], label=f'Classe {classes[i]} (AUC = {roc_auc[i]:.2f})')
                
                # Precision-Recall
                precisions[i], recalls[i], _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
                pr_auc[i] = average_precision_score(y_test_bin[:, i], y_proba[:, i])
                ax[2].plot(recalls[i], precisions[i], label=f'Classe {classes[i]} (AP = {pr_auc[i]:.2f})')
            
            # Métricas Globais Multiclasse
            auc_global = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
            results["ROC AUC"] = auc_global
            results["Gini"] = (2 * auc_global) - 1 # Gini Global
            results["Average Precision"] = average_precision_score(y_test_bin, y_proba, average="weighted")
            
            ax[1].set_title(f'ROC Curve (Weighted AUC: {results["ROC AUC"]:.2f})')
            ax[2].set_title(f'PR Curve (Weighted AP: {results["Average Precision"]:.2f})')
            
        else:
            # Lógica Binária
            
            # Curva ROC e Gini Global
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            results["ROC AUC"] = roc_auc
            results["Gini"] = (2 * roc_auc) - 1
            
            ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax[1].set_title(f'Curva ROC (Gini = {results["Gini"]:.2f})')

            # Curva Precision-Recall
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
            ap_score = average_precision_score(y_test, y_proba)
            results["Average Precision"] = ap_score
            
            ax[2].plot(recalls, precisions, color='forestgreen', lw=2, label=f'PR curve (AP = {ap_score:.2f})')
            ax[2].set_title('Curva Precision-Recall')

            # Encontrar melhor threshold (F1)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
            best_index = np.argmax(f1_scores)
            results["Best Threshold (F1)"] = thresholds[best_index] if best_index < len(thresholds) else 1.0
            results["Threshold Used"] = threshold_used

            # --- Tabela de KS e Gini por Decis (Apenas Binário) ---
            df = pd.DataFrame({'target': y_test, 'proba': y_proba})
            
            # Adiciona um ruído ínfimo para evitar erro de quantis com muitas probabilidades iguais (ex: modelos de árvore)
            df['proba'] = df['proba'] + np.random.uniform(0, 1e-9, size=len(df))
            
            # Criação dos bins (decis por padrão)
            df['bin'] = pd.qcut(df['proba'], q=n_bins, labels=False, duplicates='drop')
            
            # Agrupamento e cálculos
            ks_table = df.groupby('bin', observed=False).agg(
                min_prob=('proba', 'min'),
                max_prob=('proba', 'max'),
                total=('target', 'count'),
                events=('target', 'sum')
            ).sort_values(by='min_prob', ascending=False).reset_index(drop=True)
            
            ks_table['non_events'] = ks_table['total'] - ks_table['events']
            ks_table['event_rate'] = ks_table['events'] / ks_table['total']
            
            ks_table['cum_events'] = ks_table['events'].cumsum()
            ks_table['cum_non_events'] = ks_table['non_events'].cumsum()
            
            ks_table['cum_pct_events'] = ks_table['cum_events'] / ks_table['events'].sum()
            ks_table['cum_pct_non_events'] = ks_table['cum_non_events'] / ks_table['non_events'].sum()
            
            # Cálculo do KS (Max dif. entre pct cumulativo de eventos e não eventos)
            ks_table['KS'] = np.abs(ks_table['cum_pct_events'] - ks_table['cum_pct_non_events']) * 100
            
            results["KS_Table"] = ks_table
            results["Max_KS"] = ks_table['KS'].max()

        # Formatação Eixo ROC (ax[1])
        ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax[1].set_xlim([0.0, 1.0])
        ax[1].set_ylim([0.0, 1.05])
        ax[1].set_xlabel('Taxa de Falso Positivo (FPR)')
        ax[1].set_ylabel('Taxa de Verdadeiro Positivo (TPR)')
        ax[1].legend(loc="lower right")
        
        # Formatação Eixo PR (ax[2])
        ax[2].set_xlim([0.0, 1.0])
        ax[2].set_ylim([0.0, 1.05])
        ax[2].set_xlabel('Recall (TPR)')
        ax[2].set_ylabel('Precision')
        ax[2].legend(loc="lower left")
        
    else:
        ax[1].text(0.5, 0.5, 'y_proba não fornecido\nCurva ROC indisponível', ha='center', va='center')
        ax[2].text(0.5, 0.5, 'y_proba não fornecido\nCurva PR indisponível', ha='center', va='center')
        results["ROC AUC"], results["Gini"], results["Average Precision"], results["KS_Table"], results["Max_KS"] = None, None, None, None, None

    plt.tight_layout()
    plt.show()

    return results