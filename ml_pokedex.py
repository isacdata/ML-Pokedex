import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import MaxAbsScaler, QuantileTransformer, PowerTransformer
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, normaltest

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