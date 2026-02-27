import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import MaxAbsScaler, QuantileTransformer, PowerTransformer
import matplotlib.pyplot as plt
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