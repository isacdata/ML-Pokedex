# 📊 Previsão de Renda e Análise Temporal (Income Prediction Model)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-orange?style=for-the-badge&logo=lightgbm)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter_Tuning-blueviolet?style=for-the-badge&logo=optuna)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-150458?style=for-the-badge&logo=pandas&logoColor=white)

Um projeto completo de Machine Learning focado na construção de um modelo de classificação multiclasse (focado em *Income* - faixas de renda) ao longo do tempo. O diferencial deste projeto é o tratamento rigoroso contra vazamento de dados (*data leakage*) e a consideração do tempo na validação do modelo usando separação temporal.

## 🎯 Objetivo

O objetivo principal deste repositório é treinar e otimizar um **LGBMClassifier** robusto, evitando os problemas comuns de *Overfitting* ao longo do tempo. Através de *TimeSeriesSplit* e tunagem de hiperparâmetros automatizada com **Optuna**, garantimos que o modelo não "preveja o passado usando o futuro" e lide adequadamente com as quebras temporais.

---

## 📁 Estrutura do Repositório

O pipeline de desenvolvimento foi dividido em três notebooks principais para manter a organização e facilitar o processo analítico:

* **`0.criando_datasets.ipynb`**: Foco em **Data Prep**. Carga de dados, limpeza, tratamento de valores nulos e estruturação inicial. Transforma os dados puros (texto, numéricos) em formatos aceitos pelo modelo, garantindo que a base esteja ordenada cronologicamente para evitar data leakage no futuro.
* **`1.modeling.ipynb`**: O coração do projeto. Utiliza o script associado para definir o espaço de busca, integrando `TimeSeriesSplit` e `LightGBM`. Realiza a otimização matemática rigorosa do modelo.
* **`2.evaluating_results.ipynb`**: Avaliação profunda do classificador. Geração de relatórios de métricas (F1-Score, Acurácia, AUC), plotagem de gráficos temporais para monitorar o F1 ao longo dos anos (e verificar *Concept Drift*), além de extrair as importâncias das *features* (análise de Pareto).

### Arquivos de Suporte e Utilitários
* **`optuna_utils.py`**: Um script modularizado contendo a função `objective` para o Optuna, que parametriza e encapsula o *TimeSeriesSplit*, espaço de busca de hiperparâmetros (como `learning_rate`, `n_estimators`, regularizações L1/L2) e a lógica de fit/predict.
* **`resultados_optuna.csv`**: Histórico detalhado de todos os *trials* rodados (parâmetros vs valores alcançados), ideal para entender o que deu certo e o que deu errado durante a otimização.
* **`importancias_modelo.csv`**: Export das pontuações (*Feature Importances*), mostrando a relevância e o peso de cada variável no modelo treinado final.
* **Imagens (`evolucao_*.png`)**: Visões gráficas do desempenho do modelo em produção/teste. Mostram como as métricas se sustentam em ambientes inéditos de diferentes anos.

---

## 🚀 Como Executar

### Pré-requisitos

Recomenda-se criar um ambiente virtual (venv) para evitar conflitos de dependências.
```bash
python -m venv .env
source .env/bin/activate  # Para Linux/Mac
# ou
.env\Scripts\activate     # Para Windows

Instale as bibliotecas principais do projeto:
pip install pandas numpy scikit-learn lightgbm optuna matplotlib seaborn jupyter

### Ordem de Execução

1. Abra o **Jupyter Notebook** rodando `jupyter notebook` no terminal.
2. Inicie pelo **`0.criando_datasets.ipynb`** para processar e gerar os *DataFrames* base (treino e validação).
3. Vá para o **`1.modeling.ipynb`**. O Optuna orquestrará o LightGBM, testando dezenas de hiperparâmetros. Ao final deste passo, o arquivo `melhor_modelo_lgbm.pkl` guardará a inteligência gerada.
4. Finalize com o **`2.evaluating_results.ipynb`**. Aqui você fará as predições no conjunto `dataset_full` (ou conjunto de teste isolado), validando a evolução do desempenho temporal e identificando as variáveis-chave (como `population`, `data_products_score`, etc.) que mais impactam no resultado.

---

## 📈 Destaques Analíticos e Técnicos

* **Proteção Temporal (`TimeSeriesSplit`)**: Em vez de validações cruzadas K-Fold puras, a separação cronológica das janelas simula perfeitamente o ambiente de produção. O modelo foi condicionado a aprender com "Janeiro" para prever "Fevereiro", impedindo-o de memorizar variáveis tardias.
* **Lidando com Desbalanceamento**: Ajuste dos parâmetros com o suporte nativo do LightGBM (`class_weight='balanced'`) e avaliações pautadas pelo *F1-Score Ponderado (Weighted)* e *ROC AUC*, protegendo-se da ilusão de uma acurácia mentirosa em classes majoritárias.
* **Engenharia Categórica Dinâmica**: Conversão de dados de texto para o tipo *category* nativo do LightGBM (`.astype('category')`), otimizando o treinamento sem a necessidade imediata do peso extra de algoritmos tipo *OneHotEncoder*.
* **Análise de Pareto para Features**: Observando o `importancias_modelo.csv`, provou-se matematicamente quais atributos explicam quase 80% da variância nos dados, permitindo futuras limpezas que reduzam drasticamente o peso de inferência do projeto mantendo alta assertividade.

---
**Status do Projeto**: Finalizado e validado.