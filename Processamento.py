#Divisão da base em treino e teste e balanceamento do conjunto de treino

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import os

# --- 1. Carregar a Base de Dados Final ---
arquivo_para_ler = 'pns_final_pre_balanceamento.csv'
try:
    df = pd.read_csv(arquivo_para_ler, sep=';', decimal=',')
    print(f"Arquivo '{arquivo_para_ler}' carregado com sucesso.")
    print(f"Shape da base (desbalanceada): {df.shape}")
except FileNotFoundError:
    print(f"🛑 Erro: Arquivo '{arquivo_para_ler}' não foi encontrado.")
    exit()

# --- 2. DEFINIR A VARIÁVEL ALVO (y) ---
TARGET_COLUMN = 'DIABETES' 

if TARGET_COLUMN not in df.columns:
    print(f"🛑 Erro: A coluna alvo '{TARGET_COLUMN}' não foi encontrada no DataFrame.")
    exit()

# --- 3. Definir Listas de Features (X) ---
colunas_numericas = [
    "IDADE", "RENDA_TOTAL", "Peso_Final", "Altura_Final_cm", "IMC"
]

colunas_ordinais = [
    "FEIJAO", "VERDURA_LEGUME", "FREQ_VERDURA_LEGUME", "CARNE_VERMELHA", 
    "FRANGO_GALINHA", "PEIXE", "SUCO_INDUSTRIALIZADO", "SUCO_NATURAL", 
    "FRUTA_SEMANA", "FREQ_FRUTA_DIA", "REFRIGERANTE_SEMANA", "DOCES_SEMANA", 
    "SUBSTITUIR_REFEICAO_DOCE_SEMANA", "CONSUMO_SAL", "NIVEL_CONSUMO_ALCOOL", 
    "NIVEL_ATIVIDADE_FISICA", "FAIXA_RENDA_SM"
]

colunas_nominais = [
    "SEXO", "PLANO_SAUDE", "GRAVIDEZ", "TIPO_SUCO_INDUSTIALIZADO", 
    "TIPO_REFRIGERANTE", "LEITE_SEMANA", "TIPO_LEITE", "COMA_DIABETICO"
]

# --- 4. Definir X e y ---
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# --- 5. Fazer o Train-Test Split (ANTES do pré-processamento) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\n--- Distribuição das classes (Antes do SMOTE) ---")
print("Treino (y_train):\n", y_train.value_counts(normalize=True))
print("\nTeste (y_test):\n", y_test.value_counts(normalize=True))

# --- 6. Criar Pipelines de Pré-processamento ---
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Pipeline para dados ordinais
# OrdinalEncoder, que pode lidar com categorias desconhecidas
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    # 'unknown_value=-1' atribui -1 para qualquer categoria
    # que ele não viu no treino, evitando o erro.
])

nominal_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


# --- 7. Aplicar Encoding (ColumnTransformer) ---
# Juntar TODOS os transformers em um só passo
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, colunas_numericas),
        ('nom', nominal_transformer, colunas_nominais),
        ('ord', ordinal_transformer, colunas_ordinais) 
    ],
    remainder='drop' # Descarta colunas que não foram listadas 
)

print("\nAplicando One-Hot, Ordinal Encoding e Scaling...")

# Ajusta (fit) no TREINO e transforma o TREINO
X_train_final = preprocessor.fit_transform(X_train)

# Apenas transforma o TESTE
X_test_final = preprocessor.transform(X_test)

print("Pré-processamento (Encoding/Scaling) concluído.")

# --- 9. Aplicar o SMOTE (SOMENTE no conjunto de TREINO) ---
print("\nAplicando SMOTE apenas nos dados de TREINO...")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_final, y_train)

print("\n--- Distribuição das classes (Depois do SMOTE) ---")
print("Treino (y_train_smote):\n", pd.Series(y_train_smote).value_counts(normalize=True))

print("\n--- SHAPES FINAIS ---")
print(f"X_train_smote (p/ Modelo): {X_train_smote.shape}")
print(f"y_train_smote (p/ Modelo): {y_train_smote.shape}")
print(f"X_test_final (p/ Avaliação): {X_test_final.shape}")
print(f"y_test (p/ Avaliação): {y_test.shape}")

print("\n✅ Processo concluído!")
print("Você está pronto para treinar seu modelo.")
print("Use (X_train_smote, y_train_smote) para TREINAR.")
print("Use (X_test_final, y_test) para AVALIAR.")

# --- 10. Salvar os conjuntos de dados ---
np.save('X_train_smote.npy', X_train_smote)
np.save('y_train_smote.npy', y_train_smote)
np.save('X_test_final.npy', X_test_final)
np.save('y_test.npy', y_test)


print("\n💾 Conjuntos de treino (balanceado) e teste (desbalanceado) salvos como arquivos .npy.")
