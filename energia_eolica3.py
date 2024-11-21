# FILE: energia_eolica3.py
import pandas as pd
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Função para preparar dados com lag
def df_to_Xy(X_scaled, y, lag):
    X, y_lag = [], []
    for i in range(len(X_scaled) - lag):
        X.append(X_scaled[i:i+lag])
        y_lag.append(y.iloc[i+lag])
    return np.array(X), np.array(y_lag)

# Carregar o dataset
df = pd.read_csv('Location2.csv')

# Exibir as primeiras linhas e as colunas disponíveis
print("Primeiras linhas do DataFrame:")
print(df.head())
print("\nColunas disponíveis no DataFrame:", df.columns.tolist())

# Definir a coluna alvo (atualizado para 'Power')
target_column = 'Power'  # Nome correto da coluna

# Verificar se a coluna alvo existe
if target_column not in df.columns:
    raise ValueError(f"A coluna alvo '{target_column}' não foi encontrada no DataFrame.")

# Analisar a distribuição da variável alvo 'Power'
sns.histplot(df[target_column], bins=50, kde=True)
plt.title('Distribuição da Energia Eólica (Power)')
plt.xlabel('Power')
plt.ylabel('Frequência')
plt.savefig('power_distribution.png')  # Salvar o plot
plt.close()

# Processar a coluna de tempo
df['Time'] = pd.to_datetime(df['Time'])
df['Time'] = df['Time'].dt.strftime('%d-%m')
df = df.groupby('Time').mean().reset_index()
df = df.set_index('Time')

# Separar Features e Target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Verificar se há valores faltantes
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    print("Valores faltantes encontrados. Preenchendo com a média das colunas.")
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)

# Normalizar as Features
scaler_X = MinMaxScaler()
scaled_X = scaler_X.fit_transform(X)
scaled_X_df = pd.DataFrame(scaled_X, columns=X.columns)

# Normalizar a variável alvo
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Definir o tamanho do lag
lag = 20  # Ajuste conforme necessário

# Preparar os conjuntos de dados com lag
X_lag, y_lag = df_to_Xy(scaled_X_df.values, pd.Series(y_scaled), lag)

print(f"X shape: {X_lag.shape}, y shape: {y_lag.shape}")

# Dividir os dados em treino, validação e teste
X_train, X_temp, y_train, y_temp = train_test_split(
    X_lag, y_lag, test_size=0.2, shuffle=False
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, shuffle=False
)

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

# Construir o modelo LSTM
model = Sequential()
model.add(LSTM(units=64, input_shape=(lag, X_train.shape[2])))
model.add(Dense(units=1, activation='linear'))

model.summary()

# Compilar o modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Treinar o modelo
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val)
)

# Plotar as perdas de treinamento e validação
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Perda de Treinamento e Validação')
plt.xlabel('Época')
plt.ylabel('Perda (MSE)')
plt.legend()
plt.savefig('training_validation_loss.png')  # Salvar o plot
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Erro Absoluto Médio de Treinamento e Validação')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.legend()
plt.savefig('training_validation_mae.png')  # Salvar o plot
plt.close()

# Avaliar o modelo
evaluation = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {evaluation[0]}")
print(f"Test MAE: {evaluation[1]}")

# Salvar o modelo treinado e os scalers específicos para energia eólica
with open('modelos/energia_eolica_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('modelos/scaler_eolica.pkl', 'wb') as scaler_file:
    pickle.dump(scaler_X, scaler_file)

with open('modelos/scaler_y.pkl', 'wb') as scaler_y_file:
    pickle.dump(scaler_y, scaler_y_file)

print("Model, scaler_X, and scaler_y saved successfully.")