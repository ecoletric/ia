# FILE: energia_solar2.py

import pandas as pd
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys
import requests
import datetime
import matplotlib.pyplot as plt

# Função para preparar dados com lag
def df_to_Xy(X_scaled, y, lag):
    X, y_lag = [], []
    for i in range(len(X_scaled) - lag):
        X.append(X_scaled[i:i+lag])
        y_lag.append(y.iloc[i+lag])
    return np.array(X), np.array(y_lag)

# Função para gerar o modelo solar
features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
generation_data = pd.read_csv('Plant_2_Generation_Dataset.csv')
weather_data = pd.read_csv('Plant_2_Weather_Sensor_Data.csv')
# Mesclar os datasets
df_solar = pd.merge(generation_data.drop(columns=['PLANT_ID']), weather_data.drop(columns=['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
# Converter 'DATE_TIME' para datetime e ordenar
df_solar['DATE_TIME'] = pd.to_datetime(df_solar['DATE_TIME'])
df_solar = df_solar.sort_values('DATE_TIME')
df_solar.set_index('DATE_TIME', inplace=True)
# Selecionar features e target
X = df_solar[features]
y = df_solar['AC_POWER']
# Tratar valores faltantes
X.fillna(method='ffill', inplace=True)
y.fillna(method='ffill', inplace=True)
# Filtrar apenas onde IRRADIATION > 0 (horas diurnas)
mask = X['IRRADIATION'] > 0
X = X[mask]
y = y[mask]
# Normalizar as features
scaler_X = MinMaxScaler()
scaled_X = scaler_X.fit_transform(X)
# Normalizar o target
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
# Definir lag
lag = 24  # Ajuste conforme necessário
# Preparar dados com lag
X_lag, y_lag = df_to_Xy(pd.DataFrame(scaled_X, columns=features), pd.Series(y_scaled, index=X.index), lag)
# Dividir dados em treino, validação e teste
X_train, X_temp, y_train, y_temp = train_test_split(
    X_lag, y_lag, test_size=0.2, shuffle=False
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, shuffle=False
)
# Construir o modelo LSTM
model = Sequential()
model.add(LSTM(units=64, input_shape=(lag, len(features))))
model.add(Dense(units=1, activation='linear'))
# Compilar o modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# Treinar o modelo
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
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
plt.savefig('training_validation_loss_solar.png')  # Salvar o plot
plt.close()
plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Erro Absoluto Médio de Treinamento e Validação')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.legend()
plt.savefig('training_validation_mae_solar.png')  # Salvar o plot
plt.close()
# Avaliar o modelo
evaluation = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {evaluation[0]}")
print(f"Test MAE: {evaluation[1]}")
# Salvar o modelo e os scalers
with open('modelos/energia_solar_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('modelos/scaler_solar_X.pkl', 'wb') as scaler_file:
    pickle.dump(scaler_X, scaler_file)
with open('modelos/scaler_solar_y.pkl', 'wb') as scaler_file:
    pickle.dump(scaler_y, scaler_file)
print("Modelo e scalers salvos com sucesso.")

# Função para previsão de energia diária
# Chamar a função de previsão
