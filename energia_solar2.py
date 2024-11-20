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

# Função para validar nomes
def validar_nome(nome, tipo):
    """Valida se o nome contém apenas letras."""
    if not nome.replace(" ", "").isalpha():
        print(f"O {tipo} deve conter apenas letras.")
        raise ValueError

# Função para obter condições solares
def estimate_solar_conditions(cidade, estado):
    from novo_ia import latLotCidade
    resposta = latLotCidade(nomeCidade=cidade, estado=estado)

    latitude = resposta.get('latitude')
    longitude = resposta.get('longitude')
    url = "https://api.open-meteo.com/v1/forecast"
    today = datetime.date.today()
    start_date = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")  # Incluir o dia anterior
    end_date = today.strftime("%Y-%m-%d")
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "shortwave_radiation"],
        "timezone": "America/Sao_Paulo",
        "start_date": start_date,
        "end_date": end_date
    }
    response = requests.get(url, params=params)
    data = response.json()
    df_conditions = pd.DataFrame({
        'AMBIENT_TEMPERATURE': data['hourly']['temperature_2m'],
        'IRRADIATION': data['hourly']['shortwave_radiation'],
    }, index=pd.to_datetime(data['hourly']['time']))
    # Estimando MODULE_TEMPERATURE como AMBIENT_TEMPERATURE + offset
    df_conditions['MODULE_TEMPERATURE'] = df_conditions['AMBIENT_TEMPERATURE'] + np.random.uniform(3, 6, size=len(df_conditions))
    # Converter IRRADIATION de W/m² para kW/m²
    df_conditions['IRRADIATION'] = df_conditions['IRRADIATION'] / 1000.0
    return df_conditions

# Função para gerar o modelo solar
def gerar_modelo_solar():    
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
    with open('energia_solar_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('scaler_solar_X.pkl', 'wb') as scaler_file:
        pickle.dump(scaler_X, scaler_file)

    with open('scaler_solar_y.pkl', 'wb') as scaler_file:
        pickle.dump(scaler_y, scaler_file)

    print("Modelo e scalers salvos com sucesso.")

# Função para previsão de energia diária
def predict_solar_daily_energy(cidade, estado, usina_capacidade=1000, lag=24):
    features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']

    # Carregar o modelo e os scalers
    with open('energia_solar_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler_solar_X.pkl', 'rb') as scaler_file:
        scaler_X = pickle.load(scaler_file)
    with open('scaler_solar_y.pkl', 'rb') as scaler_file:
        scaler_y = pickle.load(scaler_file)

    # Obter as condições solares
    df_conditions = estimate_solar_conditions(cidade, estado)

    # Filtrar apenas horários diurnos onde a irradiação é maior que zero
    df_conditions = df_conditions[df_conditions['IRRADIATION'] > 0]

    # Verificar se temos dados suficientes
    if len(df_conditions) < lag:
        lag = len(df_conditions) - 1
        if lag < 1:
            print("Dados insuficientes para previsão.")
            return

    # Preprocessar as condições
    df_conditions = df_conditions[features]
    scaled_conditions = scaler_X.transform(df_conditions)

    # Preparar os dados para previsão
    predictions = []
    time_intervals = df_conditions.index[lag:]
    for i in range(lag, len(scaled_conditions)):
        X_input = scaled_conditions[i-lag:i]
        X_input = X_input.reshape((1, lag, len(features)))
        predicted_scaled = model.predict(X_input)
        predicted = scaler_y.inverse_transform(predicted_scaled)[0][0]
        predictions.append(max(predicted, 0))  # Garantir que não haja valores negativos

    # Criar DataFrame com as previsões
    df_predictions = pd.DataFrame({
        'AC_POWER_PREDICTED': predictions
    }, index=time_intervals)

    # Converter índice para o timezone local
    df_predictions = df_predictions.tz_localize('UTC').tz_convert("America/Sao_Paulo")

    # Filtrar as previsões para o dia atual
    df_predictions_today = df_predictions[df_predictions.index.date == datetime.date.today()]

    if df_predictions_today.empty:
        print("Sem previsões para o dia atual.")
        return

    # Calcular a energia total gerada no dia
    energia_total_kwh = 0
    total_delta_horas = 0  # Para calcular horas de geração
    for i in range(1, len(df_predictions_today)):
        delta_horas = (df_predictions_today.index[i] - df_predictions_today.index[i-1]).total_seconds() / 3600
        energia_intervalo = df_predictions_today['AC_POWER_PREDICTED'].iloc[i] * delta_horas
        energia_total_kwh += energia_intervalo
        total_delta_horas += delta_horas

    # Exibir o resultado
    print(f"Energia total prevista para o dia em {cidade}/{estado}: {energia_total_kwh:.2f} kWh")

    # Opcional: Ajustar a previsão conforme a capacidade da sua usina
    capacidade_plant2_kw = 1000  # 1 MW = 1000 kW
    capacidade_sua_usina_kw = usina_capacidade
    fator_escala = capacidade_sua_usina_kw / capacidade_plant2_kw
    energia_ajustada = energia_total_kwh * fator_escala

    # Calcular a energia máxima possível baseada na capacidade da usina e horas de geração
    max_energia_possivel = capacidade_sua_usina_kw * total_delta_horas

    # Ajustar para não exceder a energia máxima possível
    energia_ajustada = min(energia_ajustada, max_energia_possivel)

    print(f"Energia total ajustada para a sua usina de {capacidade_sua_usina_kw} kW: {energia_ajustada:.2f} kWh")

    # Opcional: Plotar as previsões
    plt.figure(figsize=(10, 6))
    plt.plot(df_predictions_today.index, df_predictions_today['AC_POWER_PREDICTED'], marker='o')
    plt.title(f'Previsão de AC_POWER ao longo do dia em {cidade}/{estado}')
    plt.xlabel('Hora')
    plt.ylabel('AC_POWER (kW)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ac_power_predictions.png')  # Salvar o plot
    plt.close()

    return energia_ajustada

# Chamar a função de previsão
if __name__ == '__main__':
    while True:
        try:
            opcao = int(input('Menu: \n1 - Gerar modelo solar \n2 - Prever energia solar diária \n3 - Sair \nSelecione uma opção: '))
            if opcao not in [1, 2, 3]:
                raise ValueError
        except ValueError:
            print("Insira um valor válido (1, 2 ou 3).")
        else:
            if opcao == 1:
                gerar_modelo_solar()
            elif opcao == 2:
                try:
                    cidade = input("Escreva o nome da cidade:\n")
                    validar_nome(cidade, "cidade")
                    estado = input("Escreva o nome do estado:\n")
                    validar_nome(estado, "estado")
                    predict_solar_daily_energy(cidade, estado)
                except ValueError:
                    print("Insira um valor válido para cidade e estado.")
            elif opcao == 3:
                sys.exit()