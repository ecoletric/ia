# FILE: novo_ia.py
from flask import Flask, jsonify, request
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import pickle
import requests
import oracledb
import numpy as np
import unicodedata
import logging
import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Configurar o logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Carregar os modelos treinados e os scalers
try:
    with open('modelos/energia_eolica_model.pkl', 'rb') as file:
        energia_eolica_model = pickle.load(file)
    logger.info("Modelo treinado carregado com sucesso.")
except FileNotFoundError:
    logger.error("Arquivo 'energia_eolica_model.pkl' não encontrado.")
    energia_eolica_model = None
except Exception as e:
    logger.error(f"Erro ao carregar 'energia_eolica_model.pkl': {e}")
    energia_eolica_model = None
try:
    with open('modelos/scaler_eolica.pkl', 'rb') as file:
        scaler = pickle.load(file)
    logger.info("Scaler carregado com sucesso.")
except FileNotFoundError:
    logger.error("Arquivo 'scaler_eolica.pkl' não encontrado.")
    scaler = None
except Exception as e:
    logger.error(f"Erro ao carregar 'scaler_eolica.pkl': {e}")
    scaler = None
try:
    with open('modelos/scaler_y.pkl', 'rb') as file:
        scaler_y = pickle.load(file)
    logger.info("Scaler para y carregado com sucesso.")
except FileNotFoundError:
    logger.error("Arquivo 'scaler_y.pkl' não encontrado.")
    scaler_y = None
except Exception as e:
    logger.error(f"Erro ao carregar 'scaler_y.pkl': {e}")
    scaler_y = None
try:
    with open('modelos/modelo_solar.pkl', 'rb') as file:
        solar_prediction_model = pickle.load(file)
except FileNotFoundError:
    logger.error("Arquivo 'modelo_solar.pkl' não encontrado.")    
# Verificar se todos os componentes foram carregados corretamente
if not energia_eolica_model or not scaler or not scaler_y:
    logger.error("Modelo ou scalers não carregados corretamente.")
    # Dependendo do caso, você pode querer encerrar a aplicação ou lidar com isso de outra forma

# Definir os nomes das features utilizadas no treinamento do modelo eólico
feature_names = [
    'temperature_2m',          # Adicionando a temperatura
    'relativehumidity_2m',
    'dewpoint_2m',
    'windspeed_10m',
    'windspeed_100m',
    'winddirection_10m',
    'winddirection_100m',
    'windgusts_10m'
]

# Funções auxiliares
def kmh_para_ms(velocidade_kmh):
    """Converte a velocidade de km/h para m/s."""
    return velocidade_kmh / 3.6

def estimar_velocidade_rajada_vento(x_ms, h1=10, h2=100, alpha=0.14):
    """Estima a velocidade do vento/rajada a uma altura diferente usando a Lei do Poder."""
    return x_ms * (h2 / h1) ** alpha

def remover_acentos(texto):
    """Remove acentos de um texto."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def uf_estado(uf):
    """Retorna o nome completo do estado a partir da sigla."""
    ufs_para_estados = { 
        "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá", "BA": "Bahia", 
        "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", 
        "MA": "Maranhão", "MG": "Minas Gerais", "MS": "Mato Grosso do Sul", "MT": "Mato Grosso", 
        "PA": "Pará", "PB": "Paraíba", "PE": "Pernambuco", "PI": "Piauí", "PR": "Paraná", 
        "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte", "RO": "Rondônia", 
        "RR": "Roraima", "RS": "Rio Grande do Sul", "SC": "Santa Catarina", 
        "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins"}
    return ufs_para_estados.get(uf)

def latLotCidade(nomeCidade, estado):
    """Obtém a latitude e longitude de uma cidade utilizando a API Open-Meteo."""
    def remover_acentos_inner(texto):
        return ''.join(
            c for c in unicodedata.normalize('NFD', texto)
            if unicodedata.category(c) != 'Mn'
        ).lower()

    list_cidades = str(nomeCidade).split(' ')
    novoNomeCidade = remover_acentos_inner('+'.join(list_cidades))
    estado_normalizado = remover_acentos_inner(estado)

    url = f"https://geocoding-api.open-meteo.com/v1/search?name={novoNomeCidade}&count=10&language=pt&format=json"
    response = requests.get(url)

    if response.status_code != 200:
        logger.error("Falha ao fazer o request para a API de geolocalização.")
        return {"error": "Falha ao fazer o request para a API de geolocalização"}, response.status_code

    data = response.json()
    results = data.get('results', [])

    for result in results:
        if result.get('country') == 'Brasil':
            admin_fields = [
                remover_acentos_inner(result.get('admin1', '').lower()),
                remover_acentos_inner(result.get('admin2', '').lower()),
                remover_acentos_inner(result.get('admin3', '').lower()),
                remover_acentos_inner(result.get('admin4', '').lower())
            ]

            if estado_normalizado in admin_fields:
                latitude = result.get('latitude')
                longitude = result.get('longitude')
                return {'latitude': latitude, 'longitude': longitude}

    logger.error("Local não encontrado com os parâmetros fornecidos.")
    return {"error": "Local não encontrado com os parâmetros fornecidos"}, 404

def climaDia(latitude, longitude):
    """Obtém dados climáticos diários da API Open-Meteo."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}&current_weather=true&"
        f"daily=temperature_2m_max,temperature_2m_min,wind_speed_10m_max,"
        f"wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum&"
        f"timezone=America/Sao_Paulo&forecast_days=1"
    )
    response = requests.get(url)
    if response.status_code != 200:
        logger.error("Não foi possível puxar informações sobre clima da cidade.")
        return {"error" : "Não foi possível puxar informações sobre clima da cidade." }
    data = response.json()
    return data

def arredondar(numero):
    """Arredonda um número para o inteiro mais próximo."""
    try:
        return int(Decimal(float(numero)).to_integral_value(rounding=ROUND_HALF_UP))
    except (ValueError, ArithmeticError) as e:
        logger.error(f"Erro ao arredondar o número: {numero} - {e}")
        return None

# Função para obter conexão com o banco de dados usando oracledb
def get_connection():
    try:
        connection = oracledb.connect(
            user='rm556448',
            password='fiap24',
            dsn='oracle.fiap.com.br/orcl'
        )
        return connection
    except oracledb.Error as e:
        logger.error(f"Erro ao conectar ao banco de dados: {e}")
        return None

# Função para obter endereço a partir do id_sitio
def get_endereco(id_sitio):
    try:
        with get_connection() as con:
            with con.cursor() as cur:
                # Consulta para obter id_endereco
                cur.execute(
                    "SELECT id_endereco FROM t_gl_sitio WHERE id_sitio = :id_sitio", 
                    {"id_sitio": id_sitio}
                )
                result = cur.fetchone()
                if not result:
                    logger.error(f"id_sitio {id_sitio} não encontrado.")
                    return {"error": "id_sitio não encontrado"}, 404
                id_endereco = result[0]
        with get_connection() as con:
            with con.cursor() as cur:
                # Consulta para obter cidade e uf
                cur.execute(
                    "SELECT cidade, uf FROM t_gl_endereco WHERE id_endereco = :id_endereco", 
                    {"id_endereco": id_endereco}
                )
                result = cur.fetchone()
                if not result:
                    logger.error(f"id_endereco {id_endereco} não encontrado.")
                    return {"error": "id_endereco não encontrado"}, 404
                cidade, uf = result

        return {"cidade": cidade, "uf": uf}
    except oracledb.Error as e:
        logger.error(f"Erro ao consultar o banco de dados: {e}")
        return {"error": "Erro interno no banco de dados"}, 500
   





# Função para estimar condições de vento
def estimate_wind_conditions(cidade: str, estado: str) -> dict:
    resposta = latLotCidade(nomeCidade=cidade, estado=estado)
    if isinstance(resposta, tuple):
        return {"error": resposta[0]}, resposta[1]
    if "error" in resposta:
        return {"error": resposta["error"]}, resposta.get("status_code", 400)
    latitude = resposta.get('latitude')
    longitude = resposta.get('longitude')
    data = climaDia(latitude=latitude, longitude=longitude)
    
    if "error" in data:
        return {"error": data["error"]}, 400

    daily = data.get('daily', {})
    current = data.get('current_weather', {})

    try:
        temperatura_maxima = daily.get('temperature_2m_max')[0]
        temperatura_minima = daily.get('temperature_2m_min')[0]
        temperatura_media = (temperatura_maxima + temperatura_minima) / 2

        wind_direction_10m_dominant = daily.get('wind_direction_10m_dominant')[0]
        wind_gusts_10m_max = kmh_para_ms(daily.get('wind_gusts_10m_max')[0])
        wind_speed_10m_max = kmh_para_ms(daily.get('wind_speed_10m_max')[0])
        wind_direction_100m = wind_direction_10m_dominant
        wind_gusts_100m_max = estimar_velocidade_rajada_vento(wind_gusts_10m_max)
        wind_speed_100m_max = estimar_velocidade_rajada_vento(wind_speed_10m_max)

        umidade = current.get('relative_humidity_2m', 50)  # Valor padrão se ausente

    except (IndexError, TypeError, KeyError) as e:
        logger.error(f"Erro ao processar dados climáticos: {e}")
        return {"error": f"Erro ao processar dados climáticos: {str(e)}"}, 500
    
    return {
        'temperature_2m': temperatura_media,  # Adicionando a temperatura média
        'relativehumidity_2m': umidade,
        'dewpoint_2m': 2.4,  # Ajuste conforme necessário ou torne dinâmico
        'windspeed_10m': wind_speed_10m_max,
        'windspeed_100m': wind_speed_100m_max,
        'winddirection_10m': wind_direction_10m_dominant,
        'winddirection_100m': wind_direction_100m,
        'windgusts_10m': wind_gusts_10m_max
    }

def somar_capacidade_usina(id_sitio : int):
    sql = "select SUM(potencia) from t_gl_aparelho_gerador where id_sitio = :id_sitio"
    with get_connection() as con:
        with con.cursor() as cur:
            cur.execute(
                        "select SUM(potencia) from t_gl_aparelho_gerador where id_sitio = :id_sitio", 
                        {"id_sitio": id_sitio}
                    )
            result = cur.fetchone()
            if not result:
                logger.error(f"capacidade da usina no sítio {id_sitio} não encontrada.")
                return {"error": "id_endereco não encontrado"}, 404
    return result[0]

def estimate_solar_conditions(cidade, estado):
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

def predict_solar_daily_energy(cidade, estado, usina_capacidade=1000, lag=24):
    features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']

    # Carregar o modelo e os scalers
    with open('modelos/energia_solar_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('modelos/scaler_solar_X.pkl', 'rb') as scaler_file:
        scaler_X = pickle.load(scaler_file)
    with open('modelos/scaler_solar_y.pkl', 'rb') as scaler_file:
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


    return energia_ajustada

# Rotas para previsão de energia solar e eólica
@app.route('/ai/predict-solar/', methods=['POST'])
def prever_solar():
    data = request.get_json()
    cidade = data.get('cidade')
    estado = data.get('estado')
    usina_capacidade = data.get('usina_capacidade')
    resultado = predict_solar_daily_energy(cidade=cidade, estado=estado, usina_capacidade=usina_capacidade)
    return jsonify({'energia_diaria_estimada': resultado})

@app.route('/ai/predict-eolic/', methods=['POST'])
def prever_eolica():
    data = request.get_json()
    cidade = data.get('cidade')
    estado = data.get('estado')
    usina_capacidade = data.get('usina_capacidade')
    
    resultado = estimate_wind_conditions(cidade, estado)
    
    if "error" in resultado:
        return jsonify(resultado), resultado.get("status_code", 400)
    
    # Carregar e processar o dataset Location2.csv
    try:
        df = pd.read_csv('Location2.csv')
        logger.info("Arquivo 'Location2.csv' carregado com sucesso.")
    except FileNotFoundError:
        logger.error("Arquivo 'Location2.csv' não encontrado.")
        return jsonify({"error": "Arquivo Location2.csv não encontrado."}), 500
    except Exception as e:
        logger.error(f"Erro ao ler 'Location2.csv': {e}")
        return jsonify({"error": f"Erro ao ler Location2.csv: {e}"}), 500

    try:
        df['Time'] = pd.to_datetime(df['Time'])
        df['Time'] = df['Time'].dt.strftime('%d-%m')
        df = df.groupby('Time').mean().reset_index()
        df = df.set_index('Time')
        logger.info("Dados do 'Location2.csv' processados com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao processar 'Location2.csv': {e}")
        return jsonify({"error": f"Erro ao processar Location2.csv: {e}"}), 500
    
    # Selecionar as últimas 20 entradas para criar o input
    lag = 20
    if len(df) < lag:
        logger.error("Dados insuficientes no dataset 'Location2.csv'.")
        return jsonify({"error": "Dados insuficientes no dataset Location2.csv."}), 400
    input_data = df.tail(lag).copy()

    # Adicionar os dados climáticos como novas features
    input_data['temperature_2m'] = resultado['temperature_2m']                     # Adicionando a temperatura
    input_data['relativehumidity_2m'] = resultado['relativehumidity_2m']
    input_data['dewpoint_2m'] = resultado['dewpoint_2m']
    input_data['windspeed_10m'] = resultado['windspeed_10m']
    input_data['windspeed_100m'] = resultado['windspeed_100m']
    input_data['winddirection_10m'] = resultado['winddirection_10m']
    input_data['winddirection_100m'] = resultado['winddirection_100m']
    input_data['windgusts_10m'] = resultado['windgusts_10m']
    # input_data['windgusts_100m'] = resultado['windgusts_100m']  # Descomente se necessário

    # Verificar se as features correspondem às usadas no treinamento
    missing_features = [feat for feat in feature_names if feat not in input_data.columns]
    if missing_features:
        logger.error(f"Faltando features: {missing_features}")
        return jsonify({"error": f"Faltando features: {missing_features}"}), 400
    
    # Selecionar apenas as features necessárias na ordem correta
    input_features = input_data[feature_names]
    logger.debug(f"Features de Entrada:\n{input_features.head()}")

    # Verificar quais features têm NaN
    na_features = input_features.columns[input_features.isnull().any()].tolist()
    if na_features:
        logger.error(f"Valores NaN encontrados nas features: {na_features}")
        return jsonify({"error": f"Valores NaN encontrados nas features: {na_features}"}), 400
    else:
        logger.info("Nenhum valor NaN encontrado nas features de entrada.")

    # Normalizar os dados
    try:
        scaled_input = scaler.transform(input_features)
        logger.info("Dados normalizados com sucesso.")
    except ValueError as ve:
        logger.error(f"Erro ao normalizar os dados: {ve}")
        return jsonify({"error": str(ve)}), 400
    
    logger.debug(f"Dados Normalizados:\n{scaled_input}")

    # Reshape para o modelo LSTM
    scaled_input = scaled_input.reshape((1, scaled_input.shape[0], scaled_input.shape[1]))
    
    # Fazer a previsão
    try:
        predicted_power_scaled = energia_eolica_model.predict(scaled_input)
        logger.info("Previsão realizada com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao fazer a previsão: {e}")
        return jsonify({"error": "Erro ao fazer a previsão."}), 500
    
    # Verificar se a previsão resultou em NaN
    if np.isnan(predicted_power_scaled[0][0]):
        logger.error("O modelo retornou NaN para a previsão.")
        return jsonify({"error": "A previsão resultou em um valor inválido (NaN)."}), 500
    
    # Desescalonar a previsão
    try:
        predicted_power_original = scaler_y.inverse_transform(predicted_power_scaled)
        logger.info(f"Previsão desescalonada: {predicted_power_original[0][0]}")
    except Exception as e:
        logger.error(f"Erro ao desescalonar a previsão: {e}")
        return jsonify({"error": "Erro ao desescalonar a previsão."}), 500

    # Arredondar o valor
    predicted_power_rounded = float(predicted_power_original[0][0])
    if predicted_power_rounded is None:
        logger.error("Erro ao arredondar a previsão.")
        return jsonify({"error": "Erro ao arredondar a previsão."}), 500
    
    logger.info(f"Previsão concluída: {predicted_power_rounded}")
    
    return jsonify({'energia_diaria_estimada': predicted_power_rounded * usina_capacidade})


# As rotas abaixo são utilizadas inclusive no Front End, porém pode-se usar as funções acima com o objetivo de teste
@app.route('/predict-solar/<int:id_sitio>', methods=['GET'])
def prever_solar_sitio(id_sitio: int):
    resultado = get_endereco(id_sitio=id_sitio)
    cidade = resultado.get('cidade')
    uf = resultado.get('uf')
    print(cidade)
    print(uf)
    estado = uf_estado(uf)
    print(estado)
    capacidade_sitio = somar_capacidade_usina(id_sitio)

    resultado = predict_solar_daily_energy(cidade=cidade, estado=estado, usina_capacidade=capacidade_sitio)
   
    # Retornar a previsão como JSON
    return jsonify({'energia_diaria_estimada': resultado})


@app.route('/predict-eolic/<int:id_sitio>', methods=['GET'])
def prever_eolica_sitio(id_sitio: int):
    if not energia_eolica_model or not scaler or not scaler_y:
        logger.error("Modelo ou scalers não carregados corretamente.")
        return jsonify({"error": "Servidor mal configurado. Tente novamente mais tarde."}), 500

    endereco = get_endereco(id_sitio)
    if "error" in endereco:
        return jsonify(endereco), endereco.get("status_code", 500)
    
    cidade = endereco['cidade']
    uf = endereco['uf']
    estado = uf_estado(uf)
    
    if not estado:
        logger.error(f"Estado inválido: {uf}")
        return jsonify({"error": "Estado inválido"}), 400
    
    resultado = estimate_wind_conditions(cidade, estado)
    
    if "error" in resultado:
        return jsonify(resultado), resultado.get("status_code", 400)
    
    # Carregar e processar o dataset Location2.csv
    try:
        df = pd.read_csv('Location2.csv')
        logger.info("Arquivo 'Location2.csv' carregado com sucesso.")
    except FileNotFoundError:
        logger.error("Arquivo 'Location2.csv' não encontrado.")
        return jsonify({"error": "Arquivo Location2.csv não encontrado."}), 500
    except Exception as e:
        logger.error(f"Erro ao ler 'Location2.csv': {e}")
        return jsonify({"error": f"Erro ao ler Location2.csv: {e}"}), 500

    try:
        df['Time'] = pd.to_datetime(df['Time'])
        df['Time'] = df['Time'].dt.strftime('%d-%m')
        df = df.groupby('Time').mean().reset_index()
        df = df.set_index('Time')
        logger.info("Dados do 'Location2.csv' processados com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao processar 'Location2.csv': {e}")
        return jsonify({"error": f"Erro ao processar Location2.csv: {e}"}), 500
    
    # Selecionar as últimas 20 entradas para criar o input
    lag = 20
    if len(df) < lag:
        logger.error("Dados insuficientes no dataset 'Location2.csv'.")
        return jsonify({"error": "Dados insuficientes no dataset Location2.csv."}), 400
    input_data = df.tail(lag).copy()

    # Adicionar os dados climáticos como novas features
    input_data['temperature_2m'] = resultado['temperature_2m']                     # Adicionando a temperatura
    input_data['relativehumidity_2m'] = resultado['relativehumidity_2m']
    input_data['dewpoint_2m'] = resultado['dewpoint_2m']
    input_data['windspeed_10m'] = resultado['windspeed_10m']
    input_data['windspeed_100m'] = resultado['windspeed_100m']
    input_data['winddirection_10m'] = resultado['winddirection_10m']
    input_data['winddirection_100m'] = resultado['winddirection_100m']
    input_data['windgusts_10m'] = resultado['windgusts_10m']
    # input_data['windgusts_100m'] = resultado['windgusts_100m']  # Descomente se necessário

    # Verificar se as features correspondem às usadas no treinamento
    missing_features = [feat for feat in feature_names if feat not in input_data.columns]
    if missing_features:
        logger.error(f"Faltando features: {missing_features}")
        return jsonify({"error": f"Faltando features: {missing_features}"}), 400
    
    # Selecionar apenas as features necessárias na ordem correta
    input_features = input_data[feature_names]
    logger.debug(f"Features de Entrada:\n{input_features.head()}")

    # Verificar quais features têm NaN
    na_features = input_features.columns[input_features.isnull().any()].tolist()
    if na_features:
        logger.error(f"Valores NaN encontrados nas features: {na_features}")
        return jsonify({"error": f"Valores NaN encontrados nas features: {na_features}"}), 400
    else:
        logger.info("Nenhum valor NaN encontrado nas features de entrada.")

    # Normalizar os dados
    try:
        scaled_input = scaler.transform(input_features)
        logger.info("Dados normalizados com sucesso.")
    except ValueError as ve:
        logger.error(f"Erro ao normalizar os dados: {ve}")
        return jsonify({"error": str(ve)}), 400
    
    logger.debug(f"Dados Normalizados:\n{scaled_input}")

    # Reshape para o modelo LSTM
    scaled_input = scaled_input.reshape((1, scaled_input.shape[0], scaled_input.shape[1]))
    
    # Fazer a previsão
    try:
        predicted_power_scaled = energia_eolica_model.predict(scaled_input)
        logger.info("Previsão realizada com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao fazer a previsão: {e}")
        return jsonify({"error": "Erro ao fazer a previsão."}), 500
    
    # Verificar se a previsão resultou em NaN
    if np.isnan(predicted_power_scaled[0][0]):
        logger.error("O modelo retornou NaN para a previsão.")
        return jsonify({"error": "A previsão resultou em um valor inválido (NaN)."}), 500
    
    # Desescalonar a previsão
    try:
        predicted_power_original = scaler_y.inverse_transform(predicted_power_scaled)
        logger.info(f"Previsão desescalonada: {predicted_power_original[0][0]}")
    except Exception as e:
        logger.error(f"Erro ao desescalonar a previsão: {e}")
        return jsonify({"error": "Erro ao desescalonar a previsão."}), 500

    # Arredondar o valor
    potencia = somar_capacidade_usina(id_sitio)
    predicted_power_rounded = float(predicted_power_original[0][0])
    if predicted_power_rounded is None:
        logger.error("Erro ao arredondar a previsão.")
        return jsonify({"error": "Erro ao arredondar a previsão."}), 500
    
    logger.info(f"Previsão concluída: {predicted_power_rounded}")
    return jsonify({
        'id_sitio': id_sitio,
        'predicted_power': predicted_power_rounded,
        'potencia' : potencia,
        'energia_eolica' : potencia * predicted_power_rounded
    })



if __name__ == '__main__':
    app.run()