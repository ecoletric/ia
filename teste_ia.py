from flask import Flask, jsonify, request
import requests
import pickle
import pandas as pd  # Importar pandas para criar DataFrame
import unicodedata
from decimal import Decimal, ROUND_HALF_UP
import oracledb as db

app = Flask(__name__)

# Carregar o modelo treinado
with open('modelo_solar.pkl', 'rb') as file:
    solar_prediction_model = pickle.load(file)

with open('melhor_modelo_eolico.pkl', 'rb') as file:
    eolic_prediction_model = pickle.load(file)    

# Definir os nomes das features na mesma ordem do treinamento
feature_names = ['DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DC_POWER']

def kmh_para_ms(velocidade_kmh):
    """
    Converte a velocidade de km/h para m/s.
    
    Parâmetros:
    velocidade_kmh (float): Velocidade em km/h.
    
    Retorna:
    float: Velocidade em m/s.
    """
    return velocidade_kmh / 3.6

def estimar_velocidade_rajada_vento(x_ms, h1=10, h2=100, alpha=0.14):
    """
    Estima a velocidade do vento/rajada a uma altura diferente usando a Lei do Poder.
    
    Parâmetros:
    x_kmh (float): Velocidade do vento a 10 metros em km/h.
    h1 (int): Altura original em metros (padrão é 10 metros).
    h2 (int): Nova altura em metros (padrão é 100 metros).
    alpha (float): Expoente de cisalhamento do vento (padrão é 0.14).
    
    Retorna:
    float: Velocidade do vento estimada a 100 metros em m/s.
    """
      # Converte de km/h para m/s
    return x_ms * (h2 / h1) ** alpha



def remover_acentos(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )


def uf_estado(uf):
    ufs_para_estados = { "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá", "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão", "MG": "Minas Gerais", "MS": "Mato Grosso do Sul", "MT": "Mato Grosso", "PA": "Pará", "PB": "Paraíba", "PE": "Pernambuco", "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte", "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul", "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins"}
    return ufs_para_estados.get(uf)

def latLotCidade(nomeCidade, estado):

    # Função para remover acentos e normalizar textos
    def remover_acentos(texto):
        import unicodedata
        return ''.join(
            c for c in unicodedata.normalize('NFD', texto)
            if unicodedata.category(c) != 'Mn'
        ).lower()

    list_cidades = str(nomeCidade).split(' ')
    novoNomeCidade = remover_acentos('+'.join(list_cidades))
    estado_normalizado = remover_acentos(estado)

    url = f"https://geocoding-api.open-meteo.com/v1/search?name={novoNomeCidade}&count=10&language=pt&format=json"
    response = requests.get(url)

    if response.status_code != 200:
        return {"error": "Falha ao fazer o request para a API de geolocalização"}, response.status_code

    data = response.json()

    # Processar os resultados para encontrar o item desejado
    results = data.get('results', [])

    for result in results:
        if result.get('country') == 'Brasil':
            admin_fields = [
                remover_acentos(result.get('admin1', '').lower()),
                remover_acentos(result.get('admin2', '').lower()),
                remover_acentos(result.get('admin3', '').lower()),
                remover_acentos(result.get('admin4', '').lower())
            ]

            if estado_normalizado in admin_fields:
                # Encontrou o resultado correspondente
                latitude = result.get('latitude')
                longitude = result.get('longitude')
                return {'latitude': latitude, 'longitude': longitude}

    # Se não encontrar, retornar um erro
    return {"error": "Local não encontrado com os parâmetros fornecidos"}, 404

def climaDia(latitude, longitude):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={str(latitude)}&longitude={str(longitude)}&current=relative_humidity_2m&daily=temperature_2m_max,temperature_2m_min,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum&timezone=America%2FSao_Paulo&forecast_days=1"
    response = requests.get(url)
    if response.status_code != 200:
        return {"error" : "Não foi possível puxar informações sobre clima da cidade" }
    data = response.json()
    return data


def arredondar(numero):
    return int(Decimal(numero).to_integral_value(rounding=ROUND_HALF_UP))


@app.route('/predict/', methods=['POST'])
def prever():
    data = request.get_json()
    
    # Obter os dados de entrada
    daily_yield = data.get('daily_yield', '')
    total_yield = data.get('total_yield', '')
    ambient_temperature = data.get('ambient_temperature', '')
    module_temperature = data.get('module_temperature', '')
    irradiation = data.get('irradiation', '')
    dc_power = data.get('dc_power', '')
    
    # Validar e converter os dados de entrada para float
    try:
        daily_yield = float(daily_yield)
        total_yield = float(total_yield)
        ambient_temperature = float(ambient_temperature)
        module_temperature = float(module_temperature)
        irradiation = float(irradiation)
        dc_power = float(dc_power)
    except ValueError:
        return jsonify({'error': 'Dados de entrada inválidos. Certifique-se de que todos os campos são numéricos.'}), 400
    
    # Criar um DataFrame com os nomes das features
    X_new = pd.DataFrame([[daily_yield, total_yield, ambient_temperature, module_temperature, irradiation, dc_power]], columns=feature_names)
    
    # Fazer a previsão usando o modelo carregado
    predicted_ac_power = solar_prediction_model.predict(X_new)
    
    # Retornar a previsão como JSON
    return jsonify({'predicted_ac_power_kwh': predicted_ac_power[0]})

def get_connection():
    return db.connect(user='rm556448', password='fiap24',dsn='oracle.fiap.com.br/orcl') 


def get_endereco(id_sitio):
    sql = "select id_endereco from t_gl_sitio where id_sitio = :id_sitio"
    with get_connection() as con:
        with con.cursor() as cur:
            cur.execute(sql, {"id_sitio": id_sitio})
            id_endereco = cur.fetchone()[0]
    
    sql = "select cidade, uf from t_gl_endereco where id_endereco = :id_endereco"
    with get_connection() as con:
        with con.cursor() as cur:
            cur.execute(sql, {"id_endereco": id_endereco})
            endereco_sitio = cur.fetchone()
    return endereco_sitio


def estimate_wind_conditions(cidade : str, estado : str) -> dict:
    resposta = latLotCidade(nomeCidade=cidade, estado=estado)
    if type(resposta) != dict:
        return jsonify({"error": f"Não foi possível resgatar latLotCidade: {resposta}"})
    latitude = resposta.get('latitude')
    longitude = resposta.get('longitude')
    data = climaDia(latitude=latitude, longitude=longitude)
    print(data)
    daily = data.get('daily', {})
    current = data.get('current', {})
    print(f"current : {current}")
    
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

        umidade = current.get('relative_humidity_2m')

    except (IndexError, TypeError, KeyError) as e:
        return jsonify({"error": f"Erro ao processar dados climáticos: {str(e)}"}), 500
    
    return {
        'temperatura': arredondar(temperatura_maxima - 1.5),
        'direcao_vento_10m': wind_direction_10m_dominant,
        'rajadas_vento_10m': wind_gusts_10m_max,
        'velocidade_maxima_vento_10m': wind_speed_10m_max,
        'umidade' : umidade,
        'direcao_vento_100m' : wind_direction_100m,
        'rajadas_vento_100m' : wind_gusts_100m_max,
        'velocidade_maxima_vento_100m' : wind_speed_100m_max
    }

@app.route('predict-eolic/<int:id_sitio>')
def prever_helice_sitio(id_sitio : int):
    cidade, uf = get_endereco(id_sitio)
    estado = uf_estado(uf)
    resultado = estimate_wind_conditions(cidade, estado)
    
    
    return jsonify({
        "cd"
    })



@app.route('/cidade/', methods=['POST'])
def teste_cidade():
    data = request.get_json()
    cidade = data.get('cidade', '')
    uf = data.get('uf', '')

    ufs_para_estados = { "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá", "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão", "MG": "Minas Gerais", "MS": "Mato Grosso do Sul", "MT": "Mato Grosso", "PA": "Pará", "PB": "Paraíba", "PE": "Pernambuco", "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte", "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul", "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins"}
    
    estado = ufs_para_estados.get(uf)
    resposta = latLotCidade(nomeCidade=cidade, estado=estado)
    if type(resposta) != dict:
        return jsonify({"error" : f"Não foi possível resgatar latLotCidade: {resposta}"})
    latitude = resposta.get('latitude')
    longitude = resposta.get('longitude')
    resultado = climaDia(latitude=latitude, longitude=longitude)


    return jsonify(resultado)
    # return jsonify(resposta)

if __name__ == '__main__':
    print(get_endereco(17))
    # app.run(debug=True)