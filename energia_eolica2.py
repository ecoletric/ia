import pandas as pd
from pycaret.regression import *

# Carregar o dataset
df = pd.read_csv('Location2.csv')

# Converter a coluna 'Time' para datetime e extrair apenas dia e mês
df['Time'] = pd.to_datetime(df['Time'])
df['Time'] = df['Time'].dt.strftime('%d-%m')

# Agrupar por 'Time' e calcular a média dos valores
df = df.groupby('Time').mean().reset_index()

# Definir a coluna alvo (por exemplo, 'Power')
target_column = 'Power'  # Substitua por sua variável alvo real

# Inicializar o ambiente do PyCaret
reg = setup(data=df, target=target_column)

# Comparar modelos e selecionar o melhor
best_model = compare_models()

# Exibir o melhor modelo
print(best_model)

# Fazer previsões no conjunto de teste ou novos dados
predictions = predict_model(best_model, data=df)

# Mostrar as primeiras linhas com as previsões
print(predictions.head())

# Salvar o modelo para uso futuro
save_model(best_model, 'melhor_modelo_eolico')