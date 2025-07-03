# -*- coding: utf-8 -*-
"""
Script para previsão de temperatura usando dados meteorológicos
Versão corrigida com:
- Atualização do uso de palette no seaborn
- Melhorias na documentação
- Manutenção de todas as funcionalidades originais
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
import pickle

# Configuração de estilo para os gráficos
plt.style.use('seaborn-v0_8-darkgrid')
palette = sns.color_palette("husl", 3)

def carregar_dados(caminho_arquivo='../data/Belem.csv'):
    """Carrega e processa dados meteorológicos de um arquivo CSV"""
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            linhas = f.readlines()

        dados = []
        for linha in linhas[2:]:
            linha_limpa = linha.strip().replace('"', '')
            partes = linha_limpa.split(';')

            linha_processada = []
            for parte in partes:
                parte = parte.strip()
                if ',' in parte and parte.replace(',', '').replace('.', '').isdigit():
                    parte = parte.replace('.', '').replace(',', '.')
                linha_processada.append(parte)

            dados.append(linha_processada)

        colunas = [
            'Data', 'Hora', 'Temp_Inst', 'Temp_Max', 'Temp_Min',
            'Umidade_Inst', 'Umidade_Max', 'Umidade_Min',
            'Orvalho_Inst', 'Orvalho_Max', 'Orvalho_Min',
            'Pressao_Inst', 'Pressao_Max', 'Pressao_Min',
            'Vento_Vel', 'Vento_Dir', 'Vento_Rajada', 'Radiacao', 'Chuva'
        ]

        df = pd.DataFrame(dados, columns=colunas)
        numeric_cols = df.columns.drop(['Data', 'Hora'])
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        df['Hora'] = df['Hora'].astype(str).str.zfill(4)
        df['DateTime'] = pd.to_datetime(
            df['Data'].dt.strftime('%Y-%m-%d') + ' ' + df['Hora'].str[:2] + ':' + df['Hora'].str[2:]
        )

        df = df.dropna(subset=['DateTime'])
        print('Dados carregados com sucesso!')
        print(f"Total de registros: {len(df)}")
        return df

    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return None

def preprocessar_dados(df):
    """Pré-processamento dos dados e criação de novas features"""
    df['Hora_num'] = df['DateTime'].dt.hour
    df['Dia'] = df['DateTime'].dt.day
    df['Mes'] = df['DateTime'].dt.month
    df['Ano'] = df['DateTime'].dt.year
    df['Dia_do_ano'] = df['DateTime'].dt.dayofyear
    df['Dia_da_semana'] = df['DateTime'].dt.dayofweek

    df['Hora_sin'] = np.sin(2 * np.pi * df['Hora_num']/24)
    df['Hora_cos'] = np.cos(2 * np.pi * df['Hora_num']/24)

    df['Temp_Media_Movel'] = df['Temp_Inst'].rolling(24, min_periods=1).mean()

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df

def plot_comparacao_real_vs_previsto(previsoes):
    """Plota gráfico comparativo entre valores reais e previstos"""
    plt.figure(figsize=(15, 7))

    plt.plot(previsoes['DataHora'], previsoes['Temperatura_Real'],
             '-', color='black', linewidth=2, label='Temperatura Real')

    modelos = ['Regressão Linear', 'Random Forest', 'SVR']
    for i, modelo in enumerate(modelos):
        plt.plot(previsoes['DataHora'], previsoes[f'{modelo}_Previsto'],
                '--', color=palette[i], linewidth=1.5, alpha=0.8,
                label=f'{modelo} Previsto')

    plt.title('Comparação entre Temperatura Real e Previsões dos Modelos', fontsize=14)
    plt.xlabel('Data e Hora', fontsize=12)
    plt.ylabel('Temperatura (°C)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_margem_erro(previsoes):
    """Plota a margem de erro absoluto dos modelos"""
    plt.figure(figsize=(15, 6))

    modelos = ['Regressão Linear', 'Random Forest', 'SVR']
    for i, modelo in enumerate(modelos):
        plt.plot(previsoes['DataHora'], previsoes[f'{modelo}_Margem_Erro'],
                '-', color=palette[i], linewidth=1.5, alpha=0.7,
                label=f'Margem de Erro {modelo}')

    plt.title('Margem de Erro Absoluto dos Modelos', fontsize=14)
    plt.xlabel('Data e Hora', fontsize=12)
    plt.ylabel('Erro Absoluto (°C)', fontsize=12)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_erros_temporais(previsoes):
    """Plota a evolução temporal dos erros de previsão"""
    plt.figure(figsize=(15, 8))

    modelos = ['Regressão Linear', 'Random Forest', 'SVR']
    for i, modelo in enumerate(modelos):
        plt.plot(previsoes['DataHora'], previsoes[f'{modelo}_Erro'],
                '-', color=palette[i], alpha=0.7, linewidth=1,
                label=f'Erro {modelo}')

    plt.title('Evolução Temporal dos Erros de Previsão', fontsize=14)
    plt.xlabel('Data e Hora', fontsize=12)
    plt.ylabel('Erro (Temperatura Real - Prevista) °C', fontsize=12)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_distribuicao_erros(previsoes):
    """Plota a distribuição dos erros de previsão"""
    plt.figure(figsize=(15, 6))
    modelos = ['Regressão Linear', 'Random Forest', 'SVR']

    for i, modelo in enumerate(modelos):
        sns.kdeplot(previsoes[f'{modelo}_Erro'], color=palette[i],
                   label=modelo, linewidth=2)

    plt.title('Distribuição dos Erros de Previsão', fontsize=14)
    plt.xlabel('Erro (Temperatura Real - Prevista) °C', fontsize=12)
    plt.ylabel('Densidade', fontsize=12)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_metricas_comparativas(resultados):
    """Plota gráficos comparativos das métricas dos modelos"""
    metricas = ['RMSE', 'MAE', 'R²']
    plt.figure(figsize=(15, 5))

    for i, metrica in enumerate(metricas):
        plt.subplot(1, 3, i+1)
        sns.barplot(x='Modelo', y=metrica, data=resultados,
                   hue='Modelo', palette=palette, legend=False)
        plt.title(f'Comparação de {metrica}', fontsize=12)
        plt.xticks(rotation=45)
        if metrica == 'R²':
            plt.ylim(0.9, 1.0)

    plt.tight_layout()
    plt.show()

def plot_erros_por_hora(df, previsoes):
    """Plota erros médios por hora do dia"""
    df_erros = df[['DateTime', 'Hora_num']].merge(
        previsoes, left_on='DateTime', right_on='DataHora')

    modelos = ['Regressão Linear', 'Random Forest', 'SVR']
    plt.figure(figsize=(15, 6))

    for i, modelo in enumerate(modelos):
        erro_medio = df_erros.groupby('Hora_num')[f'{modelo}_Margem_Erro'].mean()
        plt.plot(erro_medio.index, erro_medio.values,
                '-o', color=palette[i], linewidth=2, markersize=6,
                label=modelo)

    plt.title('Erro Médio Absoluto por Hora do Dia', fontsize=14)
    plt.xlabel('Hora do Dia', fontsize=12)
    plt.ylabel('Erro Médio Absoluto (°C)', fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def treinar_e_avaliar(df):
    """Treina e avalia os modelos de previsão"""
    features = [
        'Hora_num', 'Dia', 'Mes', 'Ano', 'Dia_do_ano', 'Dia_da_semana',
        'Hora_sin', 'Hora_cos',
        'Umidade_Inst', 'Orvalho_Inst', 'Pressao_Inst',
        'Vento_Vel', 'Vento_Dir', 'Radiacao', 'Chuva',
        'Temp_Media_Movel'
    ]
    target = 'Temp_Inst'

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)

    modelos = [
        ('Regressão Linear', make_pipeline(
            MinMaxScaler(),
            LinearRegression()
        )),
        ('Random Forest', make_pipeline(
            MinMaxScaler(),
            RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        )),
        ('SVR', make_pipeline(
            MinMaxScaler(),
            SVR(kernel='rbf', C=10, gamma='auto', epsilon=0.5)
        ))
    ]

    resultados = []
    previsoes = pd.DataFrame({
        'DataHora': df.loc[X_test.index, 'DateTime'],
        'Temperatura_Real': y_test.values
    })

    for nome, modelo in modelos:
        print(f"\nTreinando {nome}...")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        erro = y_test.values - y_pred
        margem_erro = np.abs(erro)
        percentual_erro = (margem_erro / y_test.values) * 100

        previsoes[f'{nome}_Previsto'] = y_pred
        previsoes[f'{nome}_Erro'] = erro
        previsoes[f'{nome}_Margem_Erro'] = margem_erro
        previsoes[f'{nome}_%_Erro'] = percentual_erro

        resultados.append({
            'Modelo': nome,
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R²': r2_score(y_test, y_pred),
            'Erro_Médio_Absoluto': np.mean(margem_erro),
            'Erro_Perc_Médio': np.mean(percentual_erro)
        })

        with open(f'modelo_{nome.lower().replace(" ", "_")}.pkl', 'wb') as f:
            pickle.dump({'model': modelo, 'features': features}, f)

    return pd.DataFrame(resultados), previsoes, features

def prever_temperatura_amanha(df, modelos_treinados, features):
    """Preve a temperatura média para o dia seguinte"""
    ultima_data = df['DateTime'].max()
    amanha = ultima_data + pd.Timedelta(days=1)

    ultimas_24h = df[df['DateTime'] >= (ultima_data - pd.Timedelta(days=1))]

    features_amanha = {
        'Hora_num': [h for h in range(24)],
        'Dia': [amanha.day] * 24,
        'Mes': [amanha.month] * 24,
        'Ano': [amanha.year] * 24,
        'Dia_do_ano': [amanha.dayofyear] * 24,
        'Dia_da_semana': [amanha.dayofweek] * 24,
        'Hora_sin': np.sin(2 * np.pi * np.array(range(24))/24).tolist(),
        'Hora_cos': np.cos(2 * np.pi * np.array(range(24))/24).tolist(),
        'Umidade_Inst': [ultimas_24h['Umidade_Inst'].median()] * 24,
        'Orvalho_Inst': [ultimas_24h['Orvalho_Inst'].median()] * 24,
        'Pressao_Inst': [ultimas_24h['Pressao_Inst'].median()] * 24,
        'Vento_Vel': [ultimas_24h['Vento_Vel'].median()] * 24,
        'Vento_Dir': [ultimas_24h['Vento_Dir'].mode()[0]] * 24,
        'Radiacao': [ultimas_24h['Radiacao'].median()] * 24,
        'Chuva': [ultimas_24h['Chuva'].median()] * 24,
        'Temp_Media_Movel': [ultimas_24h['Temp_Inst'].median()] * 24
    }

    df_amanha = pd.DataFrame(features_amanha)
    X_amanha = df_amanha[features]

    medias_previstas = {'Data': amanha.strftime('%d/%m/%Y')}

    for nome, modelo_info in modelos_treinados:
        modelo = modelo_info['model']
        previsoes = modelo.predict(X_amanha)
        medias_previstas[f'Média {nome}'] = np.mean(previsoes)

    return medias_previstas

def main():
    """Função principal que executa todo o fluxo"""
    print("Carregando dados...")
    df = carregar_dados()

    if df is None:
        return

    print("\nPré-processando dados...")
    df = preprocessar_dados(df)

    print("\nTreinando modelos...")
    resultados, previsoes, features = treinar_e_avaliar(df)

    print("\nResultados dos Modelos:")
    print(resultados.to_string(index=False))

    modelos_treinados = []
    for nome in ['Regressão Linear', 'Random Forest', 'SVR']:
        with open(f'modelo_{nome.lower().replace(" ", "_")}.pkl', 'rb') as f:
            modelos_treinados.append((nome, pickle.load(f)))

    print("\nPrevendo temperatura média para amanhã...")
    previsao_media_amanha = prever_temperatura_amanha(df, modelos_treinados, features)

    print("\nPrevisão Média de Temperatura para Amanhã:")
    print(f"Data: {previsao_media_amanha['Data']}")
    print(f"Média Regressão Linear: {previsao_media_amanha['Média Regressão Linear']:.2f}°C")
    print(f"Média Random Forest: {previsao_media_amanha['Média Random Forest']:.2f}°C")
    print(f"Média SVR: {previsao_media_amanha['Média SVR']:.2f}°C")

    # Geração dos gráficos
    plot_comparacao_real_vs_previsto(previsoes)
    plot_margem_erro(previsoes)
    plot_erros_temporais(previsoes)
    plot_distribuicao_erros(previsoes)
    plot_metricas_comparativas(resultados)
    plot_erros_por_hora(df, previsoes)

    # Salva os resultados
    pd.DataFrame([previsao_media_amanha]).to_csv('previsao_media_amanha.csv', index=False)
    previsoes.to_csv('previsoes_detalhadas.csv', index=False)
    resultados.to_csv('resultados_modelos.csv', index=False)
    print("\nResultados salvos em arquivos CSV")

if __name__ == "__main__":
    main()