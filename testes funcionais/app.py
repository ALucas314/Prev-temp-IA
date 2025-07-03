import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import pickle

# Função auxiliar para converter hex para rgb
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Configurações de caminho
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SRC_DIR, "..", "models")

# Configurações de estilo
st.set_page_config(
    page_title="ClimateVision - Análise Preditiva",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de cores profissional com tema escuro
colors = {
    'Regressão Linear': '#636EFA',
    'Random Forest': '#EF553B',
    'SVR': '#00CC96',
    'Real': '#AB63FA',
    'Destaque': '#FFA15A',
    'background': '#0E1117',  # Fundo escuro do Streamlit
    'card': '#1E1E1E',
    'text': '#FAFAFA'
}

# Template para gráficos com tema escuro
plotly_template = {
    'layout': {
        'paper_bgcolor': colors['background'],
        'plot_bgcolor': colors['card'],
        'font': {'color': colors['text']},
        'xaxis': {
            'gridcolor': '#2A2A2A',
            'linecolor': '#444',
            'zerolinecolor': '#444'
        },
        'yaxis': {
            'gridcolor': '#2A2A2A',
            'linecolor': '#444',
            'zerolinecolor': '#444'
        },
        'hoverlabel': {
            'bgcolor': colors['card'],
            'font': {'color': colors['text']}
        }
    }
}

# Funções de carregamento de dados
@st.cache_data
def load_data():
    try:
        model_metrics = pd.read_csv(os.path.join(MODELS_DIR, 'resultados_modelos.csv'))
        predictions = pd.read_csv(os.path.join(MODELS_DIR, 'previsoes_detalhadas.csv'))
        predictions['DataHora'] = pd.to_datetime(predictions['DataHora'])
        tomorrow_pred = pd.read_csv(os.path.join(MODELS_DIR, 'previsao_media_amanha.csv')).iloc[0]
        
        models = {
            'Regressão Linear': pickle.load(open(os.path.join(MODELS_DIR, 'modelo_regressão_linear.pkl'), 'rb')),
            'Random Forest': pickle.load(open(os.path.join(MODELS_DIR, 'modelo_random_forest.pkl'), 'rb')),
            'SVR': pickle.load(open(os.path.join(MODELS_DIR, 'modelo_svr.pkl'), 'rb'))
        }
        
        return model_metrics, predictions, tomorrow_pred, models
    
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None, None, None, None

model_metrics, predictions, tomorrow_pred, models = load_data()

if model_metrics is None:
    st.stop()

# Sidebar - Controles de Análise
with st.sidebar:
    st.markdown(f"""
    <div style="border-bottom: 1px solid #444; padding-bottom: 10px; margin-bottom: 20px;">
        <h1 style="color: {colors['Regressão Linear']};">🌡️ ClimateVision</h1>
        <p style="color: {colors['text']};">Controle de Análise Preditiva</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**📅 Período de Análise**")
    min_date = predictions['DataHora'].min().to_pydatetime()
    max_date = predictions['DataHora'].max().to_pydatetime()
    date_range = st.date_input(
        "Selecione o intervalo:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    st.markdown("**🤖 Modelos para Comparação**")
    selected_models = st.multiselect(
        "Selecione os modelos:",
        options=['Regressão Linear', 'Random Forest', 'SVR'],
        default=['Regressão Linear', 'Random Forest', 'SVR'],
        label_visibility="collapsed"
    )
    
    st.markdown("**⚙️ Filtros Avançados**")
    hour_range = st.slider("Intervalo de horas:", 0, 23, (6, 18))
    show_confidence = st.checkbox("Mostrar intervalos de confiança", True)
    smooth_data = st.checkbox("Suavizar dados (média móvel 7 dias)", False)

# Filtragem de dados
def filter_data(df, date_range, hour_range):
    mask = (
        (df['DataHora'] >= pd.to_datetime(date_range[0])) &
        (df['DataHora'] <= pd.to_datetime(date_range[1])) &
        (df['DataHora'].dt.hour >= hour_range[0]) &
        (df['DataHora'].dt.hour <= hour_range[1])
    )
    return df[mask].copy()

filtered_data = filter_data(predictions, date_range, hour_range)

# Cálculo de métricas
def calculate_metrics(df, models):
    metrics = []
    for model in models:
        mae = np.mean(np.abs(df[f'{model}_Erro']))
        rmse = np.sqrt(np.mean(df[f'{model}_Erro']**2))
        r2 = 1 - (np.sum(df[f'{model}_Erro']**2) / np.sum((df['Temperatura_Real'] - df['Temperatura_Real'].mean())**2))
        metrics.append({'Modelo': model, 'MAE': mae, 'RMSE': rmse, 'R²': r2})
    return pd.DataFrame(metrics)

current_metrics = calculate_metrics(filtered_data, selected_models)

# Layout principal
st.markdown(f"""
<div style="border-bottom: 1px solid #444; padding-bottom: 10px; margin-bottom: 30px;">
    <h1 style="color: {colors['Regressão Linear']};">🌡️ ClimateVision - Análise Preditiva de Temperatura</h1>
    <p style="color: {colors['text']};">Dashboard interativo para comparação de modelos de machine learning</p>
</div>
""", unsafe_allow_html=True)

# Métricas rápidas
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Período Analisado", f"{date_range[0].strftime('%d/%m/%Y')} a {date_range[1].strftime('%d/%m/%Y')}")
with col2:
    best_model = current_metrics.loc[current_metrics['RMSE'].idxmin(), 'Modelo']
    st.metric("Melhor Modelo", best_model)
with col3:
    avg_temp = np.mean([tomorrow_pred[f'Média {model}'] for model in selected_models])
    st.metric("Previsão Amanhã", f"{avg_temp:.1f}°C")
with col4:
    total_points = len(filtered_data)
    st.metric("Dados Analisados", f"{total_points:,}")

# Gráfico principal
st.markdown("### 📈 Comparação: Temperatura Real vs Prevista")
fig_main = go.Figure()

fig_main.add_trace(go.Scatter(
    x=filtered_data['DataHora'],
    y=filtered_data['Temperatura_Real'],
    name='Temperatura Real',
    line=dict(color=colors['Real'], width=3),
    mode='lines'
))

for model in selected_models:
    fig_main.add_trace(go.Scatter(
        x=filtered_data['DataHora'],
        y=filtered_data[f'{model}_Previsto'],
        name=f'{model} - Previsto',
        line=dict(color=colors[model], width=2),
        mode='lines'
    ))
    
    if show_confidence:
        rgb = hex_to_rgb(colors[model])
        fig_main.add_trace(go.Scatter(
            x=filtered_data['DataHora'],
            y=filtered_data[f'{model}_Previsto'] + filtered_data[f'{model}_Margem_Erro'],
            name=f'{model} - Margem',
            line=dict(width=0),
            mode='lines',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_main.add_trace(go.Scatter(
            x=filtered_data['DataHora'],
            y=filtered_data[f'{model}_Previsto'] - filtered_data[f'{model}_Margem_Erro'],
            name=f'{model} - Margem',
            fill='tonexty',
            fillcolor=f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.2)",
            line=dict(width=0),
            mode='lines',
            showlegend=False,
            hoverinfo='skip'
        ))

if smooth_data:
    for model in selected_models:
        filtered_data[f'{model}_Smooth'] = filtered_data[f'{model}_Previsto'].rolling(window=24*7, min_periods=1).mean()
        fig_main.add_trace(go.Scatter(
            x=filtered_data['DataHora'],
            y=filtered_data[f'{model}_Smooth'],
            name=f'{model} - Tendência',
            line=dict(color=colors[model], width=3, dash='dash'),
            mode='lines'
        ))

fig_main.update_layout(
    height=600,
    template=plotly_template,
    xaxis_title='Data e Hora',
    yaxis_title='Temperatura (°C)',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig_main, use_container_width=True)

# Análises detalhadas
st.markdown("### 🔍 Análises Detalhadas")
tab1, tab2, tab3 = st.tabs(["Desempenho por Hora", "Distribuição de Erros", "Comparação de Modelos"])

with tab1:
    st.markdown("#### 📊 Desempenho por Hora do Dia")
    hourly_stats = filtered_data.groupby(filtered_data['DataHora'].dt.hour).agg({
        'Temperatura_Real': 'mean',
        **{f'{model}_Margem_Erro': 'mean' for model in selected_models}
    }).reset_index()
    
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Bar(
        x=hourly_stats['DataHora'],
        y=hourly_stats['Temperatura_Real'],
        name='Temperatura Real Média',
        marker_color='#666666',
        opacity=0.5
    ))
    
    for model in selected_models:
        fig_hourly.add_trace(go.Scatter(
            x=hourly_stats['DataHora'],
            y=hourly_stats[f'{model}_Margem_Erro'],
            name=f'Erro {model}',
            line=dict(color=colors[model], width=3),
            mode='lines+markers'
        ))
    
    fig_hourly.update_layout(
        height=500,
        template=plotly_template,
        xaxis_title='Hora do Dia',
        yaxis_title='Temperatura/Erro (°C)',
        xaxis=dict(tickmode='linear', dtick=1),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_hourly, use_container_width=True)

with tab2:
    st.markdown("#### 📉 Distribuição de Erros")
    col1, col2 = st.columns(2)
    
    with col1:
        error_data = []
        for model in selected_models:
            error_data.append(pd.DataFrame({
                'Modelo': model,
                'Erro': filtered_data[f'{model}_Erro']
            }))
        
        error_df = pd.concat(error_data)
        fig_box = px.box(
            error_df,
            x='Modelo',
            y='Erro',
            color='Modelo',
            color_discrete_map=colors,
            points="all",
            title='Distribuição dos Erros'
        )
        fig_box.update_layout(height=500, template=plotly_template)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        fig_density = go.Figure()
        for model in selected_models:
            fig_density.add_trace(go.Violin(
                x=filtered_data[f'{model}_Erro'],
                name=model,
                box_visible=True,
                line_color=colors[model],
                meanline_visible=True
            ))
        fig_density.update_layout(
            height=500,
            template=plotly_template,
            title='Densidade dos Erros',
            xaxis_title='Erro (Real - Previsto)',
            showlegend=False
        )
        st.plotly_chart(fig_density, use_container_width=True)

with tab3:
    st.markdown("#### 🤖 Comparação Detalhada dos Modelos")
    
    st.markdown("##### Correlação entre Erros dos Modelos")
    error_cols = [f'{model}_Erro' for model in selected_models]
    corr_matrix = filtered_data[error_cols].corr()
    corr_matrix.columns = selected_models
    corr_matrix.index = selected_models
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale='RdBu',
        range_color=[-1, 1],
        title='Correlação entre Erros dos Modelos'
    )
    fig_corr.update_layout(template=plotly_template)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("##### Evolução do Desempenho")
    rolling_window = st.slider("Janela de Suavização (horas)", 24, 24*30, 24*7)
    
    fig_evolution = go.Figure()
    for model in selected_models:
        squared_errors = filtered_data[f'{model}_Erro']**2
        filtered_data[f'{model}_Rolling_RMSE'] = np.sqrt(
            squared_errors.rolling(window=rolling_window).mean()
        )
        fig_evolution.add_trace(go.Scatter(
            x=filtered_data['DataHora'],
            y=filtered_data[f'{model}_Rolling_RMSE'],
            name=model,
            line=dict(color=colors[model], width=2),
            mode='lines'
        ))
    
    fig_evolution.update_layout(
        height=500,
        template=plotly_template,
        title=f'RMSE Móvel ({rolling_window} horas)',
        yaxis_title='RMSE (°C)',
        xaxis_title='Data',
        hovermode="x unified"
    )
    st.plotly_chart(fig_evolution, use_container_width=True)

# Rodapé
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {colors['text']}; font-size: 0.9rem;">
    <p>🌍 <strong>ClimateVision</strong> - Sistema de Análise Preditiva | Dados atualizados em {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
</div>
""", unsafe_allow_html=True)