import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import pickle

def hex_to_rgba(hex_color, alpha=1):
    """Converte cor HEX para RGBA com opacidade"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"

# Configurações de caminho
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SRC_DIR, "..", "models")

# Paleta de cores modernizada e sofisticada
colors = {
    'primary': '#00C4CC',          # Ciano vibrante
    'secondary': '#7B61FF',       # Roxo elétrico
    'accent': '#FF6B6B',          # Vermelho coral
    'success': '#2ED573',         # Verde esmeralda
    'background': '#0F172A',      # Fundo escuro azulado
    'card': '#1E293B',            # Card mais claro
    'text': '#E2E8F0',            # Texto branco suave
    'border': '#334155',          # Bordas
    'highlight': '#F9A825',       # Amarelo ouro para destaques
    'warning': '#FFA15A',         # Laranja para avisos
    'gradient_start': '#3B82F6',  # Azul para gradientes
    'gradient_end': '#8B5CF6'     # Roxo para gradientes
}

# Template de gráficos premium
plotly_template = {
    'layout': {
        'paper_bgcolor': colors['background'],
        'plot_bgcolor': colors['card'],
        'font': {'color': colors['text'], 'family': 'Arial'},
        'xaxis': {
            'gridcolor': colors['border'],
            'linecolor': colors['border'],
            'zerolinecolor': colors['border'],
            'title_font': {'size': 14, 'color': colors['text']},
            'showspikes': True,
            'spikemode': 'across',
            'spikesnap': 'cursor',
            'spikecolor': colors['primary'],
            'spikethickness': 1,
            'tickfont': {'color': colors['text']}
        },
        'yaxis': {
            'gridcolor': colors['border'],
            'linecolor': colors['border'],
            'zerolinecolor': colors['border'],
            'title_font': {'size': 14, 'color': colors['text']},
            'showspikes': True,
            'spikemode': 'across',
            'spikecolor': colors['primary'],
            'spikethickness': 1,
            'tickfont': {'color': colors['text']}
        },
        'hoverlabel': {
            'bgcolor': colors['card'],
            'font': {'color': colors['text']},
            'bordercolor': colors['border']
        },
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
            'bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': colors['text']}
        },
        'margin': {'l': 60, 'r': 30, 't': 60, 'b': 60},
        'modebar': {
            'bgcolor': colors['card'],
            'color': colors['text'],
            'activecolor': colors['primary']
        },
        'transition': {'duration': 300}
    }
}

# Configurações da página
st.set_page_config(
    page_title="ClimateVision - Análise Preditiva",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS premium
def load_css():
    st.markdown(f"""
    <style>
        /* Estilos gerais */
        body {{
            color: {colors['text']};
            background-color: {colors['background']};
        }}
        
        /* Sidebar */
        .sidebar .sidebar-content {{
            background: linear-gradient(180deg, {colors['background']} 0%, #0A1120 100%);
            border-right: 1px solid {colors['border']};
            box-shadow: 4px 0 15px rgba(0,0,0,0.2);
        }}
        
        /* Widgets */
        .stSelectbox, .stSlider, .stDateInput, .stTextInput {{
            background-color: {colors['card']};
            border-radius: 10px;
            padding: 10px;
            border: 1px solid {colors['border']};
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .stSelectbox div[data-baseweb="select"] {{
            background-color: {colors['card']};
        }}
        
        /* Botões */
        .stButton>button {{
            background: linear-gradient(90deg, {colors['primary']} 0%, {colors['secondary']} 100%);
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: bold;
            padding: 8px 16px;
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            padding: 0 4px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background: {colors['card']};
            border-radius: 8px;
            padding: 10px 20px;
            border: 1px solid {colors['border']};
            transition: all 0.3s ease;
            margin-right: 0 !important;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(90deg, {colors['primary']} 0%, {colors['secondary']} 100%);
            color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        /* Cards de métricas */
        .metric-card {{
            background: linear-gradient(135deg, {colors['card']} 0%, #25304A 100%);
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid {colors['primary']};
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }}
        
        /* Tooltips */
        .stTooltip {{
            background-color: {colors['card']} !important;
            color: {colors['text']} !important;
            border: 1px solid {colors['border']} !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
            border-radius: 8px !important;
        }}
        
        /* Barra de ferramentas do Plotly */
        .modebar-container {{
            position: absolute !important;
            top: 15px !important;
            right: 15px !important;
            z-index: 1000 !important;
            background-color: {colors['card']} !important;
            border-radius: 8px !important;
            padding: 6px !important;
            border: 1px solid {colors['border']} !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
        }}
        
        .modebar-btn {{
            background-color: transparent !important;
            border-radius: 4px !important;
            transition: all 0.2s ease !important;
        }}
        
        .modebar-btn:hover {{
            background-color: {colors['border']} !important;
            transform: scale(1.1) !important;
        }}
        
        .modebar-btn svg path {{
            fill: {colors['text']} !important;
        }}
        
        .modebar-btn.active svg path {{
            fill: {colors['primary']} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

load_css()

# Função para criar um cabeçalho bonito
def create_header(title, subtitle, icon):
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, {colors['background']} 0%, {colors['card']} 100%);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 30px;
        border-left: 5px solid {colors['primary']};
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    ">
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="
                background: linear-gradient(135deg, {colors['primary']} 0%, {colors['secondary']} 100%);
                width: 50px;
                height: 50px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                color: white;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            ">
                {icon}
            </div>
            <div>
                <h1 style="
                    color: {colors['text']};
                    margin: 0;
                    font-size: 28px;
                    font-weight: 700;
                ">{title}</h1>
                <p style="
                    color: {colors['text']};
                    opacity: 0.8;
                    margin: 5px 0 0 0;
                    font-size: 16px;
                ">{subtitle}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Função para criar cards de métricas modernos
def create_metric_card(label, value, delta=None, icon="📊"):
    delta_color = colors['success'] if (delta and delta >= 0) else colors['accent'] if delta else colors['text']
    delta_icon = "📈" if (delta and delta >= 0) else "📉" if delta else ""
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p style="
                    font-size: 0.9rem;
                    margin-bottom: 8px;
                    opacity: 0.8;
                    color: {colors['text']};
                ">{label}</p>
                <h3 style="
                    margin-top: 0;
                    margin-bottom: 0;
                    font-size: 24px;
                    font-weight: 700;
                    background: linear-gradient(90deg, {colors['primary']} 0%, {colors['secondary']} 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                ">{value}</h3>
            </div>
            <div style="
                background: {hex_to_rgba(colors['primary'], 0.2)};
                width: 50px;
                height: 50px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                color: {colors['primary']};
            ">
                {icon}
            </div>
        </div>
        {f'<div style="display: flex; align-items: center; margin-top: 8px; color: {delta_color}; font-size: 0.9rem;">{delta_icon} {delta if delta >= 0 else -delta}%</div>' if delta is not None else ''}
    </div>
    """, unsafe_allow_html=True)

# Função para configurar os gráficos de forma consistente
def configure_plotly_figure(fig, height=500):
    fig.update_layout(
        height=height,
        template=plotly_template,
        margin=dict(l=60, r=30, t=60, b=60),
        hovermode="x unified",
        modebar=dict(
            orientation='v',
            bgcolor=colors['card'],
            color=colors['text'],
            activecolor=colors['primary']
        )
    )
    return fig

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
    <div style="
        background: linear-gradient(180deg, {colors['primary']} 0%, {colors['secondary']} 100%);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    ">
        <h1 style="color: white; font-size: 24px; margin-bottom: 5px;">🌍 ClimateVision</h1>
        <p style="color: white; font-size: 14px; opacity: 0.9;">Análise Preditiva de Temperatura</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Seção de Período de Análise
    with st.expander("📅 **Período de Análise**", expanded=True):
        min_date = predictions['DataHora'].min().to_pydatetime()
        max_date = predictions['DataHora'].max().to_pydatetime()
        
        selected_dates = st.date_input(
            "Selecione o intervalo:",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            key="date_range_selector",
            format="DD/MM/YYYY"
        )
        
        if len(selected_dates) != 2:
            st.warning("Selecione exatamente duas datas (inicial e final)")
            st.stop()
        
        start_date, end_date = selected_dates[0], selected_dates[1]
        
        if start_date > end_date:
            st.warning("A data inicial deve ser anterior à data final")
            st.stop()
    
    # Seção de Modelos
    with st.expander("🤖 **Modelos para Comparação**", expanded=True):
        selected_models = st.multiselect(
            "Selecione os modelos:",
            options=['Regressão Linear', 'Random Forest', 'SVR'],
            default=['Regressão Linear', 'Random Forest', 'SVR'],
            label_visibility="collapsed"
        )
        
        if not selected_models:
            st.warning("Selecione pelo menos um modelo para análise")
            st.stop()
    
    # Seção de Filtros Avançados
    with st.expander("⚙️ **Filtros Avançados**", expanded=True):
        st.markdown("**Intervalo de Horas**")
        hour_range = st.slider(
            "Selecione o intervalo de horas:",
            0, 23, (6, 18),
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            show_confidence = st.checkbox("Mostrar intervalos de confiança", True)
        with col2:
            smooth_data = st.checkbox("Suavizar dados (média móvel)", False)
        
        if smooth_data:
            window_size = st.slider(
                "Tamanho da janela (horas):",
                1, 168, 24,
                help="Número de horas para calcular a média móvel"
            )

# Função de filtragem de dados
def filter_data(df, date_range, hour_range):
    try:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        
        mask = (
            (df['DataHora'] >= start_date) &
            (df['DataHora'] <= end_date) &
            (df['DataHora'].dt.hour >= hour_range[0]) &
            (df['DataHora'].dt.hour <= hour_range[1])
        )
        return df[mask].copy()
    except Exception as e:
        st.error(f"Erro ao filtrar dados: {str(e)}")
        return pd.DataFrame()

date_range = [start_date, end_date]
filtered_data = filter_data(predictions, date_range, hour_range)

# Função para calcular métricas
def calculate_metrics(df, models):
    metrics = []
    for model in models:
        mae = np.mean(np.abs(df[f'{model}_Erro']))
        rmse = np.sqrt(np.mean(df[f'{model}_Erro']**2))
        r2 = 1 - (np.sum(df[f'{model}_Erro']**2) / np.sum((df['Temperatura_Real'] - df['Temperatura_Real'].mean())**2))
        metrics.append({'Modelo': model, 'MAE': mae, 'RMSE': rmse, 'R²': r2})
    return pd.DataFrame(metrics)

# Layout principal
create_header(
    "ClimateVision - Análise Preditiva de Temperatura",
    "Dashboard interativo para comparação de modelos de machine learning",
    "🌡️"
)

# Seção de métricas rápidas
st.markdown("### 📊 Visão Geral")
col1, col2, col3, col4 = st.columns(4)
with col1:
    create_metric_card(
        "Período Analisado", 
        f"{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}",
        icon="📅"
    )
    
with col2:
    current_metrics = calculate_metrics(filtered_data, selected_models)
    best_model = current_metrics.loc[current_metrics['RMSE'].idxmin(), 'Modelo']
    create_metric_card(
        "Melhor Modelo", 
        best_model,
        icon="🏆"
    )
        
with col3:
    avg_temp = np.mean([tomorrow_pred[f'Média {model}'] for model in selected_models])
    create_metric_card(
        "Previsão Amanhã", 
        f"{avg_temp:.1f}°C",
        icon="🔮"
    )
        
with col4:
    total_points = len(filtered_data)
    create_metric_card(
        "Dados Analisados", 
        f"{total_points:,}",
        icon="📊"
    )

# Seção principal de visualização
st.markdown("### 📈 Comparação: Temperatura Real vs Prevista")

if filtered_data.empty:
    st.warning("Nenhum dado disponível para o período e filtros selecionados")
    st.stop()

# Configurações de cores para modelos
model_colors = {
    'Regressão Linear': colors['primary'],
    'Random Forest': colors['success'],
    'SVR': colors['secondary']
}

# Gráfico principal interativo
fig_main = go.Figure()

# Adiciona linha da temperatura real com efeito de sombra
fig_main.add_trace(go.Scatter(
    x=filtered_data['DataHora'],
    y=filtered_data['Temperatura_Real'],
    name='Temperatura Real',
    line=dict(color=colors['highlight'], width=4),
    mode='lines',
    hovertemplate='<b>%{x|%d/%m/%Y %H:%M}</b><br>%{y:.1f}°C<extra></extra>'
))

# Adiciona modelos selecionados com efeitos visuais
for model in selected_models:
    # Linha principal da previsão
    fig_main.add_trace(go.Scatter(
        x=filtered_data['DataHora'],
        y=filtered_data[f'{model}_Previsto'],
        name=f'{model} - Previsto',
        line=dict(color=model_colors[model], width=3),
        mode='lines',
        hovertemplate='<b>%{x|%d/%m/%Y %H:%M}</b><br>%{y:.1f}°C<extra></extra>'
    ))
    
    # Intervalo de confiança com gradiente
    if show_confidence:
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
            fillcolor=hex_to_rgba(model_colors[model], 0.15),
            line=dict(width=0),
            mode='lines',
            showlegend=False,
            hoverinfo='skip'
        ))

# Adiciona suavização se ativada
if smooth_data:
    window = window_size if 'window_size' in locals() else 24*7
    for model in selected_models:
        filtered_data[f'{model}_Smooth'] = filtered_data[f'{model}_Previsto'].rolling(window=window, min_periods=1).mean()
        fig_main.add_trace(go.Scatter(
            x=filtered_data['DataHora'],
            y=filtered_data[f'{model}_Smooth'],
            name=f'{model} - Tendência',
            line=dict(color=model_colors[model], width=4, dash='dash'),
            mode='lines',
            hovertemplate='<b>%{x|%d/%m/%Y %H:%M}</b><br>%{y:.1f}°C<extra></extra>'
        ))

# Configuração do gráfico principal
fig_main = configure_plotly_figure(fig_main, height=600)
fig_main.update_layout(
    xaxis_title='Data e Hora',
    yaxis_title='Temperatura (°C)',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Adiciona controles de range slider
fig_main.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all")
            ]),
            bgcolor=colors['card'],
            activecolor=colors['primary'],
            bordercolor=colors['border']
        ),
        rangeslider=dict(
            visible=True,
            bgcolor=colors['card'],
            bordercolor=colors['border']
        ),
        type="date"
    )
)

st.plotly_chart(fig_main, use_container_width=True)

# Análises detalhadas com abas estilizadas
st.markdown("### 🔍 Análises Detalhadas")
tab1, tab2, tab3 = st.tabs(["📊 Desempenho por Hora", "📉 Distribuição de Erros", "🤖 Comparação de Modelos"])

with tab1:
    st.markdown("#### Desempenho por Hora do Dia")
    
    # Calcula estatísticas por hora
    hourly_stats = filtered_data.groupby(filtered_data['DataHora'].dt.hour).agg({
        'Temperatura_Real': ['mean', 'std'],
        **{f'{model}_Erro': ['mean', 'std'] for model in selected_models}
    }).reset_index()
    
    # Gráfico de barras interativo
    fig_hourly = go.Figure()
    
    # Barra da temperatura real com efeito 3D
    fig_hourly.add_trace(go.Bar(
        x=hourly_stats['DataHora'],
        y=hourly_stats[('Temperatura_Real', 'mean')],
        error_y=dict(
            type='data',
            array=hourly_stats[('Temperatura_Real', 'std')],
            visible=True
        ),
        name='Temperatura Real Média',
        marker=dict(
            color=colors['highlight'],
            opacity=0.8,
            line=dict(width=2, color=colors['border'])
        ),
        hovertemplate='<b>%{x}:00h</b><br>Média: %{y:.1f}°C<br>Desvio: ±%{error_y.array:.1f}°C<extra></extra>'
    ))
    
    # Linhas de erro para cada modelo
    for model in selected_models:
        fig_hourly.add_trace(go.Scatter(
            x=hourly_stats['DataHora'],
            y=hourly_stats[(f'{model}_Erro', 'mean')].abs(),
            name=f'Erro Médio {model}',
            line=dict(color=model_colors[model], width=3),
            mode='lines+markers',
            marker=dict(size=8),
            hovertemplate='<b>%{x}:00h</b><br>Erro Médio: %{y:.2f}°C<extra></extra>'
        ))
    
    # Configuração do gráfico
    fig_hourly = configure_plotly_figure(fig_hourly)
    fig_hourly.update_layout(
        xaxis_title='Hora do Dia',
        yaxis_title='Temperatura/Erro (°C)',
        xaxis=dict(tickmode='linear', dtick=1),
        barmode='overlay',
        bargap=0.1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_hourly, use_container_width=True)

with tab2:
    st.markdown("#### Distribuição de Erros")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Box Plot dos Erros")
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
            color_discrete_map=model_colors,
            points="all",
            title='Distribuição dos Erros por Modelo',
            hover_data={'Erro': ':.2f'},
            notched=True
        )
        
        # Configuração do gráfico
        fig_box = configure_plotly_figure(fig_box)
        fig_box.update_layout(
            xaxis_title='Modelo',
            yaxis_title='Erro (Real - Previsto)',
            boxmode='group',
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        st.markdown("##### Densidade dos Erros")
        fig_density = go.Figure()
        for model in selected_models:
            fig_density.add_trace(go.Violin(
                x=filtered_data[f'{model}_Erro'],
                name=model,
                box_visible=True,
                line_color=model_colors[model],
                meanline_visible=True,
                points=False,
                hoverinfo='x',
                spanmode='hard',
                width=0.9
            ))
        
        # Configuração do gráfico
        fig_density = configure_plotly_figure(fig_density)
        fig_density.update_layout(
            title='Distribuição de Densidade dos Erros',
            xaxis_title='Erro (Real - Previsto)',
            showlegend=True,
            violingap=0.1,
            violingroupgap=0.1,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_density, use_container_width=True)

with tab3:
    st.markdown("#### Comparação Detalhada dos Modelos")
    
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
        title='Matriz de Correlação entre Erros dos Modelos',
        aspect="auto",
        labels=dict(color="Correlação")
    )
    
    # Configuração do gráfico
    fig_corr = configure_plotly_figure(fig_corr)
    fig_corr.update_layout(
        xaxis_title='Modelo',
        yaxis_title='Modelo',
        coloraxis_colorbar=dict(
            title="Correlação",
            thickness=20,
            tickvals=[-1, -0.5, 0, 0.5, 1]
        )
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("##### Evolução do Desempenho")
    rolling_window = st.slider(
        "Janela de Suavização (horas):", 
        24, 24*30, 24*7, 
        key="rolling_window",
        help="Tamanho da janela para cálculo do RMSE móvel"
    )
    
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
            line=dict(color=model_colors[model], width=3),
            mode='lines',
            hovertemplate='<b>%{x|%d/%m/%Y %H:%M}</b><br>RMSE: %{y:.2f}°C<extra></extra>'
        ))
    
    # Configuração do gráfico
    fig_evolution = configure_plotly_figure(fig_evolution)
    fig_evolution.update_layout(
        title=f'RMSE Móvel ({rolling_window} horas)',
        yaxis_title='RMSE (°C)',
        xaxis_title='Data',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Adiciona controles de range slider
    fig_evolution.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(step="all")
                ]),
                bgcolor=colors['card'],
                activecolor=colors['primary'],
                bordercolor=colors['border']
            ),
            rangeslider=dict(
                visible=True,
                bgcolor=colors['card'],
                bordercolor=colors['border']
            ),
            type="date"
        )
    )
    
    st.plotly_chart(fig_evolution, use_container_width=True)

# Rodapé premium
st.markdown("---")
st.markdown(f"""
<div style="
    background: linear-gradient(90deg, {colors['card']} 0%, {colors['background']} 100%);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-top: 30px;
">
    <div style="display: flex; justify-content: center; align-items: center; gap: 15px;">
        <div style="
            background: linear-gradient(135deg, {colors['primary']} 0%, {colors['secondary']} 100%);
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 18px;
        ">
            🌍
        </div>
        <div>
            <h3 style="margin: 0; color: {colors['text']};">ClimateVision - Sistema de Análise Preditiva</h3>
            <p style="margin: 5px 0 0 0; color: {colors['text']}; opacity: 0.8; font-size: 14px;">
                Dados atualizados em {datetime.now().strftime('%d/%m/%Y %H:%M')} | Versão 2.0
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)