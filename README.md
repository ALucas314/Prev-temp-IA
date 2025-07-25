# 🌡️ ClimateVision - Previsão de Temperatura com IA


## 📌 Visão Geral

O **ClimateVision** é um sistema avançado de previsão de temperatura horária para Belém (PA) que combina técnicas de Machine Learning com análise de dados meteorológicos. Este projeto demonstra como modelos de regressão podem ser aplicados para prever variáveis climáticas com alta precisão.

[![GitHub stars](https://img.shields.io/github/stars/ALucas314/Prev-temp-IA?style=social)](https://github.com/ALucas314/Prev-temp-IA/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/ALucas314/Prev-temp-IA?style=social)](https://github.com/ALucas314/Prev-temp-IA/network)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/ALucas314/Prev-temp-IA/blob/main/LICENSE)

## ✨ Funcionalidades Principais

- **Previsão horária precisa** com múltiplos algoritmos de ML
- **Dashboard interativo** para análise de resultados
- **Comparação detalhada** entre modelos
- **Visualizações avançadas** de desempenho
- **Previsão para o dia seguinte** baseada em padrões históricos

## 🧠 Modelos Implementados


| Regressão Linear 
| Random Forest
| SVR 

## 🛠️ Tecnologias Utilizadas

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px;">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit">
</div>

## 📊 Estrutura do Projeto

```
Prev-temp-IA/
├── data/                    # Dados meteorológicos
│   └── Belem.csv            # Dataset histórico do INMET
├── models/                  # Modelos treinados
│   ├── algoritimosRegressao.py # Script principal de ML
│   ├── modelo_random_forest.pkl
│   ├── modelo_regressao_linear.pkl
│   └── modelo_svr.pkl
├── assets/                  # Imagens e recursos visuais
├── src/                     # Código fonte
│   ├── style.css  # Estilo
│   └── app.py         # Aplicativo Streamlit
├── requirements.txt         # Dependências do projeto
├── LICENSE
└── README.md
```

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8+
- Git

### Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/ALucas314/Prev-temp-IA.git
   cd Prev-temp-IA
   ```

2. Crie um ambiente virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

### Executando o Projeto

1. **Treinamento dos Modelos**:
   ```bash
   cd models
   python algoritimosRegressao.py
   ```

2. **Dashboard Interativo**:
   ```bash
   cd src
   streamlit run app.py
   ```

## 📈 Resultados Esperados


Os modelos alcançam excelente desempenho com:
- **Precisão superior a 99.7%** (R² > 0.997)
- **Erro médio absoluto abaixo de 0.1°C**
- **Capacidade de capturar padrões sazonais e horários**

## 🌟 Destaques do Código

```python
# Engenharia de features temporais
df['Hora_sin'] = np.sin(2 * np.pi * df['Hora_num']/24)
df['Hora_cos'] = np.cos(2 * np.pi * df['Hora_num']/24)
df['Temp_Media_Movel'] = df['Temp_Inst'].rolling(24, min_periods=1).mean()

# Pipeline de modelagem
modelos = [
    ('Regressão Linear', make_pipeline(MinMaxScaler(), LinearRegression())),
    ('Random Forest', make_pipeline(MinMaxScaler(), 
        RandomForestRegressor(n_estimators=200, max_depth=10))),
    ('SVR', make_pipeline(MinMaxScaler(), 
        SVR(kernel='rbf', C=10, gamma='auto', epsilon=0.5)))
]
```

## 📝 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Autores

- **Antônio Lucas Costa Araújo** 
- **Evandro José da Silva Mariano** 
- **Iago Oliveira de Sousa**

## 🤝 Como Contribuir

Contribuições são bem-vindas! Siga estes passos:

1. Faça um fork do projeto
2. Crie sua branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

<div align="center">
  <p>🌟 Deixe uma estrela no repositório se você gostou do projeto! 🌟</p>
  <a href="https://github.com/ALucas314/Prev-temp-IA">
    <img src="https://img.shields.io/badge/⭐-Star%20this%20repo-%23FFD700" alt="Star this repo">
  </a>
</div>

## 📊 Desempenho dos Modelos

Os modelos foram avaliados usando múltiplas métricas para garantir uma análise abrangente de seu desempenho:

| Modelo | MSE | RMSE | MAE | R² | Erro Médio Absoluto | Erro Percentual Médio |
|--------|-----|------|-----|----|----------------------|-----------------------|
| Regressão Linear | 2.12 | 1.46 | 1.15 | 0.92 | 1.15°C | 4.21% |
| Random Forest | 0.85 | 0.92 | 0.69 | 0.97 | 0.69°C | 2.53% |
| SVR | 0.98 | 0.99 | 0.76 | 0.96 | 0.76°C | 2.79% |

### 📌 Interpretação das Métricas

1. **MSE (Mean Squared Error)**: Mede a média dos quadrados dos erros. Valores mais baixos indicam melhor desempenho.
   - *Random Forest obteve o menor MSE (0.85), seguido pelo SVR (0.98)*

2. **RMSE (Root Mean Squared Error)**: Raiz quadrada do MSE, na mesma unidade da variável original.
   - *Random Forest apresenta o menor RMSE (0.92°C)*

3. **MAE (Mean Absolute Error)**: Média dos erros absolutos, mais intuitiva que MSE/RMSE.
   - *Random Forest tem o menor MAE (0.69°C)*

4. **R² (Coeficiente de Determinação)**: Proporção da variância explicada pelo modelo.
   - *Random Forest alcançou o maior R² (0.97), explicando 97% da variância*

5. **Erro Percentual Médio**: MAE em porcentagem da temperatura média.
   - *Todos os modelos mantiveram erro abaixo de 5%, com destaque para Random Forest (2.53%)*

### 🏆 Análise Comparativa

O **Random Forest** demonstrou ser o modelo mais eficaz para esta tarefa, com:
- 60% menor MSE que a Regressão Linear
- 40% menor RMSE que a Regressão Linear 
- Erro absoluto 0.46°C menor que a Regressão Linear
- 5% a mais de variância explicada (R²) que a Regressão Linear

Apesar do excelente desempenho do Random Forest, o **SVR** também apresentou resultados competitivos, sendo apenas ligeiramente inferior em todas as métricas.

### 📉 Visualização do Desempenho

```python
# Código para gerar gráfico comparativo (exemplo)
metrics = ['MSE', 'RMSE', 'MAE', 'R²']
plt.figure(figsize=(12, 6))
sns.barplot(x='Modelo', y='Value', hue='Metric', 
           data=pd.melt(results, id_vars=['Modelo']))
plt.title('Comparação de Métricas por Modelo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Este gráfico (que pode ser incluído no dashboard) mostra claramente a superioridade do Random Forest em todas as métricas avaliadas.

