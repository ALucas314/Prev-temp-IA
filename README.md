# 🌡️ Previsão de Temperatura com IA - Versão COLAB

## 📌 Visão Geral

Este projeto implementa um sistema avançado de previsão de temperatura horária para Belém (PA) utilizando técnicas de Machine Learning com análise de dados meteorológicos. O sistema demonstra como modelos de regressão podem ser aplicados para prever variáveis climáticas com alta precisão.

**Versão COLAB**: Esta é a versão otimizada para execução no Google Colab, com todos os modelos pré-treinados e funcionalidades completas.

**Link para o COLAB** https://colab.research.google.com/drive/1hUGLWRgSp0vrtfgGAhG3ztvL4nZId9Zp#scrollTo=9WyjDCkPscwz

## ✨ Funcionalidades Principais

* **Previsão horária precisa** com múltiplos algoritmos de ML
* **Modelos pré-treinados** prontos para uso imediato
* **Dashboard interativo** para análise de resultados
* **Comparação detalhada** entre modelos
* **Visualizações avançadas** de desempenho
* **Previsão para o dia seguinte** baseada em padrões históricos
* **Execução otimizada** no Google Colab

## 🧠 Modelos Implementados

| Modelo | Descrição | Performance |
|--------|-----------|-------------|
| **Regressão Linear** | Modelo linear clássico | R² = 0.995 |
| **Random Forest** | Ensemble de árvores de decisão | R² = 0.998 |
| **SVR** | Support Vector Regression | R² = 0.979 |

## 🛠️ Tecnologias Utilizadas

* **Python 3.8+**
* **Scikit-Learn** - Algoritmos de ML
* **Pandas** - Manipulação de dados
* **NumPy** - Computação numérica
* **Matplotlib/Seaborn** - Visualizações
* **Pickle** - Serialização de modelos

## 📊 Estrutura do Projeto

```
Prev-temp-IA/
├── data/                           # Dados meteorológicos
│   └── Belem.csv                  # Dataset histórico do INMET
├── models/                         # Modelos treinados
│   ├── modelo_random_forest.pkl   # Modelo Random Forest
│   ├── modelo_regressão_linear.pkl # Modelo Regressão Linear
│   └── modelo_svr.pkl             # Modelo SVR
├── algoritimos_regressao_notebook.ipynb # Notebook principal
└── README.md                      # Este arquivo
```

## 🚀 Como Executar

### Pré-requisitos

* Google Colab ou Jupyter Notebook
* Python 3.8+
* Bibliotecas Python (instaladas automaticamente no Colab)

### Execução no Google Colab

1. **Abra o notebook** `algoritimos_regressao_notebook.ipynb` no Google Colab
2. **Execute todas as células** sequencialmente
3. **Os modelos pré-treinados** serão carregados automaticamente
4. **Resultados e visualizações** serão gerados automaticamente

### Execução Local

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/ALucas314/Prev-temp-IA.git
   cd Prev-temp-IA
   ```

2. **Instale as dependências**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. **Execute o notebook**:
   ```bash
   jupyter notebook algoritimos_regressao_notebook.ipynb
   ```

## 📈 Resultados e Performance

### Métricas de Desempenho

| Modelo | MSE | RMSE | MAE | R² | Erro Médio Absoluto | Erro Percentual Médio |
|--------|-----|------|-----|----|---------------------|----------------------|
| **Random Forest** | 0.020 | 0.140 | 0.096 | 0.998 | 0.096°C | 0.34% |
| **Regressão Linear** | 0.044 | 0.210 | 0.143 | 0.995 | 0.143°C | 0.50% |
| **SVR** | 0.195 | 0.442 | 0.341 | 0.979 | 0.341°C | 1.28% |

### 🏆 Análise Comparativa

O **Random Forest** demonstrou ser o modelo mais eficaz:

* **R² = 0.998** - Explica 99.8% da variância
* **Erro médio absoluto de apenas 0.096°C**
* **Erro percentual médio de apenas 0.34%**
* **Superior em todas as métricas** comparado aos outros modelos

## 🔬 Características Técnicas

### Engenharia de Features

```python
# Features temporais cíclicas
df['Hora_sin'] = np.sin(2 * np.pi * df['Hora_num']/24)
df['Hora_cos'] = np.cos(2 * np.pi * df['Hora_num']/24)

# Média móvel da temperatura
df['Temp_Media_Movel'] = df['Temp_Inst'].rolling(24, min_periods=1).mean()
```

### Pipeline de Modelagem

```python
modelos = [
    ('Regressão Linear', make_pipeline(MinMaxScaler(), LinearRegression())),
    ('Random Forest', make_pipeline(MinMaxScaler(), 
        RandomForestRegressor(n_estimators=200, max_depth=10))),
    ('SVR', make_pipeline(MinMaxScaler(), 
        SVR(kernel='rbf', C=10, gamma='auto', epsilon=0.5)))
]
```

## 📊 Dados Utilizados

* **Fonte**: INMET (Instituto Nacional de Meteorologia)
* **Estação**: Belém (A001)
* **Frequência**: Horária
* **Variáveis**: Temperatura, Umidade, Pressão, Vento, Radiação, Chuva
* **Período**: Dados históricos completos
* **Total de registros**: 4.271 observações

## 🎯 Funcionalidades do Sistema

### 1. Carregamento Automático de Modelos
- Verifica modelos pré-treinados existentes
- Carrega automaticamente se disponíveis
- Treina novos modelos se necessário

### 2. Pré-processamento Inteligente
- Limpeza automática de dados
- Criação de features temporais
- Tratamento de valores ausentes

### 3. Avaliação Completa
- Métricas múltiplas (MSE, RMSE, MAE, R²)
- Comparação visual entre modelos
- Análise de erros temporais

### 4. Previsão Futura
- Previsão para o próximo dia
- 24 previsões horárias
- Média ponderada por modelo

### 5. Visualizações Avançadas
- Gráficos comparativos
- Análise de erros por hora
- Distribuição estatística dos erros

## 🌟 Destaques do Código

* **Código modular** e bem documentado
* **Tratamento de erros** robusto
* **Verificação automática** de modelos existentes
* **Execução otimizada** para Colab
* **Resultados exportáveis** em CSV
* **Visualizações profissionais** com matplotlib/seaborn

## 👥 Autores

* **Antônio Lucas Costa Araújo** - antoniolucas9028@gmail.com
* **Evandro José da Silva Mariano** - evandromariano49@gmail.com
* **Iago Oliveira de Sousa** - iago.sousa@castanhal.ufpa.br

## 🤝 Como Contribuir

Contribuições são bem-vindas! Siga estes passos:

1. Faça um fork do projeto
2. Crie sua branch (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🌟 Suporte

* **Issues**: Reporte bugs ou solicite funcionalidades no GitHub
* **Documentação**: Consulte este README e os comentários no código
* **Contato**: Entre em contato com os autores para dúvidas específicas

---

🌟 **Deixe uma estrela no repositório se você gostou do projeto!** 🌟

**Versão COLAB** - Otimizada para execução no Google Colab com todos os modelos pré-treinados e funcionalidades completas.
