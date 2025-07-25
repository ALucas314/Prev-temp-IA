-----

Ótimo\! Com base no seu script, criei um `README.md` completo e profissional para o seu repositório GitHub. Ele cobre o que o projeto faz, como usá-lo, os modelos aplicados e os resultados esperados.

-----

# Previsão de Temperatura Horária em Belém (PA) com Aprendizado de Máquina

## Visão Geral do Projeto

Este projeto utiliza **modelos de regressão por Aprendizado de Máquina (ML)** para prever a temperatura horária na cidade de Belém, Pará, Brasil. O objetivo é demonstrar a aplicação de algoritmos como **Regressão Linear**, **Random Forest** e **Support Vector Regression (SVR)** para prever variáveis contínuas em contextos meteorológicos, utilizando dados históricos do Instituto Nacional de Meteorologia (INMET).

O script abrange desde o carregamento e pré-processamento dos dados até a engenharia de *features*, treinamento, avaliação comparativa dos modelos e a geração de visualizações detalhadas do desempenho.

## Funcionalidades

  * **Coleta e Pré-processamento de Dados:** Carrega dados meteorológicos históricos do INMET, tratando valores ausentes e ajustando formatos.
  * **Engenharia de Features:** Cria variáveis temporais (hora, dia, mês, ano, componentes seno/cosseno para periodicidade) e calculadas (média móvel da temperatura) para enriquecer o dataset.
  * **Treinamento e Avaliação de Modelos:**
      * **Regressão Linear:** Como modelo *baseline*.
      * **Random Forest Regressor:** Modelo de *ensemble* robusto.
      * **Support Vector Regression (SVR):** Modelo capaz de capturar não linearidades.
      * Avaliação utilizando **MSE, RMSE, MAE e R²**.
  * **Previsão para o Dia Seguinte:** Demonstra a capacidade preditiva dos modelos ao gerar estimativas para as 24 horas seguintes, baseadas em medianas das últimas 24h.
  * **Visualização de Resultados:** Geração de gráficos comparativos para análise do desempenho dos modelos (Real vs. Previsto, Margem de Erro, Erros Temporais, Distribuição de Erros, Métricas e Erros por Hora).
  * **Persistência de Modelos:** Modelos treinados são salvos em formato `.pkl` para reuso.

## Resultados Notáveis

Os modelos de Machine Learning demonstraram alta eficácia na previsão de temperatura. Destaca-se o **Random Forest** e a **Regressão Linear** por sua excelente acurácia e baixos erros, com o Random Forest atingindo um **R² superior a 0.997** e um **MAE inferior a 0.1°C**. A engenharia de *features* se mostrou crucial para capturar os padrões temporais e tendências dos dados.

## Estrutura do Repositório

```
.
├── data/
│   └── Belem.csv               # Dataset histórico do INMET
├── model_*.pkl                 # Modelos treinados salvos
├── previsao_media_amanha.csv    # Previsão da temperatura média para o dia seguinte
├── previsoes_detalhadas.csv     # Previsões detalhadas por hora para o conjunto de teste
├── resultados_modelos.csv      # Tabela com as métricas de desempenho dos modelos
└── algoritimosRegressao.py     # Script principal do projeto
└── README.md                   # Este arquivo
```

## Como Executar

Para rodar este projeto em sua máquina local, siga os passos abaixo:

### Pré-requisitos

Certifique-se de ter as seguintes bibliotecas Python instaladas:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

### Passo a Passo

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/SeuUsuario/SeuRepositorio.git
    cd SeuRepositorio
    ```
2.  **Organize o Dataset:**
    Coloque o arquivo `Belem.csv` dentro da pasta `data/` na raiz do projeto.
3.  **Execute o Script:**
    Abra seu terminal na pasta raiz do projeto e execute o script principal:
    ```bash
    python algoritimosRegressao.py
    ```

Ao final da execução, os resultados numéricos serão impressos no console, os modelos treinados e os dados de previsão serão salvos como arquivos `.pkl` e `.csv`, e uma série de gráficos de análise de desempenho serão exibidos.

## Contribuição

Contribuições são bem-vindas\! Se você tiver sugestões, melhorias ou encontrar algum problema, sinta-se à vontade para abrir uma *issue* ou enviar um *pull request*.

## Autores

  * Antônio Lucas Costa Araújo
  * Evandro José da Silva Mariano
  * Iago Oliveira de Sousa

-----
