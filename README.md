# 💳 Detecção de Fraudes em Transações de Cartão de Crédito

Este projeto consiste no desenvolvimento de um modelo de **Machine Learning** em Python para detectar fraudes em transações de cartões de crédito. O objetivo foi comparar diferentes algoritmos de classificação e técnicas de balanceamento de dados para identificar a abordagem com melhor desempenho na detecção de transações fraudulentas.

Durante o desenvoldevimento, foram realizadas as etapas de pré-processamento dos dados, transformação de atributos, balanceamento das classes, treinamento dos modelos e avaliação dos resultados por meio de métricas de classificação.

Os algoritmos avaliados foram:

* Logistic Regression
* Random Forest
* XGBoost

Para tratar o desbalanceamento entre transações legítimas e fraudulentas, foram testadas duas estratégias:

* Undersampling
* Oversampling (SMOTE)

Após a comparação dos experimentos, o repositório apresenta a configuração que obteve o melhor desempenho entre os modelos e técnicas de balanceamento avaliados.

Este foi meu primeiro projeto de Machine Learning e teve como principal objetivo consolidar conhecimentos em:

* Manipulação e pré-processamento de dados com Pandas;
* Engenharia de atributos;
* Balanceamento de classes;
* Treinamento e avaliação de modelos de classificação;
* Comparação de algoritmos de Machine Learning utilizando Scikit-learn e XGBoost.

## Tecnologias utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)

## Objetivo

Desenvolver um pipeline de Machine Learning capaz de identificar fraudes em transações financeiras e comparar diferentes abordagens para lidar com um conjunto de dados altamente desbalanceado, buscando compreender o impacto de cada técnica no desempenho dos modelos.
