import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from yellowbrick.classifier import ConfusionMatrix

base = pd.read_csv('insurance.csv', keep_default_na=False)
#keep_default_na=False para não transformar os campos vazios em NaN
base = base.drop(columns=['Unnamed: 0'])
#print(base.head)
#print(base.shape)

# sumarizando os dados nulos
#print(base.isnull().sum())

# Variável dependente (coluna 'Accident' no dataset original)
y = base['Accident'].values  

# Variáveis independentes (todas menos 'Accident')
x = base.drop(columns=['Accident']).values
#print(x)

# Convertendo variáveis categóricas em numéricas
labelencoder = LabelEncoder()
for i in range(x.shape[1]):
    if x[:, i].dtype == 'object':
        x[:, i] = labelencoder.fit_transform(x[:, i])
#print(x)

# Dividindo os dados em conjuntos de treino e teste
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.3, random_state=12)

# Criando o modelo Naive Bayes
modelo = GaussianNB()
modelo.fit(x_treinamento, y_treinamento)
#print(modelo)

# Fazendo previsões
previsoes = modelo.predict(x_teste)
#print(previsoes)

# Avaliando o modelo
acuracia = accuracy_score(y_teste, previsoes)
precisao = precision_score(y_teste, previsoes, average='weighted')
recall = recall_score(y_teste, previsoes, average='weighted')
f1 = f1_score(y_teste, previsoes, average='weighted')
print(f'Acurácia: {acuracia}, Precisão: {precisao}, Revocação: {recall}, F1-Score: {f1}')

# Matriz de confusão
report = classification_report(y_teste, previsoes)
print(report)