import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Definindo nossos dados
lista = [] #Criar uma lista vazia
X = np.arange(1,21) #X vai de 1 a 20
for x in X: #Para cada elemento de X
  lista.append(x+np.random.randint(6))
Y = np.array(lista)
df = pd.DataFrame({'X':X,'Y':Y[:20]})

#Regressão linear (Numpy)
m,b = np.polyfit(X,Y[:20],1)
eq = m*X+b #Aqui definimos a equação
plt.plot(df['X'],df['Y'], 'o')
plt.plot(df['X'], eq) #Você pode definir a função já dentro do plot, eu optei por definir antes porque fica mais simples de entender
plt.show() #Coloque uma hashtag (#) no antes de "plt" para visualizar apenas a regressão linear gerada pelo Scikit-learn

#Regressão linear (Scikit-learn)
lr = LinearRegression().fit(df['X'].values.reshape(-1,1), df['Y'].values)
predict = lr.predict(df['X'].values.reshape(-1,1))
plt.scatter(df['X'].values, df['Y'].values)
plt.plot(df['X'].values, predict, color='red')
plt.show()