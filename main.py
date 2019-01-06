import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("pais.csv", encoding='latin1')

valor = data.pop('Qtd Créd. Emit AcInt').values
anos = data.pop('Ano').values
pais = data.pop('Países Acordantes').values

valor_anos = np.vstack((valor, anos)).T

X_train, X_test, y_train, y_test = train_test_split(valor_anos, pais)

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)

test_sample = np.array([[195, 2013]])

prediction = knn.predict(test_sample)

# Prediction + Score
print(prediction[0])
print(knn.score(X_test, y_test))
