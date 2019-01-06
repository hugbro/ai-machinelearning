import numpy as np
import pandas as pd


data = pd.read_csv("acidentes.csv", encoding='latin1')

acidentes = data.pop('Qte Acidentes').values
anos = data.pop('Ano').values

acidentes_anos = np.vstack((acidentes, anos)).T

uf = data.pop('Unidade da Federação').values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(acidentes_anos, uf)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

test_sample = np.array([[3432, 2012]])

prediction = knn.predict(test_sample)

# Prediction + Score
print(prediction[0])
print(knn.score(X_test, y_test))
