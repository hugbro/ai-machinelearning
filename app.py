import pandas
import numpy

class App:
    def __init__(self, strategy=None, dataset=None):
        self.dataset = dataset or Dataset('pais.csv', 'latin1')
        self.strategy = strategy or KNearestNeighborsStrategy()

    def run(self):
        features = self.dataset.get_features('Qtd Créd. Emit AcInt', 'Ano')
        target = self.dataset.get_target('Países Acordantes')

        test_sample = numpy.array([[195, 2013]])

        self.strategy.execute(features, target, test_sample)


class Dataset:
    def __init__(self, filename, encoding):
        try:
            self.content = pandas.read_csv(filename, encoding=encoding)
        except TypeError:
            print('Filename or encoding invalid!')

    def get_target(self, target_key):
        return self._get_key_values(target_key)

    def get_features(self, *args):
        return numpy.vstack(
            tuple(self._get_key_values(a) for a in args)
        ).T

    def _get_key_values(self, key):
        return self.content.pop(key).values


class KNearestNeighborsStrategy:
    def execute(self, features, target, test_sample=None):
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(features, target)

        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train, y_train)

        if test_sample.any():
            print(f'{knn.predict(test_sample)[0]}')

        print(f'Prediction Score: {round(knn.score(x_test, y_test) * 100, 2)}%')


app = App()

if __name__ == '__main__':
    app.run()
