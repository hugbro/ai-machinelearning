import pandas
import numpy

class App:
    def __init__(self, strategy=None, dataset=None):
        self.dataset = dataset or Dataset('mushrooms.csv', 'utf-8')
        self.strategy = strategy or KNearestNeighborsStrategy()

    def run(self):
        features = self.dataset.get_features('cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat')
        target = self.dataset.get_target('class')

        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        test_sample = numpy.array([le.fit_transform(['x','s','y','t','a','f','c','b','k','e','c','s','s','w','w','p','w','o','p','n','n','g'])])

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
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        return le.fit_transform(self.content.pop(key).values)
        


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
