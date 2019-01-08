import pandas
import numpy
from strategies import KNearestNeighborsStrategy, LinearSVCStrategy



class App:
    def __init__(self, dataset=None):
        self.dataset = dataset or Dataset('mushrooms.csv', 'utf-8')
        self.linear_svc_strategy = LinearSVCStrategy()
        self.knn_strategy = KNearestNeighborsStrategy()

    def run(self):

        feature_names = ['stalk-surface-below-ring', 'stalk-color-above-ring',
                         'stalk-color-below-ring', 'veil-type', 'veil-color',
                         'ring-number', 'ring-type', 'spore-print-color',
                         'population', 'habitat']

        sample1 = ['s','w','w','p','w','o','p','n','n','g']

        features = self.dataset.get_features(*feature_names)
        target = self.dataset.get_target('class')

        test_sample = self.dataset.serialize_sample(sample1)

        knn_prediction = self.knn_strategy.execute(
            features,
            target,
            test_sample=test_sample
        )

        linear_svc_prediction = self.linear_svc_strategy.execute(
            features,
            target,
            test_sample=test_sample
        )

        for feature, sample in zip(feature_names, sample1):
            print(f'{feature}: {sample}')

        print("KNN Prediction Result: {}".format(True if knn_prediction else False))

        print("Linear SVC Prediction Result: {}". format(False if linear_svc_prediction else True))


from sklearn import preprocessing
class Dataset:
    def __init__(self, filename, encoding):
        try:
            self.content = pandas.read_csv(filename, encoding=encoding)
            self.service = preprocessing.LabelEncoder()
        except TypeError:
            print('Filename or encoding invalid!')

    def get_target(self, target_key):
        return self._get_key_values(target_key)

    def get_features(self, *args):
        return numpy.vstack(
            tuple(self._get_key_values(a) for a in args)
        ).T

    def _get_key_values(self, key):
        return self.service.fit_transform(self.content.pop(key).values)

    def serialize_sample(self, sample):
        return numpy.array([self.service.fit_transform(sample)])


app = App()

if __name__ == '__main__':
    app.run()
