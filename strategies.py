from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


class BaseStrategy:
    def execute(self, features, target, test_sample=None):
        x_train, x_test, y_train, y_test = train_test_split(features, target)

        self.service.fit(x_train, y_train)

        print(
            f'Prediction Score for {self}: {self.service.score(x_test, y_test)}'
        )

        if test_sample.any():
            return self.service.predict(test_sample)[0]


class KNearestNeighborsStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        self.service = KNeighborsClassifier(n_neighbors=5)
        super(KNearestNeighborsStrategy, self).__init__(**kwargs)

    def __str__(self):
        return 'KNearestNeighbors'

class LinearSVCStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        self.service = LinearSVC(random_state=0, tol=1e-5, max_iter=50000)
        super(LinearSVCStrategy, self).__init__(**kwargs)

    def __str__(self):
        return 'Linear SVC'
