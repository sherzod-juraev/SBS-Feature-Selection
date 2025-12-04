from sklearn.base import clone # ML modelining toza nusxasini yaratish
from sklearn.metrics import accuracy_score # model necha foiz to'g'ri topganligini aniqlaydi
from sklearn.model_selection import train_test_split
from itertools import combinations
from numpy import argmax, ndarray


class SBS:
    """Sequential Backward Selection

    Parameters
    ----------

    estimator : object
        Machine learning model object implementing 'fit' and 'predict'.
        Example: sklearn's LogisticRegression, KNeighborsClassifier, etc.

    k_features : int
        Target number of features to select. SBS will reduce features
        until only this number remains.

    scoring : callable, default=accuracy_score
        Function to evaluate the performance of the model. Should take
        two arguments: true labels and predicted labels.
        Example: sklearn.metrics.accuracy_score, f1_score, etc.

    test_size : float, default=0.25
        Proportion of the dataset to include in the test split. Must be
        between 0.0 and 1.0.

    random_state : int, default=1
        Seed for random number generator used in train-test split. Ensures
        reproducibility of results.

    Attributes
    ----------

    indices_ : tuple
        Current indices of selected features.
    """

    def __init__(self,
                 estimator,
                 k_features: int,
                 /, *,
                 scoring = accuracy_score,
                 test_size: float = .25,
                 random_state: int = 1):

        self.estimator = clone(estimator)
        self.scoring = scoring
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state


    def fit(self, X: ndarray, y: ndarray, /) -> 'SBS':
        """remove unnecessary features"""

        X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X.shape[1]
        self.indices = tuple(range(dim))
        est = self.estimator
        while dim > self.k_features:
            scores, subsets = [], []
            for indices in combinations(self.indices, r=dim - 1):
                self.estimator = clone(est)
                score = self.calculate_score(X_train, X_test, y_train, y_test, indices)
                scores.append(score)
                subsets.append(indices)
            best = argmax(scores)
            self.indices = subsets[best]
            dim -= 1

        return self


    def calculate_score(self, X_train, X_test, y_train, y_test, indices, /) -> float:

        self.estimator.fit(X_train[:, indices], y_train)
        y_predict = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_predict)
        return score