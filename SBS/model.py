from sklearn.base import clone # ML modelining toza nusxasini yaratish
from sklearn.metrics import accuracy_score # model necha foiz to'g'ri topganligini aniqlaydi
from sklearn.model_selection import train_test_split
from itertools import combinations
from numpy import argmax, ndarray


class SBS:

    def __init__(self,
                 estimator,
                 k_features: int,
                 /, *,
                 scoring = accuracy_score,
                 test_size: float = .25,
                 random_state: int = 1):

        self.estimator = estimator
        self.scoring = scoring
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state


    def fit(self, X: ndarray, y: ndarray, /) -> 'SBS':
        X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X.shape[1]
        self.indeces = tuple(range(dim))
        while dim > self.k_features:
            scores, subsets = [], []
            for indeces in combinations(self.indeces, r=dim - 1):
                score = self.calculate_score(X_train, X_test, y_train, y_test, indeces)
                scores.append(score)
                subsets.append(indeces)
            best = argmax(scores)
            self.indeces = subsets[best]
            dim -= 1

        return self


    def calculate_score(self, X_train, X_test, y_train, y_test, indeces, /) -> float:
        self.estimator.fit(X_train[:, self.indeces])
        y_predict = self.estimator.predict(X_test)
        score = self.scoring(y_test, y_predict)
        return score