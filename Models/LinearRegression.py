from typing import Callable
import numpy as np
import random


class MyLineReg:
    def __init__(
        self,
        n_iter=100,
        learning_rate: float = 0.1,
        metric=None,
        reg=None,
        l1_coef=0,
        l2_coef=0,
        sgd_sample=None,
        random_state=42,
    ) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.val = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def calc_grad(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Функция для вычислений градиента лосса по весам w.
        :param X: np.ndarray размера (n_objects, n_features) с объектами датасета
        :param y: np.ndarray размера (n_objects,) с правильными ответами
        :param weights: np.ndarray размера (n_features,) с весами линейной регрессии
        :return: np.ndarray размера (n_features,) градиент функции потерь по весам weights
        """
        if self.reg == "l1":
            regularization_term = self.l1_coef * np.sign(weights)
        elif self.reg == "l2":
            regularization_term = 2 * self.l2_coef * weights
        elif self.reg == "elasticnet":
            regularization_term = self.l1_coef * np.sign(weights) + 2 * self.l2_coef * weights
        else:
            regularization_term = np.zeros(weights.shape)

        return (2 / len(X)) * np.dot(np.subtract(np.dot(X, weights), y), X) + regularization_term

    def gradient_descent(
        self,
        w_init: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        lr: float | Callable,
        n_iter: int = 100000,
        metric=None,
    ):
        """
        Функция градиентного спуска.
        :param w_init: np.ndarray размера (n_feratures,) -- начальное значение вектора весов
        :param X: np.ndarray размера (n_objects, n_features) -- матрица объекты-признаки
        :param y: np.ndarray размера (n_objects,) -- вектор правильных ответов
        :param lr: float -- параметр величины шага, на который нужно домножать градиент
        :param n_iter: int -- сколько итераций делать
        :param metric: str -- считает метрику
        :return: Список из n_iter объектов np.ndarray размера (n_features,) -- история весов на каждом шаге
        """
        random.seed(self.random_state)
        trace = np.zeros((n_iter + 1, X.shape[1]))
        trace[0] = w_init
        for i in range(n_iter):
            if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
            elif isinstance(self.sgd_sample, float):
                sample_rows_idx = random.sample(range(X.shape[0]), int(X.shape[0] * self.sgd_sample))
            else:
                sample_rows_idx = range(X.shape[0])

            grad = self.calc_grad(X[sample_rows_idx], y[sample_rows_idx], w_init)

            if isinstance(lr, float):
                w_init = w_init - lr * grad
            else:
                w_init = w_init - lr(i + 1) * grad
            trace[i + 1] = w_init
            if metric == "mae":
                self.val = np.mean(np.abs(y[sample_rows_idx] - np.dot(X[sample_rows_idx], w_init)))
            elif metric == "mse":
                self.val = np.mean((y[sample_rows_idx] - np.dot(X[sample_rows_idx], w_init)) ** 2)
            elif metric == "rmse":
                self.val = np.sqrt(np.mean((y[sample_rows_idx] - np.dot(X[sample_rows_idx], w_init)) ** 2))
            elif metric == "mape":
                self.val = np.mean(np.abs((y[sample_rows_idx] - np.dot(X[sample_rows_idx], w_init)) / y)) * 100
            elif metric == "r2":
                y_mean = np.mean(y[sample_rows_idx])
                ss_res = np.sum((y[sample_rows_idx] - np.dot(X[sample_rows_idx], w_init)) ** 2)
                ss_tot = np.sum((y[sample_rows_idx] - y_mean) ** 2)
                self.val = 1 - (ss_res / ss_tot)
        return trace

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: False):
        """Обучаем модель."""
        X = np.asarray(X)
        y = np.asarray(y)
        X = np.hstack([X, np.ones([X.shape[0], 1])])
        w = np.full((X.shape[1]), 1)
        self.weights = self.gradient_descent(
            w,
            X,
            y,
            lr=self.learning_rate,
            n_iter=self.n_iter,
            metric=self.metric,
        )[-1]

    def get_coef(self):
        """Итоговые веса обученной модели."""
        return self.weights[:-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Функция для получения предсказания."""
        X = np.hstack([X, np.ones([X.shape[0], 1])])
        return X.dot(self.weights)

    def get_best_score(self):
        """Финальное значение метрики. Не лучшее, но такое название функции требуется для формата степика."""
        return self.val
