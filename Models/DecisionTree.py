from collections import Counter
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd


def gini(t_v):
    """Индекс Джини."""
    t_v = np.array(t_v, dtype=np.int64)
    n = len(t_v)
    if n > 0:
        k2 = np.sum(t_v)
        k1 = n - k2
        p1 = k1 / n
        p2 = k2 / n
        return sum([p1 * (1 - p1), p2 * (1 - p2)])
    return None


def Q(Rm, Rl, Rr, hr):
    """Критерий информативности."""
    if gini(Rm) is not None and gini(Rl) is not None and gini(Rr) is not None:
        if hr == "gini":
            return gini(Rm) - len(Rl) / len(Rm) * gini(Rl) - len(Rr) / len(Rm) * gini(Rr)
        if hr == "var":
            return np.var(Rm) - len(Rl) / len(Rm) * np.var(Rl) - len(Rr) / len(Rm) * np.var(Rr)
    return None


def find_best_split(
    feature_vector: Union[np.ndarray, pd.DataFrame],
    target_vector: Union[np.ndarray, pd.Series],
    task: str = "classification",
    feature_type: str = "real",
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Поиск наилучшего сплита.
    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)
    :param task: либо `classification`, либо `regression`
    :param feature_type: либо `real`, либо `categorical`

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """

    if feature_type == "real":
        sorted_feature_vector = np.sort(feature_vector)
        thresholds = (sorted_feature_vector[:-1] + sorted_feature_vector[1:]) / 2
    elif feature_type == "categorical":
        thresholds = np.array(list(set(feature_vector)))

    ginis = []

    for t in thresholds:
        if feature_type == "real":
            if task == "classification":
                Rm = target_vector
                Rl = target_vector[feature_vector < t]
                Rr = target_vector[feature_vector >= t]
                ginis.append(Q(Rm, Rl, Rr, "gini"))
            elif task == "regression":
                Rm = target_vector
                Rl = target_vector[feature_vector < t]
                Rr = target_vector[feature_vector >= t]
                ginis.append(Q(Rm, Rl, Rr, "var"))

        elif feature_type == "categorical":
            if task == "classification":
                Rm = target_vector
                Rl = target_vector[feature_vector == t]
                Rr = target_vector[feature_vector != t]
                ginis.append(Q(Rm, Rl, Rr, "gini"))
            elif task == "regression":
                Rm = target_vector
                Rl = target_vector[feature_vector == t]
                Rr = target_vector[feature_vector != t]
                ginis.append(Q(Rm, Rl, Rr, "var"))

    ginis = np.array(ginis)
    thresholds = thresholds[ginis is not None]
    ginis_not_None = ginis[ginis != None]
    if len(ginis_not_None) == 0:
        return None, None, None, None
    else:
        gini_best = np.min(ginis_not_None)
        threshold_best = thresholds[ginis.reshape(thresholds.shape) == gini_best][0]
        return thresholds, ginis, threshold_best, gini_best


class DecisionTreeModel:
    def __init__(
        self,
        feature_types: Union[List[str], np.ndarray],
        max_depth: int = None,
        min_samples_split: int = None,
        min_samples_leaf: int = None,
        task: str = "classification",
    ) -> None:

        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        # В этой переменной будем хранить узлы решающего дерева. Каждая вершина хранит в себе идентификатор того,
        # является ли она листовой. Листовые вершины хранят значение класса для предсказания, нелистовые - правого и
        # левого детей (поддеревья для продолжения процедуры предсказания)
        self._tree = {}

        # типы признаков (категориальные или числовые)
        self._feature_types = feature_types

        # гиперпараметры дерева
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self.task = task

    def _fit_node(self, sub_X: np.ndarray, sub_y: np.ndarray, node: dict) -> None:

        # критерий останова
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            # подготавливаем признак для поиска оптимального порога
            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                feature_vector = sub_X[:, feature]

            # ищем оптимальный порог
            _, _, threshold, gini = find_best_split(feature_vector, sub_y, self.task, feature_type)
            if threshold is None:
                pass

            elif gini_best is None or (gini is not None and gini > gini_best):
                feature_best = feature
                gini_best = gini

                # split - маска на объекты, которые должны попасть в левое поддерево
                if feature_type == "real":
                    threshold_best = threshold
                    split = sub_X[:, feature_best] < threshold_best
                elif feature_type == "categorical":
                    # в данной реализации это просто значение категории
                    threshold_best = threshold
                    split = sub_X[:, feature_best] == threshold_best
                else:
                    raise ValueError

        # записываем полученные сплиты в атрибуты класса
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["category_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x: np.ndarray, node: dict) -> int:
        """
        Предсказание начинается с корневой вершины дерева и рекурсивно идёт в левое или правое поддерево в зависимости от значения
        предиката на объекте. Листовая вершина возвращает предсказание.
        :param x: np.array, элемент выборки
        :param node: dict, вершина дерева
        """

        if node["type"] == "terminal":
            return node["class"]

        if self._feature_types[node["feature_split"]] == "real":
            if x[node["feature_split"]] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])

        elif self._feature_types[node["feature_split"]] == "categorical":
            if x[node["feature_split"]] == node["category_split"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._fit_node(X, y, self._tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))

        return np.array(predicted)
