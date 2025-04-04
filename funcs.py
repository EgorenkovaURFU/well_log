import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate


def feature_importance_plotter(model, feature_names: list):
    '''
    Визуализация значимости признаков.
    model: Модель, для которой будет выполнена визуализация
    features_names: список имен признаков
    '''
    feature_imp = model.feature_importances_
    dictionary = dict(zip(feature_names, feature_imp))
    dictionary = dict(sorted(dictionary.items(), key=lambda item: item[1]))
    fig = plt.figure(figsize=(8, 4))
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, list(dictionary.values()))
    plt.xlim([0, 0.3])
    plt.ylabel('Параметры')
    plt.xlabel('Значимость')
    plt.yticks(y_pos, list(dictionary.keys()))


def cross_validation(X, y, model, scoring, cv_rule):
    '''
    Рассчет метрик на кросс-валидации
    X: признаки,
    y: истиные значения,
    model: модель или pipeline,
    scoring: словарь метрик,
    cv_rule: правило кросс-валидации
    '''
    scores = cross_validate(model, X, y, scoring=scoring, cv=cv_rule)
    df_score = pd.DataFrame(scores)

    print('Ошибка на кросс-валидации')
    print('----------------------------')
    print(df_score.head())
    print('\n')
    print(df_score.mean()[2:])

