import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations
from sklearn.linear_model import LassoCV
import warnings

class LinearModel:
    """
    Класс LinearModel - для построения линейных моделей
    construct_model() - создаём и обучаем модель
    get_metrics() - получаем метрики модели
    visualise_model() - визуализация модели
    backward_elimination() - отбираем параметры обратным исключением
    full_f_stat_selection() - отбираем лучшую модель по F-статистике
    lasso_then_backward() - отбор признаков с помощью Lasso и потом обратное исключение
    search_params() - общая функция для запуска отбора параметров и построения модели
    """
    def __init__(self, data_df, target_col = 'questionnaire_z'):
        self.data = data_df.copy()
        self.X = None
        self.y = None
        self.target_col = target_col

    def construct_model(self):
        df = self.data

        # Разделение на X и y
        self.X = df.drop(columns=['user_id', self.target_col], errors="ignore")
        self.y = df['questionnaire_z']
        self.X = sm.add_constant(self.X)

        # Обучение модели
        model = sm.OLS(self.y, self.X).fit()
        return model
    
    def get_metrics(self, model, method):
        if method == 'summary':
            return model.summary()
        elif method == 'r2':
            return model.rsquared
        elif method == 'f':
            return (model.fvalue, model.f_pvalue)
        else:
            print('No such method')
            return np.nan

    def visualise_model(self, model, X_subset):
        y_pred = model.predict(X_subset)
        y_true = self.y.loc[y_pred.index]
        residuals = y_true - y_pred

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=y_true, y=y_pred, color='royalblue', edgecolor='black', s=60, alpha=0.6)
        plt.xlabel('Фактические значения')
        plt.ylabel('Предсказанные значения')
        plt.title('Фактические vs Предсказанные')
        plt.axline((0, 0), slope=1, color='red', linestyle='--')
        plt.show()

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Остатки')
        plt.title('Остатки vs Предсказанные значения')
        plt.show()


    @staticmethod
    def backward_elimination(X, y, significance_level=0.05):
        X_const = sm.add_constant(X)
        features = list(X_const.columns)

        while True:
            model = sm.OLS(y, X_const[features]).fit()
            p_vals = model.pvalues.drop('const', errors='ignore')

            max_p = p_vals.max()
            if max_p > significance_level:
                worst = p_vals.idxmax()
                features.remove(worst)
                print(f'Исключаем признак: {worst} (p-value={max_p:.4f})')
            else:
                break

        final_model = sm.OLS(y, X_const[features]).fit()
        print('[backward p] Итоговая модель:')
        print(final_model.summary())
        return final_model, X_const[features]
    

    @staticmethod
    def full_f_stat_selection(X, y):
        X_cols = list(X.columns)
        best_model = None
        best_score = -np.inf
        best_features = []

        print('[full F] Начинаем перебор...')
        for k in range(1, len(X_cols) + 1):
            for subset in combinations(X_cols, k):
                
                X_sub = sm.add_constant(X[list(subset)])
                if X_sub.empty:
                    print(f"Пропущено: {subset} — пустой X_sub")
                    continue
                if X_sub.isnull().values.any():
                    print(f"Пропущено: {subset} — есть NaN в X_sub")
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = sm.OLS(y, X_sub).fit()
                    f_score = model.fvalue
                    print(f"Пробуем признаки: {subset} → F={model.fvalue:.3f}, F-p={model.f_pvalue:.3f}")

                if f_score > best_score:
                    best_score = f_score
                    best_model = model
                    best_features = subset

        print(f'[full F] Лучшая модель по F-статистике ({best_score:.3f}) с признаками: {best_features}')
        print(best_model.summary())
        return best_model, sm.add_constant(X[list(best_features)])
    
    @staticmethod
    def lasso_then_backward(X, y, significance_level=0.05):
        lasso = LassoCV(cv=5).fit(X, y)
        mask = lasso.coef_ != 0
        selected = X.columns[mask]

        if len(selected) == 0:
            raise ValueError("Lasso не выбрал ни одного признака.")

        print(f'[lasso] Осталось признаков: {list(selected)}')
        return LinearModel.backward_elimination(X[selected], y, significance_level)
    
    def search_params(self, method='backward'):

        target = self.target_col

        self.X = self.data.drop(columns=['user_id', target], errors="ignore")
        self.y = self.data[target]
        
        X = self.X
        y = self.y
        if method == 'backward':
            return self.backward_elimination(X, y)
        elif method == 'full_f':
            print("X shape:", X.shape)
            print("X columns:", X.columns.tolist())
            return self.full_f_stat_selection(X, y)
        elif method == 'lasso+backward':
            return self.lasso_then_backward(X, y)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
