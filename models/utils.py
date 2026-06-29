import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ppc_timecourse(df, sim_matrix, model_name, window=5):
    """
    Строит график Group-Level PPC Timecourse.
    df: Оригинальный DataFrame с колонками 'trial_number', 'pumps', 'popped'.
    sim_matrix: Матрица симуляций формы (n_sims, общая_длина_df).
    model_name: Название модели для заголовка и сохранения файла.
    """
    work_df = df[['trial_number', 'pumps', 'popped']].copy()
    
    # Добавляем все симуляции в DataFrame
    for s in range(sim_matrix.shape[0]):
        work_df[f'sim_{s}'] = sim_matrix[s, :]
        
    # НАУЧНЫЙ СТАНДАРТ: Считаем Adjusted Pumps (только для нелопнувших шаров)
    # Заменяем значения на NaN там, где шар лопнул, чтобы исключить их из среднего
    work_df.loc[work_df['popped'] == True, 'pumps'] = np.nan
    for s in range(sim_matrix.shape[0]):
        work_df.loc[work_df['popped'] == True, f'sim_{s}'] = np.nan

    # Агрегируем среднее по каждому триалу (усредняем по всем пользователям)
    real_grouped = work_df.groupby('trial_number')['pumps'].mean()
    sim_grouped = work_df.drop(columns=['pumps', 'popped']).groupby('trial_number').mean()

    # Считаем среднее и 95% интервал наивысшей плотности (HDI) для симуляций
    sim_mean = sim_grouped.mean(axis=1)
    sim_hdi_low = np.percentile(sim_grouped, 2.5, axis=1)
    sim_hdi_high = np.percentile(sim_grouped, 97.5, axis=1)

    # Применяем скользящее среднее (Moving Average) для сглаживания кривой
    real_ma = real_grouped.rolling(window, min_periods=1).mean()
    sim_mean_ma = sim_mean.rolling(window, min_periods=1).mean()
    sim_low_ma = pd.Series(sim_hdi_low, index=sim_grouped.index).rolling(window, min_periods=1).mean()
    sim_high_ma = pd.Series(sim_hdi_high, index=sim_grouped.index).rolling(window, min_periods=1).mean()

    # Отрисовка
    plt.figure(figsize=(10, 6))
    plt.plot(real_ma.index, real_ma, color='black', linewidth=2, label='Real Adjusted Data')
    plt.plot(sim_mean_ma.index, sim_mean_ma, color='blue', linestyle='--', label='Simulated Mean')
    plt.fill_between(sim_mean_ma.index, sim_low_ma, sim_high_ma, color='blue', alpha=0.2, label='95% HDI')

    plt.title(f'PPC Timecourse: {model_name} (MA Window = {window})', fontsize=14)
    plt.xlabel('Trial Number', fontsize=12)
    plt.ylabel('Average Adjusted Pumps', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Сохраняем на диск (без plt.show(), чтобы скрипт не падал на суперкомпьютере)
    filename = f"{model_name.replace(' ', '_').lower()}_timecourse.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f" -> График Timecourse сохранен: {filename}")