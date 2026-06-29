import os
import pandas as pd
import numpy as np
import arviz as az

# Перевод Matplotlib в неинтерактивный режим (обязательно для суперкомпьютера)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Импорт ваших моделей (замените пути на актуальные)
try:
    from models.wullhorst_model1_hot import HBA_HOTModel
    from models.wullhorst_model2_hot import HOTModel2_HBA
    from models.haffke_model_hot import HaffkeHBAModel_Hot, prepare_haffke_input
except ImportError as e:
    print(f"[!] Предупреждение при импорте: {e}")

# =====================================================================
# Вспомогательные функции (вынесены сюда для автономности скрипта)
# =====================================================================

def save_hba_parameters(trace, user_ids, param_names, filename="hba_user_parameters.csv"):
    post = trace.posterior
    data = {"user_id": user_ids}
    for param in param_names:
        data[f"{param}_mean"] = post[param].mean(dim=["chain", "draw"]).values
        data[f"{param}_median"] = post[param].median(dim=["chain", "draw"]).values
        data[f"{param}_sd"] = post[param].std(dim=["chain", "draw"]).values
        
        # HDI интервалы
        hdi = az.hdi(post[param])
        data[f"{param}_hdi_3%"] = hdi[param].sel(hdi='lower').values
        data[f"{param}_hdi_97%"] = hdi[param].sel(hdi='higher').values
        
    df_params = pd.DataFrame(data)
    df_params.to_csv(filename, index=False)
    print(f"[✓] Индивидуальные параметры (с HDI и SD) сохранены в {filename}")
    return df_params

def run_hba_parameter_recovery(template_data, n_subjects=50):
    print(f"\n[*] HBA Parameter Recovery (N={n_subjects})")
    
    uids = template_data['user_id'].unique()[:n_subjects]
    template = pd.concat([template_data[template_data['user_id'] == uid] for uid in uids]).copy()
    template['subj_idx'], subj_labels = pd.factorize(template['user_id'])
    
    rng = np.random.default_rng(42)
    true_rho = rng.uniform(0.01, 1.0, n_subjects)
    true_lam = rng.uniform(0.01, 3.0, n_subjects)
    true_beta = rng.uniform(0.01, 1.0, n_subjects)
    
    simulated_dfs = []
    for idx, uid in enumerate(subj_labels):
        subj_data = template[template['user_id'] == uid]
        sim_df = HBA_HOTModel.simulate_data(subj_data, true_rho[idx], true_lam[idx], true_beta[idx])
        simulated_dfs.append(sim_df)
        
    recovery_data = pd.concat(simulated_dfs)
    
    # Для recovery снижено количество сэмплов для экономии времени
    recovery_model = HBA_HOTModel(recovery_data)
    rec_idata, _ = recovery_model.fit(draws=1000, tune=1000, chains=4)
    
    fit_rho = rec_idata.posterior['rho'].median(dim=("chain", "draw")).values
    fit_lam = rec_idata.posterior['lam'].median(dim=("chain", "draw")).values
    fit_beta = rec_idata.posterior['beta'].median(dim=("chain", "draw")).values
    
    df_results = pd.DataFrame({
        "true_rho": true_rho, "fit_rho": fit_rho,
        "true_lam": true_lam, "fit_lam": fit_lam,
        "true_beta": true_beta, "fit_beta": fit_beta
    })
    
    # Визуализация 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('HBA Parameter Recovery (Model 1)', fontsize=16)
    
    params = [("rho", "true_rho", "fit_rho"), ("lambda", "true_lam", "fit_lam"), ("beta", "true_beta", "fit_beta")]
    for ax, (p_name, true_col, fit_col) in zip(axes, params):
        r_val = np.corrcoef(df_results[true_col], df_results[fit_col])[0, 1]
        sns.scatterplot(x=df_results[true_col], y=df_results[fit_col], ax=ax, s=60, color='indigo', alpha=0.7)
        
        min_v = min(df_results[true_col].min(), df_results[fit_col].min())
        max_v = max(df_results[true_col].max(), df_results[fit_col].max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2)
        
        ax.set_title(f'{p_name} (Pearson r: {r_val:.3f})')
        ax.set_xlabel('True values')
        ax.set_ylabel('Recovered values')
        
    plt.tight_layout()
    # ИСПРАВЛЕНО: Сохранение файла вместо вызова интерактивного окна
    plt.savefig('hot_wullhorst1_recovery.png', dpi=300)
    plt.close()
    
    return df_results

# =====================================================================
# Главный цикл запуска скрипта
# =====================================================================

# =====================================================================
# Главный цикл запуска скрипта
# =====================================================================

def main():
    print("=" * 70)
    print("=== ЗАПУСК ПАКЕТНОГО ВЫЧИСЛЕНИЯ CCT-HOT МОДЕЛЕЙ (HBA) ===")
    print("=" * 70)
    
    # 0. Загрузка данных
    data_path = 'data/hot_data.csv'

    hot_df = pd.read_csv(data_path)
    print(f"[✓] Данные загружены. Уникальных участников: {hot_df['user_id'].nunique()}")

    # -----------------------------------------------------------------
    # 1. МОДЕЛЬ: HBA_HOTModel (Wullhorst Model 1)
    # -----------------------------------------------------------------
    print("\n" + "#" * 50)
    print("=== ЭТАП 1: HBA_HOTModel (Wullhorst 1) ===")
    print("#" * 50)
    
    hba_model = HBA_HOTModel(hot_df, scale_factor=100.0)
    # hba_model = HBA_HOTModel(hot_df, scale_factor=100.0)
    # idata, loo = hba_model.fit(draws=1500, tune=1000, chains=4)

    # # СОХРАНЕНИЕ TRACE (NETCDF ФАЙЛ)
    # az.to_netcdf(idata, "hot_wullhorst1_trace.nc")
    # print("[✓] Trace для Model 1 сохранен в hot_wullhorst1_trace.nc")
    # with open("hot_wullhorst1_looic.txt", "w") as f:

    #     f.write(str(loo))
    # ЧТЕНИЕ TRACE (без переобучения)
    # print("[*] Загрузка сохраненного trace: hot_wullhorst1_trace.nc")
    # idata = az.from_netcdf("hot_wullhorst1_trace.nc")

    # save_hba_parameters(idata, hba_model.subj_labels, ['rho', 'lam', 'beta'], "hot_wullhorst1_params.csv")

    # print("\n=== ЭТАП 2: Posterior Predictive Check ===")
    # plt.close('all')
    # ppc_metrics_df = hba_model.posterior_predictive_check(idata, save_plot_path="hot_wullhorst1_ppc_timecourse.png")
    # ppc_metrics_df.to_csv('hot_wullhorst1_ppc_metrics.csv', index=False)
    
    # print("\nСредние метрики PPC по выборке:")
    # print(ppc_metrics_df[['R2', 'RMSE', 'MAE', 'MSD', 'Hit_Rate', 'ppp']].mean().round(3))
    # print(f"[✓] PPC графики и метрики сохранены.")

    # print("\n=== ЭТАП 3: Parameter Recovery ===")
    # plt.close('all')
    # # ИСПРАВЛЕНИЕ 1: Передаем idata первым аргументом
    # recovery_data_df, recovery_metrics_df = hba_model.parameter_recovery(
    #     idata, # <-- Передаем загруженный Trace
    #     n_subjects=min(50, hba_model.n_subj), 
    #     save_plot_path="hot_wullhorst1_recovery_scatter.png"
    # )
    # recovery_data_df.to_csv("hot_wullhorst1_recovery_data.csv", index=False)
    # recovery_metrics_df.to_csv("hot_wullhorst1_recovery_metrics.csv", index=False)
    
    # print("\nМетрики Parameter Recovery:")
    # print(recovery_metrics_df)


    # -----------------------------------------------------------------
    # 2. МОДЕЛЬ: HOTModel2_HBA (Wullhorst Model 2)
    # -----------------------------------------------------------------
    # print("\n" + "#" * 50)
    # print("=== ЭТАП 2: HOTModel2_HBA (Wullhorst 2 + Prelec) ===")
    # print("#" * 50)
    
    # hot2_model = HOTModel2_HBA(hot_df, scale_factor=100.0)
    
    # # ЧТЕНИЕ TRACE (без переобучения)
    # # print("[*] Загрузка сохраненного trace: hot_wullhorst2_trace.nc")
    # # idata_2 = az.from_netcdf("hot_wullhorst2_trace.nc")

    # # 1. Распаковываем 3 переменные: idata_2, model_2, loo_2
    # idata_2, model_2, loo_2 = hot2_model.fit(draws=1500, tune=1000, chains=4)

    # # СОХРАНЕНИЕ TRACE (NETCDF ФАЙЛ)
    # az.to_netcdf(idata_2, "hot_wullhorst2_trace.nc")
    # print("[✓] Trace для Model 2 сохранен в hot_wullhorst2_trace.nc")
    # with open("hot_wullhorst2_looic.txt", "w") as f:
    #     f.write(str(loo_2))

    # save_hba_parameters(idata_2, hot2_model.subj_labels, ['rho', 'lam', 'delta', 'eta', 'beta'], "hot_wullhorst2_params.csv")

    # print("\n[*] Запуск PPC для Модели 2...")
    # plt.close('all')
    # # 2. Передаем model_2 вторым аргументом
    # ppc_metrics_df_2 = hot2_model.posterior_predictive_check(idata_2, model_2, save_plot_path="hot_wullhorst2_ppc_timecourse.png")
    # ppc_metrics_df_2.to_csv("hot_wullhorst2_ppc_metrics.csv", index=False)
    
    # print("\nСредние метрики PPC по выборке (Model 2):")
    # print(ppc_metrics_df_2[['R2', 'RMSE', 'MAE', 'MSD', 'Hit_Rate', 'ppp']].mean().round(3))

    # print("\n[*] Запуск Parameter Recovery для Модели 2...")
    # plt.close('all')
    # # ИСПРАВЛЕНИЕ 2: Передаем idata_2 первым аргументом
    # rec_data_df_2, rec_metrics_df_2 = hot2_model.parameter_recovery(
    #     idata_2, # <-- Передаем загруженный Trace
    #     n_subjects=min(50, hot2_model.n_subj), 
    #     save_plot_path="hot_wullhorst2_recovery_scatter.png"
    # )
    # rec_data_df_2.to_csv("hot_wullhorst2_recovery_data.csv", index=False)
    # rec_metrics_df_2.to_csv("hot_wullhorst2_recovery_metrics.csv", index=False)


    # -----------------------------------------------------------------
    # 3. МОДЕЛЬ: HaffkeHBAModel_Hot (Фиттинг, PPC и Parameter Recovery)
    # -----------------------------------------------------------------
    print("\n" + "#" * 50)
    print("=== ЭТАП 3: Haffke Model (Model 3) ===")
    print("#" * 50)
    
    print("[*] Подготовка данных и расчет конъюнктивных вероятностей...")
    haffke_df = prepare_haffke_input(hot_df)
    
    haffke_model = HaffkeHBAModel_Hot(haffke_df, scale_factor=100.0)
    
    # ЗАПУСК РЕАЛЬНОЙ ПОДГОНКИ МОДЕЛИ (ВМЕСТО ЧТЕНИЯ TRACE)
    print("[*] Запуск подгонки HBA NUTS для Haffke Model...")
    idata_3, model_3, loo_3 = haffke_model.fit(draws=1500, tune=1000, chains=4, target_accept=0.95)

    # СОХРАНЕНИЕ СВЕЖЕГО TRACE НА ДИСК (NETCDF ФАЙЛ)
    az.to_netcdf(idata_3, "hot_haffke_trace.nc")
    print("[✓] Trace для Haffke Model сохранен в hot_haffke_trace.nc")
    
    # Сохранение LOOIC метрики в файл
    with open("hot_haffke_looic.txt", "w") as f:
        f.write(str(loo_3))

    # Экспорт параметров
    save_hba_parameters(idata_3, haffke_model.subj_labels, ['alpha', 'lam', 'delta', 'eta', 'theta'], "hot_haffke_params.csv")

    print("\n[*] Запуск PPC для Haffke Model...")
    plt.close('all')
    # Передаем только что обученные idata_3 и model_3
    ppc_metrics_df_3 = haffke_model.posterior_predictive_check(idata_3, model=model_3, save_plot_path="hot_haffke_ppc_timecourse.png")
    ppc_metrics_df_3.to_csv("hot_haffke_ppc_metrics.csv", index=False)
    
    print("\nСредние метрики PPC по выборке (Haffke Model):")
    print(ppc_metrics_df_3[['R2', 'RMSE', 'MAE', 'MSD', 'Hit_Rate', 'ppp']].mean().round(3))

    print("\n[*] Запуск Эмпирического Parameter Recovery для Haffke Model...")
    plt.close('all')
    
    # Передаем обученный idata_3 для извлечения mu_* и sigma_* и симуляции агентов
    rec_data_df_3, rec_metrics_df_3 = haffke_model.parameter_recovery(
        idata_3, 
        n_subjects=min(50, haffke_model.n_subj), 
        trials_per_subj=48, 
        save_plot_path="hot_haffke_recovery_scatter.png"
    )
    
    rec_data_df_3.to_csv("hot_haffke_recovery_data.csv", index=False)
    rec_metrics_df_3.to_csv("hot_haffke_recovery_metrics.csv", index=False)
    
    print("\n" + "=" * 70)
    print("=== ВСЕ МОДЕЛИ CCT-HOT УСПЕШНО ЗАВЕРШЕНЫ ===")
    print("=" * 70)

if __name__ == '__main__':
    main()