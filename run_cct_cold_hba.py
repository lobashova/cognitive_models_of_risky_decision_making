import os
import pandas as pd
import numpy as np
import arviz as az

# Перевод Matplotlib в неинтерактивный режим для работы на суперкомпьютере (Headless)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Импорт ваших моделей
try:
    from models.wullhorst_model1_cold import COLDModelHBA
    from models.wullhorst_model2_cold import COLDModel2_HBA
    from models.haffke_model_cold import HaffkeColdModel_HBA
except ImportError as e:
    print(f"[!] Предупреждение при импорте моделей: {e}")

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

def main():
    print("=" * 70)
    print("=== ЗАПУСК ПАКЕТНОГО ВЫЧИСЛЕНИЯ CCT-COLD (HBA) ===")
    print("=" * 70)

    # 0. ЗАГРУЗКА И ПРОВЕРКА ДАННЫХ
    data_path = 'data/cold_data.csv' 
    if not os.path.exists(data_path):
        print(f"[!] Ошибка: Файл {data_path} не найден.")
        return

    cold_df = pd.read_csv(data_path)
    print(f"[✓] Данные успешно загружены. Уникальных участников: {cold_df['user_id'].nunique()}")

    # -----------------------------------------------------------------
    # 1. МОДЕЛЬ HBA (Wullhorst 1)
    # -----------------------------------------------------------------
    print("\n=== ЭТАП 1: Подгонка HBA (Wullhorst 1) ===")
    cold_hba = COLDModelHBA(cold_df, max_N=32, scale_factor=100.0)
    
    # ИСПРАВЛЕНИЕ: Распаковываем 3 значения
    idata_1, compiled_model_1, loo_1 = cold_hba.fit(draws=1500, tune=1000, chains=4)

    # НОВОЕ: СОХРАНЕНИЕ TRACE
    az.to_netcdf(idata_1, "cold_wullhorst1_trace.nc")
    print("[✓] Trace для Model 1 сохранен в cold_wullhorst1_trace.nc")

    # ИСПРАВЛЕНИЕ: .elpd_loo и .se вместо .loo и .loo_se
    print(f"\n[✓] Модель 1 подогнана. LOOIC = {loo_1.elpd_loo:.2f} (SE: {loo_1.se:.2f})")
    with open("cold_wullhorst1_looic.txt", "w") as f:
        f.write(str(loo_1))

    save_hba_parameters(idata_1, cold_hba.subj_labels, ['rho', 'lambd', 'beta'], "cold_wullhorst1_params.csv")

    print("\n[*] Запуск Posterior Predictive Check (PPC) Model 1...")
    plt.close('all')
    ppc_metrics_df = cold_hba.predictive_check(idata_1, compiled_model_1, save_plot_path="cold_wullhorst1_ppc_timecourse.png")
    ppc_metrics_df.to_csv('cold_wullhorst1_ppc_metrics.csv', index=False)
    
    print("\n[*] Запуск Parameter Recovery Model 1...")
    plt.close('all')
    rec_data_df, rec_metrics_df = cold_hba.parameter_recovery(n_subjects=min(50, cold_hba.n_subj), 
                                                              save_plot_path="cold_wullhorst1_recovery_scatter.png")
    rec_data_df.to_csv("cold_wullhorst1_recovery_data.csv", index=False)
    rec_metrics_df.to_csv("cold_wullhorst1_recovery_metrics.csv", index=False)


    # -----------------------------------------------------------------
    # 2. МОДЕЛЬ: COLDModel2_HBA (Wullhorst Model 2)
    # -----------------------------------------------------------------
    print("\n" + "#" * 50)
    print("=== ЭТАП 2: COLDModel2_HBA (Wullhorst 2) ===")
    print("#" * 50)
    
    try:
        cold_hba_2 = COLDModel2_HBA(cold_df, max_N=32, scale_factor=100.0)
        print("[*] Иерархическая подгонка модели 2 на всем датасете (MCMC)...")
        
        # ИСПРАВЛЕНИЕ: Ожидаем 3 значения (idata, model, loo)
        idata_2, compiled_model_2, loo_2 = cold_hba_2.fit(tune=1500, draws=1500, chains=4)

        # НОВОЕ: СОХРАНЕНИЕ TRACE
        az.to_netcdf(idata_2, "cold_wullhorst2_trace.nc")
        print("[✓] Trace для Model 2 сохранен в cold_wullhorst2_trace.nc")

        # ИСПРАВЛЕНИЕ: .elpd_loo и .se
        print(f"\n[✓] Модель 2 подогнана. LOOIC = {loo_2.elpd_loo:.2f} (SE: {loo_2.se:.2f})")
        with open("cold_wullhorst2_looic.txt", "w") as f:
            f.write(str(loo_2))

        save_hba_parameters(idata_2, cold_hba_2.subj_labels, ['rho', 'lambd', 'delta', 'eta', 'beta'], "cold_wullhorst2_params.csv")

        print("\n[*] Запуск Posterior Predictive Check (PPC) Model 2...")
        plt.close('all')
        ppc_metrics_df_2 = cold_hba_2.predictive_check(idata_2, compiled_model_2, save_plot_path="cold_wullhorst2_ppc_timecourse.png")
        ppc_metrics_df_2.to_csv('cold_wullhorst2_ppc_metrics.csv', index=False)

        print("\n[*] Запуск Parameter Recovery Model 2...")
        plt.close('all')
        rec_data_df_2, rec_metrics_df_2 = cold_hba_2.parameter_recovery(n_subjects=min(50, cold_hba_2.n_subj), 
                                                                        save_plot_path="cold_wullhorst2_recovery_scatter.png")
        rec_data_df_2.to_csv("cold_wullhorst2_recovery_data.csv", index=False)
        rec_metrics_df_2.to_csv("cold_wullhorst2_recovery_metrics.csv", index=False)
        
    except Exception as e:
        print(f"[!!!] Ошибка в ЭТАПЕ 2 (Wullhorst 2): {e}")


    # -----------------------------------------------------------------
    # 3. МОДЕЛЬ: HaffkeColdModel_HBA
    # -----------------------------------------------------------------
    print("\n" + "#" * 50)
    print("=== ЭТАП 3: HaffkeColdModel_HBA ===")
    print("#" * 50)
    
    try:
        print("[*] Подготовка специфичного датафрейма для модели Haffke...")
        cold_df_prepared = cold_df.copy()
        cold_df_prepared['cards_left'] = 32
        cold_df_prepared['gains_left'] = 32 - cold_df_prepared['loss_cards']
        cold_df_prepared['choice'] = cold_df_prepared['num_cards']
        cold_df_prepared['options'] = [list(range(0, 33)) for _ in range(len(cold_df_prepared))]

        print("\n[*] Иерархическая подгонка модели Haffke (MCMC)...")
        haffke_hba = HaffkeColdModel_HBA(cold_df_prepared, scale_factor=100.0)
        
        # ИСПРАВЛЕНИЕ: Распаковываем 3 значения
        idata_3, compiled_model_3, loo_3 = haffke_hba.fit(draws=1500, tune=1000, chains=4)
        
        # НОВОЕ: СОХРАНЕНИЕ TRACE
        az.to_netcdf(idata_3, "cold_haffke_trace.nc")
        print("[✓] Trace для Model 3 (Haffke) сохранен в cold_haffke_trace.nc")

        # ИСПРАВЛЕНИЕ: .elpd_loo и .se
        print(f"\n[✓] Модель Haffke подогнана. LOOIC = {loo_3.elpd_loo:.2f} (SE: {loo_3.se:.2f})")
        with open("cold_haffke_looic.txt", "w") as f:
            f.write(str(loo_3))

        save_hba_parameters(idata_3, haffke_hba.user_ids, 
                            ['alpha', 'lambda', 'delta', 'eta', 'theta', 'epsilon'], 
                            "cold_haffke_params.csv")

        print("\n[*] Запуск Posterior Predictive Check (PPC)...")
        plt.close('all')
        ppc_metrics_df_3 = haffke_hba.posterior_predictive_check(idata_3, compiled_model_3, save_plot_path="cold_haffke_ppc_timecourse.png")
        ppc_metrics_df_3.to_csv('cold_haffke_ppc_metrics.csv', index=False)

        print("\n[*] Запуск Parameter Recovery...")
        plt.close('all')
        rec_data_df_3, rec_metrics_df_3 = haffke_hba.parameter_recovery(n_subjects=min(50, haffke_hba.n_users), 
                                                                        save_plot_path="cold_haffke_recovery_scatter.png")
        rec_data_df_3.to_csv("cold_haffke_recovery_data.csv", index=False)
        rec_metrics_df_3.to_csv("cold_haffke_recovery_metrics.csv", index=False)

    except Exception as e:
        print(f"[!!!] Ошибка в ЭТАПЕ 3 (Haffke): {e}")

    print("\n" + "=" * 70)
    print("=== ВСЕ МОДЕЛИ CCT-COLD УСПЕШНО ЗАВЕРШЕНЫ ===")
    print("=" * 70)

if __name__ == '__main__':
    main()